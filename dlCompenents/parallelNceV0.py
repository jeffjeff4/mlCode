import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import os


def setup(rank, world_size):
    """Initializes the distributed process group (for CPU)."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'  # Any free port
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    print(f"Initialized process {rank}/{world_size} on 'gloo' backend.")


def cleanup():
    """Destroys the distributed process group."""
    dist.destroy_process_group()


def build_dataset(corpus):
    """Builds vocab, unigram counts, and (context, target) pairs."""
    vocab = {}
    idx = 0
    unigram_counts = {}

    words = corpus.split()
    for word in words:
        if word not in vocab:
            vocab[word] = idx
            idx += 1
        unigram_counts[vocab[word]] = unigram_counts.get(vocab[word], 0) + 1

    # Create (context, target) pairs (1-word context)
    pairs = []
    indices = [vocab[w] for w in words]
    for i in range(1, len(indices)):
        context_word = indices[i - 1]
        target_word = indices[i]
        pairs.append((context_word, target_word))

    vocab_size = len(vocab)
    return pairs, vocab_size, unigram_counts


class SimpleWordDataset(Dataset):
    """Simple dataset for (context, target) pairs."""

    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        context, target = self.pairs[idx]
        # Return 1-element tensors (shape [1]) instead of scalars (shape [])
        return torch.tensor([context]).long(), torch.tensor([target]).long()


class NCEModel(nn.Module):
    """
    A simple Word2Vec-style model where the 'forward' pass
    is the NCE loss (Negative Sampling) calculation.
    """

    def __init__(self, vocab_size, embed_dim):
        super(NCEModel, self).__init__()
        # Input words (context)
        self.input_embeddings = nn.Embedding(vocab_size, embed_dim)
        # Output words (target/noise)
        self.output_embeddings = nn.Embedding(vocab_size, embed_dim)

    def forward(self, input_word, true_word, noise_words):
        """
        Calculates the NCE loss (Negative Sampling).
        - input_word: [B, 1]
        - true_word: [B, 1]
        - noise_words: [B, k] (k = num_noise_samples)
        """
        batch_size = input_word.shape[0]

        # Get embeddings
        # [B, 1, D] -> [B, D]
        input_vec = self.input_embeddings(input_word).squeeze(1)
        # [B, 1, D] -> [B, D]
        true_vec = self.output_embeddings(true_word).squeeze(1)
        # [B, k, D]
        noise_vecs = self.output_embeddings(noise_words)

        # 1. Calculate score for the one TRUE pair
        # (input_vec * true_vec).sum(dim=1) -> [B]
        # We use keepdim=True to make it [B, 1]
        true_logits = (input_vec * true_vec).sum(dim=1, keepdim=True)

        # 2. Calculate scores for the 'k' NOISE pairs
        # We use batch matrix multiply (bmm) for efficiency
        # [B, k, D] @ [B, D, 1] -> [B, k, 1]
        # .squeeze(2) -> [B, k]
        noise_logits = torch.bmm(noise_vecs, input_vec.unsqueeze(2)).squeeze(2)

        # 3. Concatenate true score and noise scores
        # [B, 1] cat [B, k] -> [B, 1+k]
        all_logits = torch.cat([true_logits, noise_logits], dim=1)

        # 4. Create labels: 1.0 for true, 0.0 for noise
        # *** FIX ***: Create labels on the same device as the data
        true_labels = torch.ones(batch_size, 1, device=input_word.device)
        noise_labels = torch.zeros(batch_size, noise_logits.shape[1], device=input_word.device)
        all_labels = torch.cat([true_labels, noise_labels], dim=1)

        # 5. Calculate Binary Cross Entropy loss
        # This is the NCE loss
        loss = F.binary_cross_entropy_with_logits(all_logits, all_labels)
        return loss


def main_worker(rank, world_size):
    """The main worker function for each DDP process."""
    setup(rank, world_size)

    # *** FIX ***: Explicitly define 'cpu' as the device
    # On Apple Silicon, 'rank' (e.g., 0) can be mistaken for the MPS device.
    device = 'cpu'

    # --- 1. Generate Simple Test Dataset ---
    corpus = ("The quick brown fox jumps over the lazy dog "
              "A quick brown dog jumps over the lazy fox "
              "The dog is lazy and the fox is quick "
              "The fox and the dog are friends")

    pairs, vocab_size, unigram_counts = build_dataset(corpus)
    dataset = SimpleWordDataset(pairs)

    # Create the noise distribution (Unigram^0.75)
    sorted_counts = [unigram_counts.get(i, 0) for i in range(vocab_size)]
    noise_dist = torch.tensor(sorted_counts, dtype=torch.float).pow(0.75)

    # --- 2. Create Model and Dataloader ---
    # Move model to the CPU device
    model = NCEModel(vocab_size=vocab_size, embed_dim=32).to(device)

    # Wrap the model in DDP for CPU
    model = DDP(model, device_ids=None, output_device=None)

    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # DistributedSampler ensures each process gets a unique slice of data
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)

    loader = DataLoader(dataset, batch_size=8, sampler=sampler)

    # --- 3. Training Loop ---
    num_epochs = 20
    k_noise_samples = 5

    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)
        total_loss = 0.0

        for context, target in loader:
            # *** FIX ***: Move data to the 'cpu' device
            context = context.to(device)
            target = target.to(device)

            batch_size = context.shape[0]

            # Generate noise samples for this batch
            noise_words = torch.multinomial(
                noise_dist,
                num_samples=batch_size * k_noise_samples,
                replacement=True
            ).view(batch_size, k_noise_samples).to(device)  # *** FIX ***

            optimizer.zero_grad()
            loss = model(context, target, noise_words)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if rank == 0:
            print(f"Epoch {epoch} | Avg NCE Loss: {total_loss / len(loader):.6f}")

    # --- 4. Evaluation ---
    dist.barrier()
    if rank == 0:
        print("\n--- Evaluation ---")

    # *** FIX ***: Create eval tensors on the 'cpu' device
    eval_contexts = torch.zeros(loader.batch_size, 1).long().to(device)
    eval_targets = torch.zeros(loader.batch_size, 1).long().to(device)
    eval_noise = torch.zeros(loader.batch_size, k_noise_samples).long().to(device)

    # Rank 0 loads the data
    if rank == 0:
        try:
            c, t = next(iter(loader))
            n = torch.multinomial(
                noise_dist,
                c.shape[0] * k_noise_samples,
                replacement=True
            ).view(c.shape[0], k_noise_samples)

            # .copy_() works fine if all tensors are on the same device
            eval_contexts.copy_(c)
            eval_targets.copy_(t)
            eval_noise.copy_(n)
        except StopIteration:
            if rank == 0: print("Eval loader empty.")

    # Broadcast data from Rank 0 to all other ranks
    dist.broadcast(eval_contexts, src=0)
    dist.broadcast(eval_targets, src=0)
    dist.broadcast(eval_noise, src=0)

    with torch.no_grad():
        # All ranks run the model
        eval_loss = model(eval_contexts, eval_targets, eval_noise)

    if rank == 0:
        print(f"Final Eval NCE Loss: {eval_loss.item():.6f}")

    cleanup()


if __name__ == "__main__":
    world_size = 2

    print(f"Starting {world_size} processes for DDP on CPU...")
    mp.spawn(
        main_worker,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )
    print("All processes finished.")
