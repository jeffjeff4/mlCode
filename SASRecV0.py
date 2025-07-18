import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm


# SASRec Model
class SASRec(nn.Module):
    def __init__(self, num_items, embed_dim=64, num_heads=4, num_layers=2, dropout=0.1, max_seq_len=50):
        super(SASRec, self).__init__()
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.item_emb = nn.Embedding(num_items + 1, embed_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(embed_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.item_emb.weight, mean=0, std=0.01)
        nn.init.normal_(self.pos_emb.weight, mean=0, std=0.01)

    def forward(self, seq, return_all=False):
        batch_size, seq_len = seq.size()
        item_emb = self.item_emb(seq)
        pos = torch.arange(seq_len, device=seq.device).unsqueeze(0).expand(batch_size, seq_len)
        pos_emb = self.pos_emb(pos)
        x = item_emb + pos_emb
        x = self.dropout(x)
        padding_mask = (seq == 0)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=seq.device), diagonal=1).bool()
        x = self.transformer(
            src=x,
            mask=causal_mask,
            src_key_padding_mask=padding_mask
        )
        x = self.ln(x)
        if return_all:
            scores = torch.matmul(x, self.item_emb.weight.transpose(0, 1))
            return scores
        else:
            last_emb = x[:, -1, :]
            scores = torch.matmul(last_emb, self.item_emb.weight.transpose(0, 1))
            return scores

    def predict(self, seq):
        self.eval()
        with torch.no_grad():
            scores = self.forward(seq)
        return scores


# Data Generation
def generate_synthetic_data(num_users=1000, num_items=1000, max_seq_len=50, min_seq_len=5):
    data = []
    for user_id in range(1, num_users + 1):
        seq_len = np.random.randint(min_seq_len, max_seq_len + 1)
        seq = np.random.randint(1, num_items + 1, size=seq_len)
        data.append([user_id, seq.tolist()])
    df = pd.DataFrame(data, columns=['user_id', 'sequence'])
    return df


class SequenceDataset(Dataset):
    def __init__(self, df, max_seq_len=50):
        self.df = df
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        seq = self.df.iloc[idx]['sequence']
        seq = seq[-self.max_seq_len:]
        padded_seq = [0] * (self.max_seq_len - len(seq)) + seq
        input_seq = padded_seq[:-1]
        target = padded_seq[-1]
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target, dtype=torch.long)


# Training
def train_sasrec(model, train_loader, num_epochs=5, lr=0.001, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        for input_seq, target in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
            input_seq, target = input_seq.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(input_seq)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}')

    torch.save(model.state_dict(), 'sasrec_model.pth')
    return model


# Evaluation
def evaluate_sasrec(model, test_loader, k_values=[5, 10], device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = model.to(device)
    model.eval()
    hr = {k: 0 for k in k_values}
    ndcg = {k: 0 for k in k_values}
    total = 0

    with torch.no_grad():
        for input_seq, target in test_loader:
            input_seq, target = input_seq.to(device), target.to(device)
            scores = model(input_seq)
            for k in k_values:
                _, top_k_indices = torch.topk(scores, k, dim=1)
                hits = (top_k_indices == target.unsqueeze(1)).sum(dim=1)
                hr[k] += hits.sum().item()
                for i in range(len(target)):
                    if target[i] in top_k_indices[i]:
                        rank = (top_k_indices[i] == target[i]).nonzero(as_tuple=True)[0].item()
                        ndcg[k] += 1.0 / np.log2(rank + 2)
            total += len(target)

    for k in k_values:
        hr[k] = hr[k] / total
        ndcg[k] = ndcg[k] / total
        print(f'HR@{k}: {hr[k]:.4f}, NDCG@{k}: {ndcg[k]:.4f}')


# Run the pipeline
if __name__ == "__main__":
    # Generate data
    np.random.seed(42)
    data_df = generate_synthetic_data(num_users=1000, num_items=1000)
    train_df = data_df.sample(frac=0.8, random_state=42)
    test_df = data_df.drop(train_df.index)
    train_dataset = SequenceDataset(train_df)
    test_dataset = SequenceDataset(test_df)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # Initialize model
    model = SASRec(num_items=1000, embed_dim=64, num_heads=4, num_layers=2, max_seq_len=50)

    # Train
    model = train_sasrec(model, train_loader, num_epochs=5)

    # Evaluate
    evaluate_sasrec(model, test_loader, k_values=[5, 10])