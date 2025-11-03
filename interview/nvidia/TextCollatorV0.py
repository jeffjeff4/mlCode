####This file explains what the collate_fn is doing and why it's a critical concept.
####
####1. Data Structure Manipulation (The "What")
####
####The collate_fn's job is to transform a list of samples into a "batch".
####
####Input from DataLoader: A list of strings, e.g.,
####["This is a short sentence.", "A third one."]
####(This is batch_size=2 items from our Dataset.__getitem__)
####
####Tokenization: The tokenizer converts these strings into lists of numbers (Token IDs).
####
####"This is a short sentence." -> [101, 2023, 2003, 1037, 2460, 6251, 1012, 102] (Length 8)
####
####"A third one." -> [101, 1037, 2353, 2028, 1012, 102] (Length 6)
####
####Output (The Batch): We need to combine these into a single rectangular tensor. We can't just stack them because they have different lengths. This is where padding comes in. The tokenizer creates two tensors:
####
####input_ids: The token IDs, with a pad_token (usually 0 for BERT) added to the shorter sequences to make them all the same length as the longest one in the batch.
####
####[[101, 2023, 2003, 1037, 2460, 6251, 1012, 102],  <-- Length 8
#### [101, 1037, 2353, 2028, 1012, 102,    0,   0]]   <-- Padded to length 8
####
####
####This is now a (2, 8) tensor.
####
####attention_mask: This is a tensor of 1s and 0s that tells the model which tokens to pay attention to and which to ignore. It's the most important part of padding.
####
####[[1, 1, 1, 1, 1, 1, 1, 1],
#### [1, 1, 1, 1, 1, 1, 0, 0]]
####
####
####The model's self-attention mechanism is designed to see this mask and will completely ignore the tokens where the mask is 0. This prevents the model from getting confused by the fake "padding" tokens.
####
####The final output is a dictionary of these tensors: {'input_ids': ..., 'attention_mask': ...}.
####
####2. Efficiency: Dynamic Padding vs. Static Padding
####
####This is the key efficiency test.
####
####Static Padding (Bad): You could configure the tokenizer to pad all samples to the model's absolute maximum length, e.g., max_length=512.
####
####Our 8-token and 6-token sentences would both be padded to 512 tokens.
####
####This creates a (2, 512) tensor, where >98% of the data is just padding.
####
####This is enormously wasteful. The GPU will spend most of its time processing padding tokens that are just going to be ignored anyway.
####
####Dynamic Padding (Good): This is what we did by setting padding="longest".
####
####The tokenizer looks at the current batch only.
####
####It finds the longest sequence in that batch (length 8).
####
####It pads all other sequences in that batch to that length.
####
####This creates a (2, 8) tensor.
####
####This is the most efficient way to batch, as it creates the smallest possible rectangular tensor for any given batch, minimizing wasted computation.



import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import List, Dict, Any


# --- Part A: Dummy Dataset for Demonstration ---
# This is a simple in-memory version of the TextCorpusDataset
# from the previous question. It just returns strings from a list.
class DummyTextCorpusDataset(Dataset):
    def __init__(self, data: List[str]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]  # Returns a single raw text string


# --- Part B: The Collate Function (as a Class) ---
# We use a class here to cleanly manage the tokenizer,
# which needs to be initialized. An object of this class
# is "callable" (i.e., it can be used as a function).

class TextCollator:
    """
    A callable class that acts as the `collate_fn` for a DataLoader.
    It takes a list of raw text strings and returns a batch of
    padded tensors.
    """

    def __init__(self, tokenizer_name_or_path: str, max_length: int = 512):
        """
        Initializes the collator.

        Args:
            tokenizer_name_or_path (str): The name of the Hugging Face
                                          tokenizer to use (e.g., "bert-base-uncased").
            max_length (int): The maximum sequence length to truncate to.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        self.max_length = max_length

        # Add a padding token if the tokenizer doesn't have one (e.g., GPT-2)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"Tokenizer missing pad_token, setting to eos_token: {self.tokenizer.pad_token}")

    def __call__(self, batch: List[str]) -> Dict[str, torch.Tensor]:
        """
        This is the function that DataLoader will call.

        Args:
            batch (List[str]): A list of raw text strings,
                               e.g., ["This is sentence 1.", "A shorter one."]

        Returns:
            Dict[str, torch.Tensor]: A dictionary of tensors ready
                                     for the model.
        """

        # The Hugging Face tokenizer does all the work for us in one go!
        # - padding="longest": Pads to the longest sequence in this *batch*.
        # - truncation=True: Truncates sequences longer than `max_length`.
        # - return_tensors="pt": Returns PyTorch tensors.
        tokenized_batch = self.tokenizer(
            batch,
            padding="longest",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return tokenized_batch


# --- Example Usage ---
if __name__ == "__main__":
    # 1. Define our dummy data with varying lengths
    dummy_data = [
        "This is a short sentence.",
        "This is a much, much longer sentence that will require padding for the others.",
        "A third one.",
        "The quick brown fox."
    ]

    # 2. Initialize the Dataset
    dataset = DummyTextCorpusDataset(dummy_data)

    # 3. Initialize the Collator
    # We'll use "bert-base-uncased" as our example tokenizer
    collator = TextCollator(tokenizer_name_or_path="bert-base-uncased")

    # 4. Initialize the DataLoader
    # We pass the collator to the `collate_fn` argument.
    data_loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,  # Set to True for training
        collate_fn=collator
    )

    print(f"--- Fetching {len(data_loader)} batches... ---")

    for i, batch in enumerate(data_loader):
        print(f"\n--- BATCH {i + 1} ---")
        print("Type:", type(batch))
        print("Keys:", batch.keys())

        print("\ninput_ids (token IDs):")
        print(batch['input_ids'])
        print("Shape:", batch['input_ids'].shape)  # [batch_size, max_len_in_batch]

        print("\nattention_mask (1s for real tokens, 0s for padding):")
        print(batch['attention_mask'])
        print("Shape:", batch['attention_mask'].shape)

    print("\n--- Demonstration complete. ---")
    print("Note how 'input_ids' are padded with 0 (BERT's pad_token_id).")
    print("Note how 'attention_mask' has 1s for real tokens and 0s for padding.")
    print(
        f"In Batch 1, the longest sentence had {batch['input_ids'].shape[1]} tokens, so both sentences were padded to that length.")
