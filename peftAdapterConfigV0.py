import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, AdaLoraConfig
from torch.utils.data import Dataset, DataLoader
import numpy as np


# Custom dataset for recommendation descriptions
class RecommendationDataset(Dataset):
    def __init__(self, inputs, outputs, tokenizer, max_length=128):
        self.inputs = inputs
        self.outputs = outputs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_text = self.inputs[idx]
        output_text = self.outputs[idx]

        # Tokenize input and output
        input_encoding = self.tokenizer(
            input_text, max_length=self.max_length, padding="max_length",
            truncation=True, return_tensors="pt"
        )
        output_encoding = self.tokenizer(
            output_text, max_length=self.max_length, padding="max_length",
            truncation=True, return_tensors="pt"
        )

        return {
            "input_ids": input_encoding["input_ids"].squeeze(),
            "attention_mask": input_encoding["attention_mask"].squeeze(),
            "labels": output_encoding["input_ids"].squeeze()
        }


# Main function to run adapter-based fine-tuning
def main():
    # Set device (use CUDA if available, relevant for NVIDIA GPUs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load pre-trained model and tokenizer
    model_name = "distilgpt2"  # Lightweight model for demo; replace with LLaMA for enterprise
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Set padding token

    # Configure AdaLoRA with total_step
    num_epochs = 3
    batch_size = 2
    dataset_size = 3  # Number of examples in synthetic dataset
    total_steps = int(num_epochs * (dataset_size / batch_size) + 1)  # Round up
    adapter_config = AdaLoraConfig(
        r=16,  # Initial rank for low-rank adaptation
        lora_alpha=32,  # Scaling factor
        target_modules=["c_attn", "c_fc"],  # Target attention and feed-forward layers
        lora_dropout=0.1,
        task_type="CAUSAL_LM",
        total_step=total_steps  # Required for AdaLoRA
    )
    model = get_peft_model(model, adapter_config)
    model.to(device)

    # Print trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params} ({trainable_params / total_params * 100:.2f}%)")

    # Synthetic dataset for YouTube recommendation
    inputs = [
        "User likes cooking videos.",
        "User watched Italian recipes.",
        "User prefers short tutorials."
    ]
    outputs = [
        "I recommend 'Pasta Tutorial' (youtube.com/101) for its clear instructions.",
        "Try 'Italian Cooking Masterclass' (youtube.com/301) for authentic recipes.",
        "Check out 'Quick Cooking Tips' (youtube.com/401) for concise tutorials."
    ]
    dataset = RecommendationDataset(inputs, outputs, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training setup
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

    # Save adapter weights
    model.save_pretrained("adapter_finetuned_model")
    tokenizer.save_pretrained("adapter_finetuned_model")

    # Inference example
    model.eval()
    test_input = "User likes cooking videos."
    inputs = tokenizer(test_input, return_tensors="pt").to(device)
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=50,
        num_beams=3,
        no_repeat_ngram_size=2
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nInference Example:")
    print(f"Input: {test_input}")
    print(f"Output: {generated_text}")


if __name__ == "__main__":
    main()