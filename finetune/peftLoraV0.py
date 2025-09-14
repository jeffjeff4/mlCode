import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig
from torch.utils.data import Dataset, DataLoader
from nltk.translate.bleu_score import sentence_bleu
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
            "labels": output_encoding["input_ids"].squeeze(),
            "output_text": output_text  # For evaluation
        }


# Evaluation function using BLEU score
def evaluate_bleu(model, tokenizer, eval_dataloader, device):
    model.eval()
    bleu_scores = []

    with torch.no_grad():
        for batch in eval_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            reference_text = batch["output_text"]

            # Generate predictions with max_new_tokens
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=50,  # Generate up to 50 new tokens
                num_beams=3,
                no_repeat_ngram_size=2
            )

            # Decode predictions and references
            for i in range(len(outputs)):
                pred_text = tokenizer.decode(outputs[i], skip_special_tokens=True)
                ref_text = reference_text[i]
                # Compute BLEU score (split into words for BLEU)
                bleu = sentence_bleu([ref_text.split()], pred_text.split())
                bleu_scores.append(bleu)

    return np.mean(bleu_scores)


# Main function for LoRA fine-tuning
def main():
    # Set device (use CUDA if available, relevant for NVIDIA GPUs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load pre-trained model and tokenizer
    model_name = "distilgpt2"  # Lightweight model for demo; replace with LLaMA for enterprise
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Set padding token

    # Configure LoRA
    lora_config = LoraConfig(
        r=8,  # Rank of low-rank matrices
        lora_alpha=16,  # Scaling factor
        target_modules=["c_attn"],  # Apply LoRA to attention layers
        lora_dropout=0.1,
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.to(device)

    # Print trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params} ({trainable_params / total_params * 100:.2f}%)")

    # Synthetic dataset for YouTube recommendation
    train_inputs = [
        "User likes cooking videos.",
        "User watched Italian recipes.",
        "User prefers short tutorials.",
        "User enjoys tech reviews."
    ]
    train_outputs = [
        "I recommend 'Pasta Tutorial' (youtube.com/101) for its clear instructions.",
        "Try 'Italian Cooking Masterclass' (youtube.com/301) for authentic recipes.",
        "Check out 'Quick Cooking Tips' (youtube.com/401) for concise tutorials.",
        "I suggest 'Tech Gadgets Review' (youtube.com/501) for in-depth analysis."
    ]
    eval_inputs = [
        "User likes gaming videos.",
        "User watched travel vlogs."
    ]
    eval_outputs = [
        "I recommend 'Gaming Walkthrough' (youtube.com/601) for its detailed gameplay.",
        "Try 'World Travel Adventures' (youtube.com/701) for stunning visuals."
    ]

    # Create datasets
    train_dataset = RecommendationDataset(train_inputs, train_outputs, tokenizer)
    eval_dataset = RecommendationDataset(eval_inputs, eval_outputs, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=2)

    # Training setup
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    num_epochs = 5

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_dataloader)
        # Evaluate on validation set
        bleu_score = evaluate_bleu(model, tokenizer, eval_dataloader, device)
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}, BLEU Score: {bleu_score:.4f}")

    # Save LoRA weights
    model.save_pretrained("lora_finetuned_model")
    tokenizer.save_pretrained("lora_finetuned_model")

    # Inference example
    model.eval()
    test_input = "User likes gaming videos."
    inputs = tokenizer(test_input, return_tensors="pt").to(device)
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=50,  # Generate up to 50 new tokens
        num_beams=3,
        no_repeat_ngram_size=2
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nInference Example:")
    print(f"Input: {test_input}")
    print(f"Output: {generated_text}")


if __name__ == "__main__":
    main()