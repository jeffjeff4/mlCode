import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification, DistilBertForSequenceClassification
#from transformers import AdamW
from torch.optim import AdamW
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and models
teacher_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2).to(device)
student_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2).to(device)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Hyperparameters
T = 2.0  # Temperature
alpha = 0.5  # Weight for hard vs soft loss
epochs = 3
batch_size = 16
learning_rate = 2e-5

# Load dataset (small binary classification task for demo)
dataset = load_dataset("imdb", split="train[:2%]").train_test_split(test_size=0.5)
tokenized = dataset.map(lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=128), batched=True)
tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

train_loader = DataLoader(tokenized['train'], batch_size=batch_size, shuffle=True)
test_loader = DataLoader(tokenized['test'], batch_size=batch_size)

# Loss functions
kl_loss_fn = nn.KLDivLoss(reduction="batchmean")
ce_loss_fn = nn.CrossEntropyLoss()

optimizer = AdamW(student_model.parameters(), lr=learning_rate)

# 🧪 Training
for epoch in range(epochs):
    student_model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        with torch.no_grad():
            teacher_outputs = teacher_model(input_ids=input_ids, attention_mask=attention_mask)
            soft_targets = nn.functional.softmax(teacher_outputs.logits / T, dim=-1)

        student_outputs = student_model(input_ids=input_ids, attention_mask=attention_mask)
        student_logits = student_outputs.logits

        # distillation loss
        loss_soft = kl_loss_fn(
            nn.functional.log_softmax(student_logits / T, dim=-1),
            soft_targets
        ) * (T * T)
        loss_hard = ce_loss_fn(student_logits, labels)
        loss = alpha * loss_hard + (1 - alpha) * loss_soft

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"[Epoch {epoch+1}] Loss: {total_loss/len(train_loader):.4f}")

# 🚀 Inference
student_model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = student_model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=-1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

acc = accuracy_score(all_labels, all_preds)
print(f"[Student Accuracy] {acc:.4f}")
