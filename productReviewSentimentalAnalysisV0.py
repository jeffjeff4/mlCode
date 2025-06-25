import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.optim import AdamW  # Updated import location
from transformers import BertTokenizer, BertForSequenceClassification

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Generate synthetic dataset
num_samples = 1000
products = [
    "Wireless Bluetooth Headphones",
    "Stainless Steel Water Bottle",
    "Organic Cotton T-Shirt",
    "Smart Fitness Tracker",
    "Ceramic Coffee Mug",
    "LED Desk Lamp",
    "Yoga Mat with Carry Strap",
    "Portable Phone Charger",
    "Leather Wallet",
    "Digital Kitchen Scale"
]

comments = [
    "I love this product, works perfectly!",
    "Poor quality, broke after one week",
    "Decent but overpriced for what you get",
    "Absolutely fantastic, would buy again",
    "Not worth the money, very disappointed",
    "Better than I expected, great value",
    "Terrible experience, never buying again",
    "Good product with minor flaws",
    "Excellent quality and fast shipping",
    "Mediocre at best, does the job"
]

# Generate synthetic data
data = []
for _ in range(num_samples):
    product = np.random.choice(products)
    price = round(np.random.uniform(5, 200), 2)
    ranking = np.random.randint(1, 6)
    clicks = np.random.randint(0, 1000)
    comment = np.random.choice(comments)

    # Determine sentiment label (1=positive, 0=negative) based on comment
    positive_keywords = ['love', 'fantastic', 'great', 'excellent', 'perfectly', 'better']
    negative_keywords = ['poor', 'terrible', 'disappointed', 'never', 'mediocre']

    label = 0  # default negative
    if any(word in comment.lower() for word in positive_keywords):
        label = 1
    elif any(word in comment.lower() for word in negative_keywords):
        label = 0
    else:
        label = 1 if ranking >= 3 else 0  # fallback to ranking

    data.append([product, price, ranking, clicks, comment, label])

# Create DataFrame
df = pd.DataFrame(data, columns=['product_title', 'price', 'ranking', 'clicks', 'comment', 'sentiment'])

# Split dataset
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)


class ProductReviewDataset(Dataset):
    def __init__(self, comments, labels, tokenizer, max_len):
        self.comments = comments
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, item):
        comment = str(self.comments[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            comment,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'comment_text': comment,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# Initialize BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Create datasets
MAX_LEN = 64
BATCH_SIZE = 16

train_dataset = ProductReviewDataset(
    comments=train_df.comment.to_numpy(),
    labels=train_df.sentiment.to_numpy(),
    tokenizer=tokenizer,
    max_len=MAX_LEN
)

test_dataset = ProductReviewDataset(
    comments=test_df.comment.to_numpy(),
    labels=test_df.sentiment.to_numpy(),
    tokenizer=tokenizer,
    max_len=MAX_LEN
)

# Create data loaders
train_data_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_data_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE
)


class SentimentClassifier(torch.nn.Module):
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=n_classes
        )

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs


# Initialize model
model = SentimentClassifier(n_classes=2)
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Training setup
EPOCHS = 3
optimizer = AdamW(model.parameters(), lr=2e-5)
loss_fn = torch.nn.CrossEntropyLoss().to('cuda' if torch.cuda.is_available() else 'cpu')


# Training loop
def train_epoch(model, data_loader, optimizer, device):
    model = model.train()
    losses = []

    for batch in data_loader:
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        losses.append(loss.item())

        loss.backward()
        optimizer.step()

    return np.mean(losses)


# Evaluation function
def eval_model(model, data_loader, device):
    model = model.eval()
    correct_predictions = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            _, preds = torch.max(outputs.logits, dim=1)
            correct_predictions += torch.sum(preds == labels)

    return correct_predictions.double() / len(data_loader.dataset)


# Train the model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)

    train_loss = train_epoch(model, train_data_loader, optimizer, device)
    print(f'Train loss: {train_loss}')

    val_acc = eval_model(model, test_data_loader, device)
    print(f'Validation accuracy: {val_acc}\n')


def predict_sentiment(text, model, tokenizer, device, max_len=64):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

    _, prediction = torch.max(outputs.logits, dim=1)

    return 'Positive' if prediction == 1 else 'Negative'


# Example predictions
test_comments = [
    "This product is amazing!",
    "Worst purchase ever",
    "It's okay, nothing special",
    "Highly recommended to all my friends",
    "Complete waste of money"
]

for comment in test_comments:
    print(f"Comment: {comment}")
    print(f"Predicted sentiment: {predict_sentiment(comment, model, tokenizer, device)}\n")

