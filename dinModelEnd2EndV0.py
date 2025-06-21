##import pandas as pd
##import numpy as np
##import torch
##import torch.nn as nn
##import torch.optim as optim
##from torch.utils.data import Dataset, DataLoader
##from sklearn.model_selection import train_test_split
##from sklearn.metrics import roc_auc_score
##import random
##from tqdm import tqdm
### Replace the AUC calculation with regression metrics:
##from sklearn.metrics import mean_squared_error, r2_score
##
##
### 1-6: Generate the DataFrame with 100 samples
##def generate_data(num_samples=100, p0=0.3, p1=0.3):
##    # Generate random user_ids (10 unique users)
##    user_ids = [f"user_{random.randint(1, 10)}" for _ in range(num_samples)]
##    #user_ids = [f"user_{random.randint(1, 10)}" for _ in range(num_samples)]
##
##    # Generate random item_ids (20 unique items)
##    item_ids = [f"item_{random.randint(1, 20)}" for _ in range(num_samples)]
##    #item_ids = [f"item_{random.randint(1, 20)}" for _ in range(num_samples)]
##
##    # Generate random binary features
##    is_viewed = np.random.randint(0, 2, num_samples)
##    is_clicked = np.random.randint(0, 2, num_samples)
##    is_bought = np.random.randint(0, 2, num_samples)
##
##    # Generate timestamps (within last 30 days)
##    timestamps = pd.to_datetime(np.random.randint(
##        pd.Timestamp('2023-01-01').value,
##        pd.Timestamp('2023-01-31').value,
##        num_samples
##    ))
##
##    # Create DataFrame
##    df = pd.DataFrame({
##        'user_id': user_ids,
##        'item_id': item_ids,
##        'is_viewed': is_viewed,
##        'is_clicked': is_clicked,
##        'is_bought': is_bought,
##        'time_stamp': timestamps
##    })
##
##    # Calculate label
##    df['label'] = p0 * df['is_viewed'] + p1 * df['is_clicked'] + (1 - p0 - p1) * df['is_bought']
##
##    # Group by user_id and sort by timestamp
##    #df = df.groupby('user_id').apply(lambda x: x.sort_values('time_stamp')).reset_index(drop=True)
##    # Group by user_id and sort by time_stamp within each group
##    df = df.sort_values(['user_id', 'time_stamp']).reset_index(drop=True)
##
##    return df
##
##
##### 7: Create sequence features
####   this is for label is an int
####   Evaluate with binary metrics
####def create_sequence_features(df, max_seq_length=5):
####    # Create user browsing history sequences
####    user_sequences = df.groupby('user_id').apply(
####        lambda x: x[['item_id', 'is_viewed', 'is_clicked', 'is_bought']].values.tolist()
####    ).to_dict()
####
####    # Pad sequences to max_seq_length
####    padded_sequences = {}
####    for user_id, seq in user_sequences.items():
####        if len(seq) > max_seq_length:
####            padded_seq = seq[-max_seq_length:]
####        else:
####            padded_seq = seq + [[0, 0, 0, 0]] * (max_seq_length - len(seq))
####        padded_sequences[user_id] = padded_seq
####
####    # Add sequence features to DataFrame
####    df['history_seq'] = df['user_id'].map(padded_sequences)
####
####    # Create sliding window features (last 3 actions)
####    df['sliding_window'] = df.groupby('user_id')['item_id'].transform(
####        lambda x: x.rolling(3, min_periods=1).apply(lambda y: list(y)).shift(1)
####    )
####
####    # Create click sequence
####    df['click_seq'] = df.groupby('user_id')['is_clicked'].transform(
####        lambda x: x.rolling(3, min_periods=1).apply(lambda y: list(y)).shift(1)
####    )
####
####    return df
##
####   this is for label is a float
####   Evaluate with continuous metrics
####def create_sequence_features(df, max_seq_length=5):
####    # Create user browsing history sequences
####    user_sequences = df.groupby('user_id').apply(
####        lambda x: x[['item_id', 'is_viewed', 'is_clicked', 'is_bought']].values.tolist()
####    ).to_dict()
####
####    # Pad sequences to max_seq_length
####    padded_sequences = {}
####    for user_id, seq in user_sequences.items():
####        if len(seq) > max_seq_length:
####            padded_seq = seq[-max_seq_length:]
####        else:
####            padded_seq = seq + [['0', 0, 0, 0]] * (max_seq_length - len(seq))
####        padded_sequences[user_id] = padded_seq
####
####    # Add sequence features to DataFrame
####    df['history_seq'] = df['user_id'].map(padded_sequences)
####
####    # Create sliding window features using shift and list aggregation
####    df['sliding_window'] = df.groupby('user_id')['item_id'].apply(
####        lambda x: [x.iloc[max(0, i - 3):i].tolist() for i in range(1, len(x) + 1)]
####    ).explode().reset_index(drop=True)
####
####    # Create click sequence
####    df['click_seq'] = df.groupby('user_id')['is_clicked'].apply(
####        lambda x: [x.iloc[max(0, i - 3):i].tolist() for i in range(1, len(x) + 1)]
####    ).explode().reset_index(drop=True)
####
####    return df
##
##
##def create_sequence_features(df, max_seq_length=5):
##    # Create user browsing history sequences
##    user_sequences = df.groupby('user_id').apply(
##        lambda x: x[['item_id', 'is_viewed', 'is_clicked', 'is_bought', 'time_stamp']].values.tolist()
##    ).to_dict()
##
##    # Pad sequences to max_seq_length
##    padded_sequences = {}
##    for user_id, seq in user_sequences.items():
##        seq = sorted(seq, key= lambda x:x[-1])
##        seq1 = []
##        for item in seq:
##            seq1.append(item[:-1])
##        if len(seq1) > max_seq_length:
##            padded_seq = seq1[-max_seq_length:]
##        else:
##            padded_seq = seq1 + [['0', 0, 0, 0]] * (max_seq_length - len(seq1))
##        padded_sequences[user_id] = padded_seq
##
##    # Add sequence features to DataFrame
##    df['history_seq'] = df['user_id'].map(padded_sequences)
##
##    # Create sliding window features using shift and list aggregation
##    df['sliding_window'] = df.groupby('user_id')['item_id'].apply(
##        lambda x: [x.iloc[max(0, i - 3):i].tolist() for i in range(1, len(x) + 1)]
##    ).explode().reset_index(drop=True)
##
##    # Create click sequence
##    df['click_seq'] = df.groupby('user_id')['is_clicked'].apply(
##        lambda x: [x.iloc[max(0, i - 3):i].tolist() for i in range(1, len(x) + 1)]
##    ).explode().reset_index(drop=True)
##
##    return df
##
### 8-10: DIN Model Implementation
##class DIN(nn.Module):
##    def __init__(self, num_items, embedding_dim=32, hidden_units=[64, 32]):
##        super(DIN, self).__init__()
##        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
##
##        # Attention layers
##        self.attention_mlp = nn.Sequential(
##            nn.Linear(embedding_dim * 4, 80),
##            nn.ReLU(),
##            nn.Linear(80, 40),
##            nn.ReLU(),
##            nn.Linear(40, 1)
##        )
##
##        # DNN layers
##        self.dnn = nn.Sequential(
##            nn.Linear(embedding_dim * 2, hidden_units[0]),
##            nn.ReLU(),
##            nn.Linear(hidden_units[0], hidden_units[1]),
##            nn.ReLU(),
##            nn.Linear(hidden_units[1], 1)
##
##            # if for binary metrics, using this
##            # if for continuous metrics, DO NOT using this
##            #nn.Sigmoid()
##        )
##
##    def forward(self, target_item, history_seq):
##        # Embed target item
##        target_embed = self.item_embedding(target_item)  # (batch_size, embed_dim)
##
##        # Embed history sequence
##        history_embed = self.item_embedding(history_seq)  # (batch_size, seq_len, embed_dim)
##
##        # Expand target for attention calculation
##        target_expanded = target_embed.unsqueeze(1).expand(-1, history_embed.size(1), -1)
##
##        # Attention input: [target, hist_item, target*hist_item, target-hist_item]
##        attention_input = torch.cat([
##            target_expanded,
##            history_embed,
##            target_expanded * history_embed,
##            target_expanded - history_embed
##        ], dim=-1)
##
##        # Calculate attention weights
##        attention_weights = self.attention_mlp(attention_input)  # (batch_size, seq_len, 1)
##        attention_weights = torch.softmax(attention_weights, dim=1)
##
##        # Apply attention
##        weighted_history = torch.sum(attention_weights * history_embed, dim=1)
##
##        # Concatenate target and weighted history
##        din_input = torch.cat([target_embed, weighted_history], dim=-1)
##
##        # Final prediction
##        output = self.dnn(din_input)
##        return output.squeeze()
##
##
##class RecommendationDataset(Dataset):
##    def __init__(self, df, item_to_idx):
##        self.df = df
##        self.item_to_idx = item_to_idx
##
##    def __len__(self):
##        return len(self.df)
##
##    def __getitem__(self, idx):
##        row = self.df.iloc[idx]
##
##        # Convert items to indices
##        target_item = self.item_to_idx.get(row['item_id'], 0)
##
##        # Convert history sequence to indices
##        history_seq = [self.item_to_idx.get(item[0], 0) for item in row['history_seq']]
##
##        # Features
##        features = torch.tensor([
##            row['is_viewed'],
##            row['is_clicked'],
##            row['is_bought']
##        ], dtype=torch.float)
##
##        label = torch.tensor(row['label'], dtype=torch.float)
##
##        return {
##            'target_item': torch.tensor(target_item, dtype=torch.long),
##            'history_seq': torch.tensor(history_seq, dtype=torch.long),
##            'features': features,
##            'label': label
##        }
##
##
##def train_din_model(df, num_epochs=10, batch_size=32):
##    # Create item to index mapping
##    unique_items = df['item_id'].unique()
##    item_to_idx = {item: idx + 1 for idx, item in enumerate(unique_items)}  # 0 is for padding
##
##    # Split data
##    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
##
##    # Create datasets and dataloaders
##    train_dataset = RecommendationDataset(train_df, item_to_idx)
##    test_dataset = RecommendationDataset(test_df, item_to_idx)
##
##    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
##    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
##
##    # Initialize model
##    num_items = len(item_to_idx)
##    model = DIN(num_items)
##
##    #this is for label is an int
##    #criterion = nn.BCELoss()
##
##    #this is for label is a float
##    # Use MSELoss instead
##    criterion = nn.MSELoss()
##
##    optimizer = optim.Adam(model.parameters(), lr=0.001)
##
##    # Training loop
##    for epoch in range(num_epochs):
##        model.train()
##        total_loss = 0
##        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}')
##
##        for batch in progress_bar:
##            optimizer.zero_grad()
##
##            outputs = model(
##                batch['target_item'],
##                batch['history_seq']
##            )
##
##            loss = criterion(outputs, batch['label'])
##            loss.backward()
##            optimizer.step()
##
##            total_loss += loss.item()
##            progress_bar.set_postfix({'loss': loss.item()})
##
##        # Evaluate on test set
##        model.eval()
##        test_labels = []
##        test_preds = []
##
##        with torch.no_grad():
##            for batch in test_loader:
##                outputs = model(
##                    batch['target_item'],
##                    batch['history_seq']
##                )
##
##                test_labels.extend(batch['label'].tolist())
##                test_preds.extend(outputs.tolist())
##
##        # this is for label is an int
##        # Evaluate with binary metrics
##        #auc = roc_auc_score(test_labels, test_preds)
##        #print(f'Epoch {epoch + 1}, Train Loss: {total_loss / len(train_loader):.4f}, Test AUC: {auc:.4f}')
##
##        # this is for label is a float
##        # Evaluate with regression metrics
##        mse = mean_squared_error(test_labels, test_preds)
##        r2 = r2_score(test_labels, test_preds)
##        print(f'Epoch {epoch + 1}, Train Loss: {total_loss / len(train_loader):.4f}, Test MSE: {mse:.4f}, R2: {r2:.4f}')
##
##    return model
##
##
### Main execution
##if __name__ == "__main__":
##    # Generate data
##    df = generate_data(num_samples=100)
##
##    # Create sequence features
##    df = create_sequence_features(df)
##
##    # Train DIN model
##    model = train_din_model(df)
##
##    # Example prediction
##    sample = df.iloc[0]
##    item_to_idx = {item: idx + 1 for idx, item in enumerate(df['item_id'].unique())}
##
##    with torch.no_grad():
##        target_item = torch.tensor([item_to_idx.get(sample['item_id'], 0)], dtype=torch.long)
##        history_seq = torch.tensor([[item_to_idx.get(item[0], 0) for item in sample['history_seq']]], dtype=torch.long)
##
##        prediction = model(target_item, history_seq)
##        print(f"\nSample prediction - Actual: {sample['label']:.4f}, Predicted: {prediction.item():.4f}")

