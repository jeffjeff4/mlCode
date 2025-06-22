import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.preprocessing import LabelEncoder


# 1-9: Generate the DataFrame with 100 samples
def generate_data(num_samples=100, p0=0.3, p1=0.3):
    # Generate random user_ids (10 unique users)
    user_ids = [f"user_{random.randint(1, 10)}" for _ in range(num_samples)]

    # Generate random item_ids (20 unique items)
    item_ids = [f"item_{random.randint(1, 20)}" for _ in range(num_samples)]

    # Generate random binary features
    is_viewed = np.random.randint(0, 2, num_samples)
    is_clicked = np.random.randint(0, 2, num_samples)
    is_bought = np.random.randint(0, 2, num_samples)

    # Generate prices between 10 and 500 with 2 decimal places
    prices = np.round(np.random.uniform(10, 500, num_samples), 2)

    # Generate short reviews
    positive_reviews = [
        "Great product!", "Highly recommend", "Excellent quality",
        "Works perfectly", "Very satisfied", "Best purchase ever"
    ]
    negative_reviews = [
        "Poor quality", "Not worth it", "Disappointing",
        "Broke quickly", "Not as described", "Wouldn't buy again"
    ]
    reviews = [
        random.choice(positive_reviews) if random.random() > 0.3 else random.choice(negative_reviews)
        for _ in range(num_samples)
    ]

    # Generate grades (0-5)
    grades = np.random.randint(0, 6, num_samples)

    # Generate timestamps (within last 30 days)
    start_date = datetime.now() - timedelta(days=30)
    timestamps = [
        start_date + timedelta(
            seconds=random.randint(0, 30 * 24 * 60 * 60)
        ) for _ in range(num_samples)
    ]

    # Create DataFrame
    df = pd.DataFrame({
        'user_id': user_ids,
        'item_id': item_ids,
        'is_viewed': is_viewed,
        'is_clicked': is_clicked,
        'is_bought': is_bought,
        'price': prices,
        'review': reviews,
        'grade': grades,
        'time_stamp': timestamps
    })

    # Calculate label
    df['label'] = p0 * df['is_viewed'] + p1 * df['is_clicked'] + (1 - p0 - p1) * df['is_bought']

    # Group by user_id and sort by timestamp
    df = df.groupby('user_id').apply(lambda x: x.sort_values('time_stamp')).reset_index(drop=True)

    return df



def create_sequence_features(df, max_seq_length=5, seq_len=3):
    # Create user browsing history sequences
    user_sequences = df.groupby('user_id').apply(
        lambda x: x[['item_id', 'is_viewed', 'is_clicked', 'is_bought', 'price', 'grade']].values.tolist()
    ).to_dict()

    # Pad sequences to max_seq_length
    padded_sequences = {}
    for user_id, seq in user_sequences.items():
        if len(seq) > max_seq_length:
            padded_seq = seq[-max_seq_length:]
        else:
            padded_seq = seq + [['0', 0, 0, 0, 0, 0]] * (max_seq_length - len(seq))
        padded_sequences[user_id] = padded_seq

    # Add sequence features to DataFrame
    df['history_seq'] = df['user_id'].map(padded_sequences)

    # Create sliding window features using shift and list aggregation
    df['sliding_window'] = df.groupby('user_id')['item_id'].apply(
        lambda x: [x.iloc[max(0, i - seq_len):i].tolist() for i in range(1, len(x) + 1)]
    ).explode().reset_index(drop=True)

    # Create click sequence
    df['click_seq'] = df.groupby('user_id')['is_clicked'].apply(
        lambda x: [x.iloc[max(0, i - seq_len):i].tolist() for i in range(1, len(x) + 1)]
    ).explode().reset_index(drop=True)

    # Create price sequence (last seq_len prices)
    #df['price_seq'] = df.groupby('user_id')['price'].transform(
    #    lambda x: x.shift(1).rolling(3, min_periods=1).apply(#lambda y: list(y.dropna()))
    #)

    df['price_seq'] = df.groupby('user_id')['price'].apply(
        lambda x: [x.iloc[max(0, i - seq_len):i].tolist() for i in range(1, len(x) + 1)]
    ).explode().reset_index(drop=True)

    # Previous interactions count
    df['prev_views'] = df.groupby('user_id')['is_viewed'].shift(1).rolling(3, min_periods=1).sum().reset_index(
        drop=True)
    df['prev_clicks'] = df.groupby('user_id')['is_clicked'].shift(1).rolling(3, min_periods=1).sum().reset_index(
        drop=True)
    df['prev_purchases'] = df.groupby('user_id')['is_bought'].shift(1).rolling(3, min_periods=1).sum().reset_index(
        drop=True)

    # Previous price statistics
    df['prev_avg_price'] = df.groupby('user_id')['price'].shift(1).rolling(3, min_periods=1).mean().reset_index(
        drop=True)
    df['prev_max_price'] = df.groupby('user_id')['price'].shift(1).rolling(3, min_periods=1).max().reset_index(
        drop=True)

    # Previous grade statistics
    df['prev_avg_grade'] = df.groupby('user_id')['grade'].shift(1).rolling(3, min_periods=1).mean().reset_index(
        drop=True)

    # Time since last interaction
    df['time_since_last'] = df.groupby('user_id')['time_stamp'].diff().dt.total_seconds().fillna(0)

    # Fill NA values
    df.fillna(0, inplace=True)

    return df


# 11-13: XGBoost Model Implementation
def prepare_features(df):
    # Encode categorical features
    le_user = LabelEncoder()
    le_item = LabelEncoder()
    df['user_id_encoded'] = le_user.fit_transform(df['user_id'])
    df['item_id_encoded'] = le_item.fit_transform(df['item_id'])

    # Extract features from history sequences
    df['last_viewed'] = df['history_seq'].apply(lambda x: x[-1][1] if len(x) > 0 else 0)
    df['last_clicked'] = df['history_seq'].apply(lambda x: x[-1][2] if len(x) > 0 else 0)
    df['last_bought'] = df['history_seq'].apply(lambda x: x[-1][3] if len(x) > 0 else 0)
    df['avg_price'] = df['history_seq'].apply(
        lambda x: np.mean([item[4] for item in x if item[4] != 0]) if any(item[4] != 0 for item in x) else 0
    )
    df['avg_grade'] = df['history_seq'].apply(
        lambda x: np.mean([item[5] for item in x if item[5] != 0]) if any(item[5] != 0 for item in x) else 0
    )


    features = [
        'user_id_encoded', 'item_id_encoded',
        'is_viewed', 'is_clicked', 'is_bought',
        'price', 'grade',
        'last_viewed', 'last_clicked', 'last_bought',
        'avg_price', 'avg_grade',
        'prev_views', 'prev_clicks', 'prev_purchases',
        'prev_avg_price', 'prev_max_price', 'prev_avg_grade',
        'time_since_last'
    ]

    return df[features], df['label']


def train_xgboost_model(df):
    # Calculate group sizes (e.g., interactions per user)
    group_sizes = df.groupby('user_id').size().values

    # Prepare features
    X, y = prepare_features(df)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train, group=group_sizes, enable_categorical=True)
    dtest = xgb.DMatrix(X_test, label=y_test, group=group_sizes, enable_categorical=True)

    # Set parameters
    params = {
        'objective': 'rank:ndcg',
        'eval_metric': 'ndcg@5',
        'max_depth': 6,
        'eta': 0.1,
        'ndcg_exp_gain': 'true'
    }

    # Train model
    num_rounds = 100
    evals = [(dtrain, 'train'), (dtest, 'test')]
    model = xgb.train(
        params,
        dtrain,
        num_rounds,
        evals=evals,
        early_stopping_rounds=10,
        verbose_eval=10
    )

    # Evaluate
    preds = model.predict(dtest)
    # Manually compute NDCG if needed
    from sklearn.metrics import ndcg_score
    ndcg = ndcg_score([y_test], [preds], k=5)
    print(f"NDCG@5: {ndcg:.4f}")

    # For binary classification (if you convert labels to binary)
    # roc_auc = roc_auc_score(y_test, preds)
    # print(f"Test AUC: {roc_auc:.4f}")

    return model


# Main execution
if __name__ == "__main__":
    # Generate data
    df = generate_data(num_samples=100)

    # Create sequence features
    df = create_sequence_features(df)

    # Train XGBoost model
    model = train_xgboost_model(df)

    # Feature importance
    importance = model.get_score(importance_type='weight')
    print("\nFeature Importance:")
    for k, v in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        print(f"{k}: {v}")