import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import ndcg_score
from sklearn.model_selection import train_test_split


# 1-8: Generate the DataFrame with 100 samples
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

    # Calculate label (relevance score)
    #df['label'] = p0 * df['is_viewed'] + p1 * df['is_clicked'] + (1 - p0 - p1) * df['is_bought']
    df['label'] = pd.cut(
        p0 * df['is_viewed'] + p1 * df['is_clicked'] + (1 - p0 - p1) * df['is_bought'],
        bins=10,
        labels=False
    )

    return df


# 9-10: Create sequence features and group data
def prepare_ranking_data(df):
    # Group by user_id and sort by timestamp
    df = df.groupby('user_id').apply(lambda x: x.sort_values('time_stamp')).reset_index(drop=True)

    # Create sequence features (simplified for ranking)
    df['prev_views'] = df.groupby('user_id')['is_viewed'].shift(1).rolling(3, min_periods=1).sum()
    df['prev_clicks'] = df.groupby('user_id')['is_clicked'].shift(1).rolling(3, min_periods=1).sum()
    df['prev_purchases'] = df.groupby('user_id')['is_bought'].shift(1).rolling(3, min_periods=1).sum()
    df['prev_avg_grade'] = df.groupby('user_id')['grade'].shift(1).rolling(3, min_periods=1).mean()

    # Fill NA values
    df.fillna(0, inplace=True)

    return df


# 11: Create positive/negative samples
def create_samples(df, threshold=4):
    # Positive samples: label > threshold (4)
    # Negative samples: label <= threshold
    df['is_positive'] = (df['label'] > threshold).astype(int)
    return df


# 12-16: XGBoost Ranking Model with NDCG
def train_xgboost_ranking(df):
    # Encode categorical features
    le_user = LabelEncoder()
    le_item = LabelEncoder()
    df['user_id_encoded'] = le_user.fit_transform(df['user_id'])
    df['item_id_encoded'] = le_item.fit_transform(df['item_id'])

    # Feature columns
    features = [
        'user_id_encoded', 'item_id_encoded',
        'is_viewed', 'is_clicked', 'is_bought',
        'price', 'grade',
        'prev_views', 'prev_clicks', 'prev_purchases',
        'prev_avg_grade', 'is_positive'
    ]

    # Split data (maintain user groups)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Prepare group structure (number of items per user)
    train_groups = train_df.groupby('user_id_encoded').size().values
    test_groups = test_df.groupby('user_id_encoded').size().values

    # Create DMatrix for XGBoost ranking
    dtrain = xgb.DMatrix(
        train_df[features],
        label=train_df['label'],
        group=train_groups
    )
    dtest = xgb.DMatrix(
        test_df[features],
        label=test_df['label'],
        group=test_groups
    )

    # Ranking parameters
    params = {
        'objective': 'rank:ndcg',
        'eval_metric': 'ndcg@5',
        'eta': 0.1,
        'max_depth': 6,
        'ndcg_exp_gain': 'true'
    }

    # Train model
    evals = [(dtrain, 'train'), (dtest, 'eval')]
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=100,
        evals=evals,
        early_stopping_rounds=10,
        verbose_eval=10
    )

    # Evaluate
    preds = model.predict(dtest)
    ndcg = ndcg_score(
        [test_df['label'].values],
        [preds],
        k=5
    )
    print(f"\nTest NDCG@5: {ndcg:.4f}")

    return model


# Main execution
if __name__ == "__main__":
    # Generate data
    df = generate_data(num_samples=100)

    # Prepare ranking data
    df = prepare_ranking_data(df)

    # Create positive/negative samples
    df = create_samples(df)

    # Train XGBoost ranking model
    model = train_xgboost_ranking(df)

    # Feature importance
    importance = model.get_score(importance_type='weight')
    print("\nFeature Importance:")
    for k, v in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        print(f"{k}: {v}")