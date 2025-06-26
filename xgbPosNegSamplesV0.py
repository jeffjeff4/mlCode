##1. crating ranking log, has 1000 samples, it has
##1) product title
##2) product id
##3) user id
##4) price
##5) ranking score, an int from 1-5, 1 means bad, 5 means good
##6) whether the product is clicked or not
##7) whether the product is added to cart or not
##8) whether the product is purchased or not
##9) comments sharing users' feedback to the product, few natural language sentences, means like it or not, etc
##10) time stam
##11) a product id must get click first, then add to cart, then purchase
##12) positive samples means product has click, or add to cart, or purchase
##13) negative samples means product has no click, no add to cart, no purchase
##14) number of positive samples is much less than negative samples
##15) ranking position, an int from 1-60, 1 means top ranking position in ranking list, 60 means the last ranking position
##
##2. generate model training sample based on:
##1) group by user
##2) please generate pairwise group, which means in one group, there is a (could be 1-2) positive samples, and at least doubled negative samples
##3) sort by time stamp
##4) make sure the training dataset is usabe in xgboost, din, and esmm models

#-------------------------------------------------------
#Generate Synthetic Ranking Log Dataset (1000 samples)
#-------------------------------------------------------

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

from sklearn.model_selection import train_test_split
import xgboost as xgb

from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Generate product and user IDs
product_ids = [f'p_{i:03d}' for i in range(1, 101)]
user_ids = [f'u_{i:03d}' for i in range(1, 51)]
product_titles = [
    f"Product {i} - {random.choice(['Wireless', 'Smart', 'Premium', 'Organic', 'Luxury'])} "
    f"{random.choice(['Headphones', 'Watch', 'T-Shirt', 'Water Bottle', 'Backpack'])}"
    for i in range(1, 101)
]

# Generate timestamp base
base_time = datetime.now() - timedelta(days=30)

# Generate 1000 samples
data = []
for _ in range(1000):
    user_id = random.choice(user_ids)
    product_id = random.choice(product_ids)
    price = round(random.uniform(5, 200), 2)
    ranking_score = random.randint(1, 5)
    ranking_position = random.randint(1, 60)
    timestamp = base_time + timedelta(minutes=random.randint(0, 43200))  # 30 days range

    # Generate user actions with dependency
    clicked = random.random() < 0.3  # 30% click rate
    added_to_cart = clicked and (random.random() < 0.5)  # 50% of clicked
    purchased = added_to_cart and (random.random() < 0.7)  # 70% of cart additions

    # Generate comments based on actions
    if purchased:
        comment = random.choice([
            "Great product! Will buy again",
            "Excellent quality, highly recommend",
            "Perfect fit, very satisfied"
        ])
    elif added_to_cart:
        comment = random.choice([
            "Considering buying this",
            "Looks good but still comparing",
            "Added to cart for later"
        ])
    elif clicked:
        comment = random.choice([
            "Interesting product",
            "Not sure about this one",
            "Might come back to this later"
        ])
    else:
        comment = ""

    data.append([
        product_titles[int(product_id.split('_')[1]) - 1],
        product_id,
        user_id,
        price,
        ranking_score,
        int(clicked),
        int(added_to_cart),
        int(purchased),
        comment,
        timestamp,
        ranking_position
    ])

# Create DataFrame
columns = [
    'product_title', 'product_id', 'user_id', 'price', 'ranking_score',
    'clicked', 'added_to_cart', 'purchased', 'comment', 'timestamp', 'ranking_position'
]
df = pd.DataFrame(data, columns=columns)

# Add label column (positive = any action taken)
df['label'] = (df['clicked'] | df['added_to_cart'] | df['purchased']).astype(int)

df['unix_time'] = df['timestamp'].astype('int64') // 10**9
df['dayofweek'] = df['timestamp'].dt.dayofweek

# Verify data distribution
print(f"Positive samples: {df['label'].sum()}")
print(f"Negative samples: {len(df) - df['label'].sum()}")

from itertools import combinations

#-------------------------------------------------------
#Generate Model Training Samples (Pairwise Groups)
#-------------------------------------------------------

def generate_pairwise_samples(group):
    # Separate positive and negative samples
    positives = group[group['label'] == 1]
    negatives = group[group['label'] == 0]

    # Sort by timestamp
    positives = positives.sort_values('timestamp')
    negatives = negatives.sort_values('timestamp')

    # Generate pairs (1-2 positives + 2x negatives)
    samples = []
    for i in range(min(2, len(positives))):  # Take 1-2 positives
        pos_sample = positives.iloc[i]
        # Select double the negatives
        neg_samples = negatives.iloc[i * 2:(i + 1) * 2] if len(negatives) >= 2 else negatives

        # Create pairs
        for _, neg_sample in neg_samples.iterrows():
            # Create feature vector for each sample
            for sample in [pos_sample, neg_sample]:
                features = {
                    'user_id': sample['user_id'],
                    'product_id': sample['product_id'],
                    'price': sample['price'],
                    'ranking_score': sample['ranking_score'],
                    'ranking_position': sample['ranking_position'],
                    'comment_length': len(sample['comment']),
                    'label': sample['label'],
                    'timestamp': sample['timestamp']
                }
                samples.append(features)

    return samples

def generate_pairwise_samples_v1(group):
    # Separate positive and negative samples
    positives = group[group['label'] == 1]
    negatives = group[group['label'] == 0]

    # Sort by timestamp
    positives = positives.sort_values('timestamp')
    negatives = negatives.sort_values('timestamp')

    # Generate pairs (1-2 positives + 2x negatives)
    samples = []
    for i in range(min(2, len(positives))):  # Take 1-2 positives
        pos_sample = positives.iloc[i]
        # Select double the negatives
        neg_samples = negatives.iloc[i * 2:(i + 1) * 2] if len(negatives) >= 2 else negatives

        # Create pairs
        for _, neg_sample in neg_samples.iterrows():
            # Create feature vector for each sample
            for sample in [pos_sample, neg_sample]:
                samples.append(sample)

    return samples

# Group by user and generate samples
all_samples = []
for _, group in df.groupby('user_id'):
    all_samples.extend(generate_pairwise_samples_v1(group))

# Convert to DataFrame
all_df = pd.DataFrame(all_samples)

# Save to CSV
all_df.to_csv('ranking_all_data.csv', index=False)

#-------------------------------------------------------
#-------------------------------------------------------
# Add one-hot encoding for categoricals if needed

from sklearn.preprocessing import LabelEncoder

le_user = LabelEncoder()
le_product = LabelEncoder()

all_df['user_id_enc'] = le_user.fit_transform(all_df['user_id'])
all_df['product_id_enc'] = le_product.fit_transform(all_df['product_id'])

# Basic feature engineering
#features = all_df[[
#    'price', 'ranking_score', 'ranking_position', 'comment_length',
#    'user_id_enc', 'product_id_enc', 'clicked', 'added_to_cart', 'purchased'
#]]
#feature_cols = [
#    'price', 'ranking_score', 'ranking_position', 'comment_length',
#    'user_id_enc', 'product_id_enc', 'timestamp'
#]
feature_cols = [
    'price', 'ranking_score', 'ranking_position',
    'user_id_enc', 'product_id_enc', 'unix_time',
    'dayofweek'
]


labels = all_df['label']

train_df, test_df = train_test_split(all_df, test_size=0.2, random_state=42)

train_df_sorted = train_df.sort_values('user_id')
X_train_df_sorted = train_df_sorted[feature_cols]
y_train_df_sorted = train_df_sorted['label']
# group by user_id, count the number of samples in each group
group_sorted = train_df_sorted.groupby('user_id').size().tolist()

X_test_df = test_df[feature_cols]
y_test_df = test_df['label']

#-------------------------------------------------------
# training
#-------------------------------------------------------
dtrain = xgb.DMatrix(X_train_df_sorted, label=y_train_df_sorted, enable_categorical=True)
dtrain.set_group(group_sorted)

# 5. Define parameters (optimized for ranking with class imbalance)
params = {
    'objective': 'rank:pairwise',
    'eval_metric': ['auc', 'error', 'ndcg'],
    'eta': 0.1,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    #'scale_pos_weight': len(y_train_df_sorted[label==0])/len(y_train_df_sorted[label==1]),  # Handle imbalance
    'seed': 42,
    'tree_method': 'hist'  # Faster training
}

dtest = xgb.DMatrix(X_test_df, label=y_test_df, enable_categorical=True)


# 6. Train with early stopping
evals_result = {}
xgb_ranker = xgb.train(
    params,
    dtrain,
    num_boost_round=1000,
    evals=[(dtrain, 'train'), (dtest, 'test')],
    early_stopping_rounds=50,
    evals_result=evals_result,
    verbose_eval=20
)

# 7. Evaluate
y_pred_proba = xgb_ranker.predict(dtest)
y_pred = (y_pred_proba > 0.5).astype(int)

print("\nClassification Report:")
print(classification_report(y_test_df, y_pred))

print(f"\nROC AUC Score: {roc_auc_score(y_test_df, y_pred_proba):.4f}")

# 8. Feature importance
print("\nFeature Importance:")
importance = xgb_ranker.get_score(importance_type='gain')
for k, v in sorted(importance.items(), key=lambda x: x[1], reverse=True):
    print(f"{k}: {v:.4f}")

# 9. Save model
xgb_ranker.save_model('xgboost_ranking_model.json')


#-------------------------------------------------------
#-------------------------------------------------------

#-------------------------------------------------------
#-------------------------------------------------------

#-------------------------------------------------------
#-------------------------------------------------------

#-------------------------------------------------------
#-------------------------------------------------------

#-------------------------------------------------------
#-------------------------------------------------------

#-------------------------------------------------------
#-------------------------------------------------------

#-------------------------------------------------------
#-------------------------------------------------------

#-------------------------------------------------------
#-------------------------------------------------------
