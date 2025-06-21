##python, generate a df,
##1) with 100 samples,
##2) each sample has 6 columns, they are user_id, item_id, is_viewed, is_clicked, is_bought, time_stamp.
##3）user_id, item_id are randome id, could be duplicated
##4) is_viewed, is_clicked, is_bought are randomly generated, value either be 0 or 1
##5) generate a new column, named 'label'. its value = p0 * df['is_viewed'] + p1 * df['is_clicked'] + (1-p0-p1) * df[' is_bought']
##6) group by user_id, sort by time stamp


import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
n_samples = 100
p0 = 0.4  # Weight for is_viewed
p1 = 0.3  # Weight for is_clicked
# Weight for is_bought will be (1 - p0 - p1) = 0.3

# Generate random data
base_date = datetime(2023, 1, 1)
data = {
    'user_id': np.random.randint(1, 21, size=n_samples),  # 20 unique users
    'item_id': np.random.randint(101, 151, size=n_samples),  # 50 unique items
    'is_viewed': np.random.randint(0, 2, size=n_samples),
    'is_clicked': np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3]),  # 30% click probability
    'is_bought': np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1]),  # 10% purchase probability
    'time_stamp': [base_date + timedelta(minutes=np.random.randint(0, 1440)) for _ in range(n_samples)]
}

# Create DataFrame
df = pd.DataFrame(data)

# Calculate weighted label
df['label'] = p0 * df['is_viewed'] + p1 * df['is_clicked'] + (1 - p0 - p1) * df['is_bought']

# Group by user_id and sort by time_stamp within each group
df = df.sort_values(['user_id', 'time_stamp']).reset_index(drop=True)

# Display the first 10 rows
print("First 10 rows (grouped by user_id, sorted by time_stamp):")
print(df.head(10))

# Show summary statistics
print("\nSummary Statistics:")
print(df[['is_viewed', 'is_clicked', 'is_bought', 'label']].mean())

# Show value counts for the label column
print("\nLabel value distribution:")
print(df['label'].value_counts().sort_index())

# Show example of grouped data
print("\nExample of grouped data for user_id 1:")
print(df[df['user_id'] == 1])