import pandas as pd
import numpy as np


def generate_ctr_labels(df, user_col='user_id', item_col='item_id', click_col='is_click'):
    """
    Generate CTR labels grouped by user-item pairs
    Args:
        df: DataFrame containing user-item interactions
        user_col: column name for user identifiers
        item_col: column name for item identifiers
        click_col: column indicating click (1) or no click (0)
    Returns:
        DataFrame with CTR labels
    """
    # Calculate clicks and impressions
    ctr_data = df.groupby([user_col, item_col]).agg(
        clicks=(click_col, 'sum'),
        impressions=(click_col, 'count')
    ).reset_index()

    # Calculate CTR
    ctr_data['ctr_label'] = ctr_data['clicks'] / ctr_data['impressions']

    # Add smoothing for small sample sizes (avoid extreme CTRs)
    alpha = 5  # pseudo-clicks
    beta = 10  # pseudo-impressions
    ctr_data['smoothed_ctr_label'] = (ctr_data['clicks'] + alpha) / (ctr_data['impressions'] + beta)

    return ctr_data


# Example usage
data = pd.DataFrame({
    'user_id': [1, 1, 1, 2, 2, 2, 3, 3],
    'item_id': [101, 101, 102, 101, 102, 103, 101, 103],
    'is_click': [1, 0, 1, 0, 0, 1, 1, 0]
})

ctr_labels = generate_ctr_labels(data)
print("CTR Labels:")
print(ctr_labels)

from datetime import datetime, timedelta


def generate_ranking_labels(interactions_df, user_col='user_id', item_col='item_id',
                            time_col='timestamp', label_window_hours=24):
    """
    Generate ranking labels with time-based protection against leakage
    Args:
        interactions_df: DataFrame of user-item interactions
        label_window_hours: time window after each interaction to observe outcomes
    Returns:
        DataFrame with ranking labels (e.g., engagement metrics)
    """
    # Ensure datetime format
    interactions_df[time_col] = pd.to_datetime(interactions_df[time_col])
    interactions_df = interactions_df.sort_values([user_col, time_col])

    # Calculate future engagement within time window
    labels = []
    for _, row in interactions_df.iterrows():
        window_end = row[time_col] + timedelta(hours=label_window_hours)

        # Find future engagements for this user
        future_engagements = interactions_df[
            (interactions_df[user_col] == row[user_col]) &
            (interactions_df[time_col] > row[time_col]) &
            (interactions_df[time_col] <= window_end)
            ]

        # Create ranking label (e.g., total engagements)
        label = len(future_engagements)
        labels.append(label)

    interactions_df['ranking_label'] = labels
    return interactions_df


# Example usage
ranking_data = pd.DataFrame({
    'user_id': [1, 1, 1, 2, 2, 3],
    'item_id': [101, 102, 103, 101, 102, 101],
    'timestamp': [
        '2023-01-01 10:00',
        '2023-01-01 11:00',
        '2023-01-01 15:00',
        '2023-01-02 09:00',
        '2023-01-02 12:00',
        '2023-01-03 14:00'
    ],
    'engagement': [1, 1, 2, 1, 0, 3]
})

ranking_labels = generate_ranking_labels(ranking_data)
print("\nRanking Labels:")
print(ranking_labels[['user_id', 'item_id', 'timestamp', 'ranking_label']])


def generate_time_based_labels(features_df, labels_df,
                               entity_col='user_id',
                               feature_time_col='feature_timestamp',
                               label_time_col='label_timestamp'):
    """
    Generate labels ensuring feature time < label time
    Args:
        features_df: DataFrame containing features
        labels_df: DataFrame containing potential labels
        entity_col: column to join on (e.g., user_id)
        feature_time_col: timestamp column in features
        label_time_col: timestamp column in labels
    Returns:
        Merged DataFrame with safe label assignments
    """
    # Convert to datetime if needed
    features_df[feature_time_col] = pd.to_datetime(features_df[feature_time_col])
    labels_df[label_time_col] = pd.to_datetime(labels_df[label_time_col])

    # Create exploded DataFrame for joining
    features_df['join_key'] = 1
    labels_df['join_key'] = 1

    # Cross join then filter
    merged = pd.merge(features_df, labels_df, on=['join_key', entity_col])
    merged = merged[merged[feature_time_col] < merged[label_time_col]]

    # For each feature row, keep only the earliest future label
    merged = merged.sort_values([entity_col, feature_time_col, label_time_col])
    safe_labels = merged.groupby([entity_col, feature_time_col]).first().reset_index()

    return safe_labels


# Example usage
features = pd.DataFrame({
    'user_id': [1, 1, 2, 2, 3],
    'feature_values': [0.1, 0.2, 0.3, 0.4, 0.5],
    'feature_timestamp': [
        '2023-01-01 10:00',
        '2023-01-02 11:00',
        '2023-01-01 09:00',
        '2023-01-03 12:00',
        '2023-01-02 14:00'
    ]
})

labels = pd.DataFrame({
    'user_id': [1, 1, 2, 3, 3],
    'label_value': [1, 0, 1, 0, 1],
    'label_timestamp': [
        '2023-01-01 12:00',
        '2023-01-03 10:00',
        '2023-01-02 08:00',
        '2023-01-02 16:00',
        '2023-01-03 15:00'
    ]
})

time_safe_labels = generate_time_based_labels(features, labels)
print("\nTime-Safe Labels:")
print(time_safe_labels[['user_id', 'feature_timestamp', 'label_timestamp', 'label_value']])
