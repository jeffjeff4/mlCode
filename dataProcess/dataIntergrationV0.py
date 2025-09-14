##import pandas as pd
##import numpy as np
##from datetime import datetime, timedelta
##import random
##
### Set random seed for reproducibility
##np.random.seed(42)
##random.seed(42)
##
### ======================
### 1. Generate Realistic Datasets
### ======================
##TTL_SAMPLE_NUM=100
##RANGE_NUM0 = 500
##
### Generate TTL_SAMPLE_NUM users with demographics
##users = pd.DataFrame({
##    'user_id': [f'U{TTL_SAMPLE_NUM + i}' for i in range(TTL_SAMPLE_NUM)],
##    'age': np.random.randint(18, 70, TTL_SAMPLE_NUM),
##    'gender': np.random.choice(['M', 'F', 'Other'], TTL_SAMPLE_NUM, p=[0.48, 0.48, 0.04]),
##    'signup_date': [datetime(2022, 1, 1) + timedelta(days=np.random.randint(0, 400)) for _ in range(TTL_SAMPLE_NUM)],
##    'country': np.random.choice(['US', 'UK', 'CA', 'AU', 'DE'], TTL_SAMPLE_NUM, p=[0.5, 0.2, 0.1, 0.1, 0.1])
##})
##
### Generate 50,000 activity events
##activities = pd.DataFrame({
##    'event_id': [f'E{1000 + i}' for i in range(RANGE_NUM0)],
##    'user_id': np.random.choice(users['user_id'], RANGE_NUM0),
##    'event_time': [datetime(2023, 1, 1) + timedelta(minutes=np.random.randint(0, 60 * 24 * 180)) for _ in range(RANGE_NUM0)],
##    'event_type': np.random.choice(['page_view', 'product_view', 'search', 'add_to_cart'], RANGE_NUM0,
##                                   p=[0.5, 0.3, 0.15, 0.05]),
##    'duration_sec': np.random.exponential(30, RANGE_NUM0).astype(int),
##    'product_id': [f'P{np.random.randint(1, 50)}' for _ in range(RANGE_NUM0)]
##})
##
### Generate 5,000 purchases
##purchases = pd.DataFrame({
##    'transaction_id': [f'T{RANGE_NUM0 + i}' for i in range(5000)],
##    'user_id': np.random.choice(users['user_id'], 5000),
##    'purchase_time': [datetime(2023, 1, 1) + timedelta(minutes=np.random.randint(0, 60 * 24 * 180)) for _ in
##                      range(5000)],
##    'product_id': [f'P{np.random.randint(1, 50)}' for _ in range(5000)],
##    'amount': np.round(np.random.lognormal(3, 0.5, 5000), 2),
##    'payment_method': np.random.choice(['credit_card', 'paypal', 'apple_pay'], 5000)
##})
##
##
### ======================
### 2. Data Integration Pipeline
### ======================
##
##def integrate_datasets(users, activities, purchases):
##    """Integrate user demographics, activity logs, and purchase records"""
##
##    # Convert all timestamps
##    for df in [users, activities, purchases]:
##        if 'time' in df.columns.str.contains('time|date'):
##            df[df.columns[df.columns.str.contains('time|date')]] = \
##                pd.to_datetime(df[df.columns[df.columns.str.contains('time|date')]].squeeze())
##
##    # 1. Calculate user activity features
##    activity_features = activities.groupby('user_id').agg(
##        total_events=('event_id', 'count'),
##        avg_duration=('duration_sec', 'mean'),
##        last_activity=('event_time', 'max'),
##        product_views=('event_type', lambda x: (x == 'product_view').sum()),
##        searches=('event_type', lambda x: (x == 'search').sum())
##    ).reset_index()
##
##    # 2. Calculate purchase behavior features
##    purchase_features = purchases.groupby('user_id').agg(
##        total_spend=('amount', 'sum'),
##        avg_purchase=('amount', 'mean'),
##        first_purchase=('purchase_time', 'min'),
##        last_purchase=('purchase_time', 'max'),
##        purchase_count=('transaction_id', 'count')
##    ).reset_index()
##
##    # 3. Calculate time since last activity/purchase
##    current_date = datetime(2023, 7, 1)
##    activity_features['days_since_activity'] = (current_date - activity_features['last_activity']).dt.days
##    purchase_features['days_since_purchase'] = (current_date - purchase_features['last_purchase']).dt.days
##
##    # 4. Merge all datasets
##    merged = users.merge(activity_features, on='user_id', how='left') \
##        .merge(purchase_features, on='user_id', how='left')
##
##    # 5. Fill NA values for users without activities/purchases
##    activity_cols = ['total_events', 'avg_duration', 'product_views', 'searches', 'days_since_activity']
##    purchase_cols = ['total_spend', 'avg_purchase', 'purchase_count', 'days_since_purchase']
##
##    merged[activity_cols] = merged[activity_cols].fillna(0)
##    merged[purchase_cols] = merged[purchase_cols].fillna(0)
##    merged[['first_purchase', 'last_purchase']] = merged[['first_purchase', 'last_purchase']].fillna(pd.NaT)
##
##    # 6. Calculate derived features
##    merged['purchase_probability'] = np.where(merged['purchase_count'] > 0, 1, 0)
##    merged['avg_duration'] = merged['avg_duration'].round(1)
##    merged['avg_purchase'] = merged['avg_purchase'].round(2)
##    merged['activity_level'] = pd.cut(merged['total_events'],
##                                      bins=[-1, 0, 5, 20, 100, np.inf],
##                                      labels=['none', 'low', 'medium', 'high', 'very_high'])
##
##    return merged
##
##
### ======================
### 3. Execute Integration
### ======================
##
### Run the integration pipeline
##final_dataset = integrate_datasets(users, activities, purchases)
##
### Show results
##print("Integrated dataset shape:", final_dataset.shape)
##print("\nSample records:")
##print(final_dataset.head(3))
##
##print("\nData types:")
##print(final_dataset.dtypes)
##
##print("\nMissing values:")
##print(final_dataset.isnull().sum())
##
##
### ======================
### 4. Advanced Integration (Time-Based)
### ======================
##
##def time_based_integration(users, activities, purchases, observation_date, prediction_window=30):
##    """Advanced integration with time-based feature engineering"""
##
##    # Convert to datetime if needed
##    observation_date = pd.to_datetime(observation_date)
##    cutoff_date = observation_date - timedelta(days=prediction_window)
##
##    # Filter historical data
##    historical_acts = activities[activities['event_time'] <= observation_date]
##    historical_purchases = purchases[purchases['purchase_time'] <= observation_date]
##
##    # Create features
##    features = []
##
##    for _, user in users.iterrows():
##        user_id = user['user_id']
##
##        # User demographics
##        user_features = user.to_dict()
##
##        # Activity features (last 30 days)
##        user_acts = historical_acts[
##            (historical_acts['user_id'] == user_id) &
##            (historical_acts['event_time'] > cutoff_date)
##            ]
##
##        user_features.update({
##            'recent_events': len(user_acts),
##            'recent_product_views': (user_acts['event_type'] == 'product_view').sum(),
##            'recent_searches': (user_acts['event_type'] == 'search').sum(),
##            'last_event_type': user_acts['event_type'].iloc[-1] if len(user_acts) > 0 else None,
##            'days_since_last_activity': (observation_date - user_acts['event_time'].max()).days
##            if len(user_acts) > 0 else np.nan
##        })
##
##        # Purchase features (last 30 days)
##        user_purchases = historical_purchases[
##            (historical_purchases['user_id'] == user_id) &
##            (historical_purchases['purchase_time'] > cutoff_date)
##            ]
##
##        user_features.update({
##            'recent_purchases': len(user_purchases),
##            'recent_spend': user_purchases['amount'].sum(),
##            'last_payment_method': user_purchases['payment_method'].iloc[-1]
##            if len(user_purchases) > 0 else None,
##            'days_since_last_purchase': (observation_date - user_purchases['purchase_time'].max()).days
##            if len(user_purchases) > 0 else np.nan
##        })
##
##        # Future target (purchases in next 30 days)
##        future_purchases = purchases[
##            (purchases['user_id'] == user_id) &
##            (purchases['purchase_time'] > observation_date) &
##            (purchases['purchase_time'] <= observation_date + timedelta(days=prediction_window))
##            ]
##
##        user_features['future_purchases'] = len(future_purchases)
##
##        features.append(user_features)
##
##    return pd.DataFrame(features)
##
##
### Execute time-based integration
##observation_date = datetime(2023, 6, 1)
##time_based_features = time_based_integration(users, activities, purchases, observation_date)
##
##print("\nTime-based features shape:", time_based_features.shape)
##print("\nTime-based features sample:")
##print(time_based_features.head(3))