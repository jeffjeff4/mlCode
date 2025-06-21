##import pandas as pd
##import numpy as np
##from collections import defaultdict
##
### Sample user browsing history
##data = {
##    'user_id': [1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
##    'timestamp': [
##        '2023-01-01 10:00:00', '2023-01-01 10:01:00', '2023-01-01 10:03:00',
##        '2023-01-01 10:05:00', '2023-01-01 10:06:00', '2023-01-01 11:00:00',
##        '2023-01-01 11:01:30', '2023-01-01 11:02:00', '2023-01-01 11:04:00',
##        '2023-01-01 12:00:00', '2023-01-01 12:02:00', '2023-01-01 12:05:00'
##    ],
##    'page_url': [
##        'home', 'products', 'product_A', 'cart', 'checkout',
##        'home', 'categories', 'product_B', 'cart',
##        'home', 'product_C', 'categories'
##    ],
##    'dwell_time': [30, 45, 120, 60, 90, 40, 50, 180, 75, 60, 150, 80]
##}
##
##df = pd.DataFrame(data)
##df['timestamp'] = pd.to_datetime(df['timestamp'])
##
##def generate_sliding_windows(events, window_size=3, stride=1):
##    """Generate sliding windows from a sequence of events"""
##    windows = []
##    for i in range(0, len(events) - window_size + 1, stride):
##        window = events[i:i + window_size]
##        windows.append(window)
##    return windows
##
### Group by user and generate windows
##user_windows = defaultdict(list)
##for user_id, group in df.groupby('user_id'):
##    events = group['page_url'].tolist()
##    windows = generate_sliding_windows(events, window_size=3, stride=1)
##    user_windows[user_id] = windows
##
##print("User click sequence windows:")
##for user, windows in user_windows.items():
##    print(f"User {user}:")
##    for i, window in enumerate(windows):
##        print(f"  Window {i+1}: {window}")
##
##
##def generate_time_windows(user_df, time_window='5min'):
##    """Generate time-based sliding windows"""
##    user_df = user_df.sort_values('timestamp')
##    windows = []
##
##    # Create rolling time window
##    for i in range(len(user_df)):
##        window_mask = (user_df['timestamp'] >= user_df['timestamp'].iloc[i]) & \
##                      (user_df['timestamp'] <= user_df['timestamp'].iloc[i] + pd.Timedelta(time_window))
##        window = user_df[window_mask]
##        if len(window) > 1:  # Only consider windows with at least 2 events
##            windows.append(window)
##
##    return windows
##
##
### Apply to each user
##time_windows = defaultdict(list)
##for user_id, group in df.groupby('user_id'):
##    windows = generate_time_windows(group, time_window='5min')
##    time_windows[user_id] = windows
##
##print("\nTime-based windows (first user):")
##for i, window in enumerate(time_windows[1][:3]):  # Show first 3 windows for user 1
##    print(f"Window {i + 1}:")
##    print(window[['timestamp', 'page_url', 'dwell_time']])
##
##
##def extract_sequence_features(windows):
##    """Extract features from click sequences"""
##    features = []
##    for window in windows:
##        # Basic sequence features
##        seq_length = len(window)
##        unique_pages = len(set(window))
##
##        # Transition features
##        transitions = [f"{window[i]}=>{window[i + 1]}" for i in range(len(window) - 1)]
##
##        # Dwell time features (if available)
##        dwell_times = [dwell for dwell in window['dwell_time']] if isinstance(window, pd.DataFrame) else [0] * len(
##            window)
##
##        features.append({
##            'sequence': ' → '.join(window) if not isinstance(window, pd.DataFrame) else ' → '.join(window['page_url']),
##            'length': seq_length,
##            'unique_pages': unique_pages,
##            'transitions': transitions,
##            'avg_dwell_time': np.mean(dwell_times),
##            'total_dwell_time': np.sum(dwell_times)
##        })
##    return features
##
##
### Extract features for all users
##all_features = {}
##for user_id, windows in user_windows.items():
##    all_features[user_id] = extract_sequence_features(windows)
##
##print("\nExtracted features for user 1:")
##print(pd.DataFrame(all_features[1]).head())
##
##
##def build_transition_matrix(df):
##    """Build a Markov transition matrix for page transitions"""
##    # Create all possible transitions
##    df['next_page'] = df.groupby('user_id')['page_url'].shift(-1)
##    transitions = df.dropna(subset=['next_page'])
##
##    # Create transition matrix
##    all_pages = pd.unique(df['page_url'])
##    trans_matrix = pd.DataFrame(0, index=all_pages, columns=all_pages)
##
##    for _, row in transitions.iterrows():
##        trans_matrix.loc[row['page_url'], row['next_page']] += 1
##
##    # Normalize to probabilities
##    trans_matrix = trans_matrix.div(trans_matrix.sum(axis=1), axis=0).fillna(0)
##    return trans_matrix
##
##
##transition_matrix = build_transition_matrix(df)
##print("\nTransition Probability Matrix:")
##print(transition_matrix)
##
##
##def detect_sessions(df, inactivity_threshold='30min'):
##    """Identify browsing sessions based on inactivity"""
##    df = df.sort_values(['user_id', 'timestamp'])
##
##    # Calculate time difference between consecutive events
##    df['time_diff'] = df.groupby('user_id')['timestamp'].diff()
##
##    # New session starts when time difference > threshold or new user
##    df['new_session'] = (df['time_diff'] > pd.Timedelta(inactivity_threshold)) | (df['user_id'].diff() != 0)
##    df['session_id'] = df['new_session'].cumsum()
##
##    return df
##
##
##session_df = detect_sessions(df)
##print("\nSession detection results:")
##print(session_df[['user_id', 'timestamp', 'page_url', 'session_id']])
##
##from mlxtend.preprocessing import TransactionEncoder
##from mlxtend.frequent_patterns import apriori
##
### Prepare transaction data for pattern mining
##transactions = []
##for user, windows in user_windows.items():
##    for window in windows:
##        transactions.append(window)
##
### Convert to one-hot encoded format
##te = TransactionEncoder()
##te_ary = te.fit(transactions).transform(transactions)
##df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
##
### Find frequent patterns
##frequent_itemsets = apriori(df_encoded, min_support=0.3, use_colnames=True)
##print("\nFrequent click patterns:")
##print(frequent_itemsets.sort_values('support', ascending=False))
##
##import networkx as nx
##import matplotlib.pyplot as plt
##
### Create graph from transition matrix
##G = nx.DiGraph()
##
##for source in transition_matrix.index:
##    for target in transition_matrix.columns:
##        weight = transition_matrix.loc[source, target]
##        if weight > 0:
##            G.add_edge(source, target, weight=weight)
##
### Draw the graph
##plt.figure(figsize=(10, 8))
##pos = nx.spring_layout(G)
##nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='lightblue')
##nx.draw_networkx_edges(G, pos, edge_color='gray', width=[d['weight']*10 for (u,v,d) in G.edges(data=True)])
##nx.draw_networkx_labels(G, pos, font_size=12)
##edge_labels = {(u,v): f"{d['weight']:.2f}" for (u,v,d) in G.edges(data=True)}
##nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
##plt.title("User Click Transition Probabilities")
##plt.axis('off')
##plt.show()