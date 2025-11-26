"""
Feature Engineering Module for E.SUN AI Challenge 2025.

This module generates comprehensive features from raw transaction data,
including basic statistics, enhanced temporal features, graph-based network 
features, and alert account interaction features.

The feature engineering pipeline consists of four main parts:
    1. Basic Statistical Features (~20 features)
    2. Enhanced Behavioral Features (~15 features)
    3. Graph Network Features (~12 features)
    4. Alert Interaction Features (~6 features)

Total generated features: ~53 features

Usage:
    $ python Preprocess/feature_engineering.py [cutoff_date]
    
    cutoff_date: Optional temporal cutoff for time-series validation
                 Default is 121 (uses all data)

Example:
    $ python Preprocess/feature_engineering.py 121
    
Output:
    features/features.csv: Complete feature set with all accounts

Author: E.SUN AI Challenge Team
Date: 2025
"""

import pandas as pd
import numpy as np
import networkx as nx
import sys
import os
import warnings
from collections import Counter
warnings.filterwarnings('ignore')


def cross_ratio(s):
    """
    Calculate the ratio of cross-bank transactions.
    
    Cross-bank transactions are those with account type = 2 (Other banks),
    as opposed to type = 1 (E.SUN bank).
    
    Args:
        s (pd.Series): Series of account types (1=E.SUN, 2=Other banks)
    
    Returns:
        float: Ratio of cross-bank transactions (type=2) among all valid types.
               Returns 0.0 if no valid account types exist.
    
    Example:
        >>> s = pd.Series([1, 1, 2, 2, 2])
        >>> cross_ratio(s)
        0.6
    """
    valid = s[s.isin([1, 2])]
    return float((valid == 2).mean()) if len(valid) else 0.0


def generate_basic_features(df_txn):
    """
    Generate basic statistical features from transaction data.
    
    Creates aggregated features for both outgoing (from_acct) and 
    incoming (to_acct) transactions, including:
    - Transaction counts
    - Amount statistics (sum, mean, std, max, min)
    - Number of unique transaction partners
    - Cross-bank transaction ratios
    - Self-transaction ratios
    
    Args:
        df_txn (pd.DataFrame): Preprocessed transaction data with columns:
            - from_acct: Source account ID
            - to_acct: Destination account ID
            - txn_amt: Transaction amount
            - from_acct_type, to_acct_type: Account types
            - is_self_txn: Self-transaction indicator
    
    Returns:
        pd.DataFrame: Feature dataframe with columns:
            - acct: Account ID
            - out_cnt, in_cnt: Transaction counts
            - out_amt_sum, in_amt_sum: Total amounts
            - out_amt_mean, in_amt_mean: Average amounts
            - out_amt_std, in_amt_std: Amount standard deviations
            - out_amt_max, in_amt_max: Maximum amounts
            - out_amt_min, in_amt_min: Minimum amounts
            - unique_to_acct, unique_from_acct: Number of partners
            - cross_bank_out_ratio, cross_bank_in_ratio: Cross-bank ratios
            - self_txn_ratio: Self-transaction ratio
            - Derived features: out_in_cnt_ratio, out_in_amt_ratio, etc.
    """
    print("生成基礎特徵...")
    
    # Outgoing transaction features
    out_feat = (
        df_txn.groupby("from_acct")
        .agg(
            out_cnt=("txn_amt", "count"),
            out_amt_sum=("txn_amt", "sum"),
            out_amt_mean=("txn_amt", "mean"),
            out_amt_std=("txn_amt", "std"),
            out_amt_max=("txn_amt", "max"),
            out_amt_min=("txn_amt", "min"),
            unique_to_acct=("to_acct", "nunique"),
            cross_bank_out_ratio=("to_acct_type", cross_ratio),
            self_txn_ratio=("is_self_txn", "mean"),
        )
        .reset_index()
        .rename(columns={"from_acct": "acct"})
    )
    
    # Incoming transaction features
    in_feat = (
        df_txn.groupby("to_acct")
        .agg(
            in_cnt=("txn_amt", "count"),
            in_amt_sum=("txn_amt", "sum"),
            in_amt_mean=("txn_amt", "mean"),
            in_amt_std=("txn_amt", "std"),
            in_amt_max=("txn_amt", "max"),
            in_amt_min=("txn_amt", "min"),
            unique_from_acct=("from_acct", "nunique"),
            cross_bank_in_ratio=("from_acct_type", cross_ratio),
        )
        .reset_index()
        .rename(columns={"to_acct": "acct"})
    )
    
    # Merge features
    feat = out_feat.merge(in_feat, on="acct", how="outer")
    
    # Fill missing values
    num_cols = [c for c in feat.columns if c != 'acct']
    feat[num_cols] = feat[num_cols].fillna(0)
    
    # Derive additional features
    feat["out_in_cnt_ratio"] = feat["out_cnt"] / (feat["in_cnt"] + 1e-6)
    feat["out_in_amt_ratio"] = feat["out_amt_sum"] / (feat["in_amt_sum"] + 1e-6)
    feat["partner_cnt"] = feat["unique_to_acct"] + feat["unique_from_acct"]
    feat["cross_bank_ratio"] = (feat["cross_bank_out_ratio"] + feat["cross_bank_in_ratio"]) / 2
    feat["Connection_Ratio"] = np.where(
        feat["partner_cnt"] > 0,
        feat["partner_cnt"] / (feat["out_cnt"] + feat["in_cnt"] + 1e-6),
        0
    )
    
    return feat


def generate_enhanced_features(df_txn, feat, cutoff_date):
    """
    Generate enhanced behavioral and temporal features.
    
    Creates advanced features including:
    - date_similarity: Transaction timing regularity (key feature)
    - Temporal statistics: Early vs late activity patterns
    - Amount trends: Coefficient of variation, range ratios
    - Abnormal time periods: Night transactions, weekend transactions
    - High-order interactions: Average transactions per partner, large transaction ratios
    
    Args:
        df_txn (pd.DataFrame): Preprocessed transaction data
        feat (pd.DataFrame): Existing feature dataframe to merge with
        cutoff_date (int): Temporal cutoff for calculating early/late activity
    
    Returns:
        pd.DataFrame: Enhanced feature dataframe with additional columns:
            - date_similarity: Active days / total transactions
            - active_days, active_days_out, active_days_in: Activity days
            - early_out_cnt, late_out_cnt: Early/late period transactions
            - activity_shift: Late/early activity ratio
            - out_amt_cv, in_amt_cv: Amount coefficient of variation
            - out_amt_range, out_amt_range_ratio: Amount range features
            - night_txn_ratio_out: Night transaction ratio (22:00-06:00)
            - weekend_txn_ratio: Weekend transaction ratio
            - avg_txn_per_partner: Average transactions per partner
            - large_txn_cnt, large_txn_ratio: Large transaction features
    
    Notes:
        - date_similarity is identified as a key feature from high-scoring teams
        - Night transactions are defined as 22:00-06:00
        - Large transactions are defined as > 75th percentile
    """
    print("生成增強特徵...")
    
    # 1. Date Similarity (Key Feature)
    date_variety_out = df_txn.groupby('from_acct')['txn_date'].nunique().reset_index()
    date_variety_out.columns = ['acct', 'active_days_out']
    
    date_variety_in = df_txn.groupby('to_acct')['txn_date'].nunique().reset_index()
    date_variety_in.columns = ['acct', 'active_days_in']
    
    feat = feat.merge(date_variety_out, on='acct', how='left')
    feat = feat.merge(date_variety_in, on='acct', how='left')
    
    feat['active_days_out'] = feat['active_days_out'].fillna(0)
    feat['active_days_in'] = feat['active_days_in'].fillna(0)
    feat['active_days'] = feat['active_days_out'] + feat['active_days_in']
    
    feat['date_similarity'] = np.where(
        (feat['out_cnt'] + feat['in_cnt']) > 0,
        feat['active_days'] / (feat['out_cnt'] + feat['in_cnt'] + 1e-6),
        0
    )
    
    # 2. Temporal Statistics
    mid_point = cutoff_date // 2
    
    early_out = df_txn[df_txn['txn_date'] <= mid_point].groupby('from_acct').size().reset_index()
    early_out.columns = ['acct', 'early_out_cnt']
    
    late_out = df_txn[df_txn['txn_date'] > mid_point].groupby('from_acct').size().reset_index()
    late_out.columns = ['acct', 'late_out_cnt']
    
    feat = feat.merge(early_out, on='acct', how='left')
    feat = feat.merge(late_out, on='acct', how='left')
    
    feat['early_out_cnt'] = feat['early_out_cnt'].fillna(0)
    feat['late_out_cnt'] = feat['late_out_cnt'].fillna(0)
    
    feat['activity_shift'] = np.where(
        feat['early_out_cnt'] > 0,
        feat['late_out_cnt'] / (feat['early_out_cnt'] + 1e-6),
        0
    )
    
    # 3. Amount Trends
    feat['out_amt_cv'] = np.where(
        feat['out_amt_mean'] > 0,
        feat['out_amt_std'] / (feat['out_amt_mean'] + 1e-6),
        0
    )
    
    feat['in_amt_cv'] = np.where(
        feat['in_amt_mean'] > 0,
        feat['in_amt_std'] / (feat['in_amt_mean'] + 1e-6),
        0
    )
    
    feat['out_amt_range'] = feat['out_amt_max'] - feat['out_amt_min']
    feat['out_amt_range_ratio'] = np.where(
        feat['out_amt_mean'] > 0,
        feat['out_amt_range'] / (feat['out_amt_mean'] + 1e-6),
        0
    )
    
    # 4. Abnormal Time Periods
    df_txn['hour'] = df_txn['txn_time'].astype(str).str.zfill(6).str[:2].astype(int)
    df_txn['is_night'] = ((df_txn['hour'] >= 22) | (df_txn['hour'] < 6)).astype(int)
    
    night_out = df_txn.groupby('from_acct')['is_night'].agg(['sum', 'count']).reset_index()
    night_out.columns = ['acct', 'night_out_cnt', 'total_out']
    night_out['night_txn_ratio_out'] = night_out['night_out_cnt'] / (night_out['total_out'] + 1e-6)
    
    feat = feat.merge(night_out[['acct', 'night_txn_ratio_out']], on='acct', how='left')
    feat['night_txn_ratio_out'] = feat['night_txn_ratio_out'].fillna(0)
    
    # Weekend transactions
    df_txn['weekday'] = (df_txn['txn_date'] % 7)
    df_txn['is_weekend'] = (df_txn['weekday'].isin([0, 6])).astype(int)
    
    weekend_out = df_txn.groupby('from_acct')['is_weekend'].mean().reset_index()
    weekend_out.columns = ['acct', 'weekend_txn_ratio']
    
    feat = feat.merge(weekend_out, on='acct', how='left')
    feat['weekend_txn_ratio'] = feat['weekend_txn_ratio'].fillna(0)
    
    # 5. High-order Interactions
    feat['avg_txn_per_partner'] = np.where(
        feat['partner_cnt'] > 0,
        (feat['out_cnt'] + feat['in_cnt']) / (feat['partner_cnt'] + 1e-6),
        0
    )
    
    large_out = df_txn[df_txn['txn_amt'] > df_txn['txn_amt'].quantile(0.75)]
    large_out_cnt = large_out.groupby('from_acct').size().reset_index()
    large_out_cnt.columns = ['acct', 'large_txn_cnt']
    
    feat = feat.merge(large_out_cnt, on='acct', how='left')
    feat['large_txn_cnt'] = feat['large_txn_cnt'].fillna(0)
    feat['large_txn_ratio'] = np.where(
        feat['out_cnt'] > 0,
        feat['large_txn_cnt'] / (feat['out_cnt'] + 1e-6),
        0
    )
    
    return feat


def generate_graph_features(df_txn, alert_accts):
    """
    Generate graph-based network features from transaction relationships.
    
    Constructs a directed transaction network and calculates various
    centrality metrics and community-based features:
    - Degree centrality (in/out/total)
    - PageRank (importance in network)
    - Clustering coefficient (local density)
    - Betweenness centrality (bridge importance)
    - Weighted degrees (amount-based)
    - K-core number (network core membership)
    - Community detection and alert ratio within community
    
    Args:
        df_txn (pd.DataFrame): Transaction data with from_acct, to_acct, txn_amt
        alert_accts (set): Set of known alert account IDs
    
    Returns:
        pd.DataFrame: Graph feature dataframe with columns:
            - acct: Account ID
            - in_degree, out_degree, total_degree: Node degrees
            - pagerank: PageRank score
            - clustering_coef: Clustering coefficient
            - betweenness: Betweenness centrality (sampled for large networks)
            - weighted_in_degree, weighted_out_degree: Amount-weighted degrees
            - k_core: K-core number
            - community_id: Community assignment
            - comm_alert_ratio: Alert account ratio in community
    
    Notes:
        - For networks > 5000 nodes, betweenness is calculated on sample
        - For networks > 50000 nodes, community detection is skipped
        - Uses greedy modularity optimization for community detection
        - All missing values filled with 0
    """
    print("建立交易網絡...")
    G = nx.DiGraph()
    
    # Build network edges with weights
    for _, row in df_txn.iterrows():
        if G.has_edge(row['from_acct'], row['to_acct']):
            G[row['from_acct']][row['to_acct']]['weight'] += row['txn_amt']
            G[row['from_acct']][row['to_acct']]['count'] += 1
        else:
            G.add_edge(row['from_acct'], row['to_acct'], weight=row['txn_amt'], count=1)
    
    print(f"節點: {G.number_of_nodes():,}")
    print(f"邊: {G.number_of_edges():,}")
    
    # 1. Degree Centrality
    print("\n計算度中心性...")
    in_degree = dict(G.in_degree())
    out_degree = dict(G.out_degree())
    degree = {node: in_degree.get(node, 0) + out_degree.get(node, 0) for node in G.nodes()}
    
    # 2. PageRank
    print("計算PageRank...")
    try:
        pagerank = nx.pagerank(G, alpha=0.85, max_iter=100, weight='weight')
        print(f"  PageRank完成")
    except:
        pagerank = {node: 0 for node in G.nodes()}
        print(f"  PageRank失敗，使用預設值")
    
    # 3. Clustering Coefficient
    print("計算聚類係數...")
    G_undirected = G.to_undirected()
    clustering = nx.clustering(G_undirected)
    
    # 4. Betweenness Centrality (sampled for large networks)
    print("計算介數中心性（採樣）...")
    if G.number_of_nodes() > 5000:
        sample_nodes = np.random.choice(list(G.nodes()), size=min(5000, G.number_of_nodes()), replace=False)
        betweenness = nx.betweenness_centrality(G.subgraph(sample_nodes), weight='weight')
        betweenness = {node: betweenness.get(node, 0) for node in G.nodes()}
    else:
        betweenness = nx.betweenness_centrality(G, weight='weight')
    
    # 5. Weighted Degrees
    print("計算加權度...")
    weighted_in = {}
    weighted_out = {}
    for node in G.nodes():
        weighted_in[node] = sum(G[u][node]['weight'] for u in G.predecessors(node))
        weighted_out[node] = sum(G[node][v]['weight'] for v in G.successors(node))
    
    # 6. K-core Number
    print("計算核心數...")
    try:
        k_core = nx.core_number(G_undirected)
    except:
        k_core = {node: 0 for node in G.nodes()}
    
    # 7. Community Detection
    print("社群檢測...")
    if G_undirected.number_of_nodes() <= 50000:
        try:
            from networkx.algorithms import community
            communities = community.greedy_modularity_communities(G_undirected)
            community_map = {}
            for idx, comm in enumerate(communities):
                for node in comm:
                    community_map[node] = idx
            
            # Calculate alert ratio within each community
            comm_alert_ratio = {}
            for node in G.nodes():
                comm_id = community_map.get(node, -1)
                if comm_id >= 0:
                    comm_members = [n for n, c in community_map.items() if c == comm_id]
                    alert_in_comm = sum(1 for n in comm_members if n in alert_accts)
                    comm_alert_ratio[node] = alert_in_comm / len(comm_members) if len(comm_members) > 0 else 0
                else:
                    comm_alert_ratio[node] = 0
            
            print(f"  檢測到{len(communities)}個社群")
        except:
            community_map = {node: 0 for node in G.nodes()}
            comm_alert_ratio = {node: 0 for node in G.nodes()}
            print(f"  社群檢測失敗")
    else:
        community_map = {node: 0 for node in G.nodes()}
        comm_alert_ratio = {node: 0 for node in G.nodes()}
        print(f"  節點過多，跳過社群檢測")
    
    # Consolidate graph features
    graph_feat = pd.DataFrame({
        'acct': list(G.nodes()),
        'in_degree': [in_degree.get(node, 0) for node in G.nodes()],
        'out_degree': [out_degree.get(node, 0) for node in G.nodes()],
        'total_degree': [degree.get(node, 0) for node in G.nodes()],
        'pagerank': [pagerank.get(node, 0) for node in G.nodes()],
        'clustering_coef': [clustering.get(node, 0) for node in G.nodes()],
        'betweenness': [betweenness.get(node, 0) for node in G.nodes()],
        'weighted_in_degree': [weighted_in.get(node, 0) for node in G.nodes()],
        'weighted_out_degree': [weighted_out.get(node, 0) for node in G.nodes()],
        'k_core': [k_core.get(node, 0) for node in G.nodes()],
        'community_id': [community_map.get(node, 0) for node in G.nodes()],
        'comm_alert_ratio': [comm_alert_ratio.get(node, 0) for node in G.nodes()]
    })
    
    return graph_feat


def generate_alert_interaction_features(df_txn, feat, alert_accts):
    """
    Generate features based on interactions with known alert accounts.
    
    Creates features capturing direct relationships with alert accounts:
    - Binary indicators of sending/receiving from alert accounts
    - Counts of transactions with alert accounts
    - Ratios of alert transactions to total transactions
    
    Args:
        df_txn (pd.DataFrame): Transaction data
        feat (pd.DataFrame): Existing feature dataframe
        alert_accts (set): Set of known alert account IDs
    
    Returns:
        pd.DataFrame: Feature dataframe with additional columns:
            - has_sent_to_alert: Whether sent money to alert accounts (0/1)
            - has_recv_from_alert: Whether received from alert accounts (0/1)
            - alert_send_count: Number of transactions sent to alert accounts
            - alert_recv_count: Number of transactions received from alert accounts
            - ratio_alert_send: Alert send count / total send count
            - ratio_alert_recv: Alert receive count / total receive count
    
    Notes:
        - These features are highly predictive as fraud often forms networks
        - All missing values filled with 0
    """
    print("生成警示帳戶交互特徵...")
    
    df_txn['from_is_alert'] = df_txn['from_acct'].isin(alert_accts).astype(int)
    df_txn['to_is_alert'] = df_txn['to_acct'].isin(alert_accts).astype(int)
    
    has_alert_send = df_txn.groupby('from_acct')['to_is_alert'].max().reset_index()
    has_alert_send.columns = ['acct', 'has_sent_to_alert']
    
    has_alert_recv = df_txn.groupby('to_acct')['from_is_alert'].max().reset_index()
    has_alert_recv.columns = ['acct', 'has_recv_from_alert']
    
    alert_send_cnt = df_txn[df_txn['to_is_alert']==1].groupby('from_acct').size().reset_index()
    alert_send_cnt.columns = ['acct', 'alert_send_count']
    
    alert_recv_cnt = df_txn[df_txn['from_is_alert']==1].groupby('to_acct').size().reset_index()
    alert_recv_cnt.columns = ['acct', 'alert_recv_count']
    
    feat = feat.merge(has_alert_send, on='acct', how='left')
    feat = feat.merge(has_alert_recv, on='acct', how='left')
    feat = feat.merge(alert_send_cnt, on='acct', how='left')
    feat = feat.merge(alert_recv_cnt, on='acct', how='left')
    
    alert_cols = ['has_sent_to_alert', 'has_recv_from_alert', 
                  'alert_send_count', 'alert_recv_count']
    feat[alert_cols] = feat[alert_cols].fillna(0)
    
    feat['ratio_alert_send'] = np.where(
        feat['out_cnt'] > 0,
        feat['alert_send_count'] / feat['out_cnt'],
        0
    )
    feat['ratio_alert_recv'] = np.where(
        feat['in_cnt'] > 0,
        feat['alert_recv_count'] / feat['in_cnt'],
        0
    )
    
    return feat


def main():
    """
    Main execution function for feature engineering pipeline.
    
    Orchestrates the complete feature generation process:
    1. Load and preprocess transaction data
    2. Generate basic statistical features
    3. Generate enhanced behavioral features
    4. Generate graph network features
    5. Generate alert interaction features
    6. Save complete feature set
    
    Command-line Args:
        cutoff_date (int, optional): Temporal cutoff for validation.
                                     Default is 121 (use all data)
    
    Outputs:
        features/features.csv: Complete feature set with ~53 features
    
    Raises:
        SystemExit: If data files are not found
    """
    # ========================================================================
    # Parameter Configuration
    # ========================================================================
    if len(sys.argv) > 1:
        CUTOFF_DATE = int(sys.argv[1])
        print(f"\nCutoff Date: {CUTOFF_DATE}")
    else:
        CUTOFF_DATE = 121
        print(f"\n預設 Cutoff Date: {CUTOFF_DATE}")
    
    # ========================================================================
    # Load Data
    # ========================================================================
    print("\n載入資料...")
    
    if os.path.exists('/mnt/project/acct_transaction.csv'):
        DATA_PATH = '/mnt/project'
    elif os.path.exists('data/acct_transaction.csv'):
        DATA_PATH = 'data'
    else:
        DATA_PATH = '.'
    
    df_txn_orig = pd.read_csv(f'{DATA_PATH}/acct_transaction.csv')
    df_alert = pd.read_csv(f'{DATA_PATH}/acct_alert.csv')
    df_test = pd.read_csv(f'{DATA_PATH}/acct_predict.csv')
    
    alert_accts = set(df_alert['acct'].unique())
    
    print(f"交易: {len(df_txn_orig):,}")
    print(f"警示: {len(df_alert):,}")
    print(f"預測: {len(df_test):,}")
    
    # Temporal filtering
    df_txn = df_txn_orig[df_txn_orig['txn_date'] <= CUTOFF_DATE].copy()
    print(f"過濾後交易: {len(df_txn):,}")
    
    # Data preprocessing
    df_txn["txn_amt"] = pd.to_numeric(df_txn["txn_amt"], errors="coerce")
    df_txn["is_self_txn"] = pd.to_numeric(df_txn["is_self_txn"], errors="coerce")
    df_txn["from_acct_type"] = pd.to_numeric(df_txn["from_acct_type"], errors="coerce")
    df_txn["to_acct_type"] = pd.to_numeric(df_txn["to_acct_type"], errors="coerce")
    
    # ========================================================================
    # Part 1: Basic Statistical Features
    # ========================================================================
    print("\n" + "="*80)
    print("Part 1: 基礎統計特徵")
    print("="*80)
    
    feat = generate_basic_features(df_txn)
    print(f"基礎特徵: {feat.shape}")
    
    # ========================================================================
    # Part 2: Enhanced Features
    # ========================================================================
    print("\n" + "="*80)
    print("Part 2: 增強特徵")
    print("="*80)
    
    feat = generate_enhanced_features(df_txn, feat, CUTOFF_DATE)
    print(f"增強後特徵: {feat.shape}")
    
    # ========================================================================
    # Part 3: Graph Features
    # ========================================================================
    print("\n" + "="*80)
    print("Part 3: 圖特徵生成")
    print("="*80)
    
    graph_feat = generate_graph_features(df_txn, alert_accts)
    print(f"圖特徵: {graph_feat.shape}")
    
    # Merge graph features
    feat = feat.merge(graph_feat, on='acct', how='left')
    graph_cols = [c for c in graph_feat.columns if c != 'acct']
    feat[graph_cols] = feat[graph_cols].fillna(0)
    
    print(f"合併圖特徵後: {feat.shape}")
    
    # ========================================================================
    # Part 4: Alert Interaction Features
    # ========================================================================
    print("\n" + "="*80)
    print("Part 4: 警示帳戶交互特徵")
    print("="*80)
    
    feat = generate_alert_interaction_features(df_txn, feat, alert_accts)
    
    # ========================================================================
    # Final Processing
    # ========================================================================
    feat = feat.replace([np.inf, -np.inf], 0)
    feat = feat.fillna(0)
    
    print(f"\n最終特徵維度: {feat.shape}")
    print(f"特徵數量: {feat.shape[1]-1}")
    
    # Save features
    os.makedirs('features', exist_ok=True)
    output_file = 'features/features.csv'
    feat.to_csv(output_file, index=False)
    
    print(f"\n已儲存: {output_file}")
    print(f"  總計: {feat.shape[1]-1}個特徵")


if __name__ == "__main__":
    main()