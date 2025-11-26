"""
Exploratory Data Analysis Module for E.SUN AI Challenge 2025.

This module performs comprehensive exploratory data analysis on transaction
data to understand patterns, distributions, and relationships between alert
and normal accounts.

The EDA pipeline generates:
    - 8 visualization charts analyzing different aspects of the data
    - Statistical tests comparing alert vs normal accounts
    - Network analysis with PageRank scores
    - Comprehensive analysis report

Charts Generated:
    1. 01_amount_distribution.png: Transaction amount distributions (TWD/Forex)
    2. 02_categorical_distribution.png: Account types, channels, currencies
    3. 03_time_distribution.png: Daily transaction and alert volumes
    4. 04_alert_vs_normal_comparison.png: Statistical comparisons
    5. 05_time_series_trends.png: Temporal patterns
    6. 06_hourly_distribution.png: Transaction time patterns
    7. 07_outlier_detection.png: Anomaly detection
    8. 08_network_visualization.png: Transaction network structure

Outputs:
    visualizations/*.png: Visualization charts
    visualizations/EDA_REPORT.md: Analysis report
    features/pagerank_scores.csv: PageRank network features (optional)

Usage:
    $ python EDA/exploratory_analysis.py

Requirements:
    - pandas, numpy, matplotlib, seaborn
    - networkx (for network analysis)
    - scipy (for statistical tests)

Note:
    This is an optional exploratory step and does not affect model training.
    The analysis helps understand data characteristics and validate assumptions.

Execution Time:
    Approximately 2-3 minutes depending on data size and network complexity.

Author: E.SUN AI Challenge Team
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# Set Chinese font for matplotlib
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def load_transaction_data():
    """
    Load transaction, alert, and prediction data from CSV files.
    
    Attempts to load data from 'data/' directory. Prints basic statistics
    about the loaded datasets.
    
    Returns:
        tuple: (df_txn, df_alert, df_test)
            - df_txn (pd.DataFrame): Transaction data with columns:
                from_acct, to_acct, txn_amt, txn_date, txn_time, etc.
            - df_alert (pd.DataFrame): Known alert accounts with columns:
                acct, event_date, label
            - df_test (pd.DataFrame): Accounts to predict with column:
                acct
    
    Raises:
        FileNotFoundError: If data files are not found in expected location
    
    Example:
        >>> df_txn, df_alert, df_test = load_transaction_data()
        交易資料: 4,400,000 筆
        警示帳戶: 1,004 個
        待預測帳戶: 4,779 個
    """
    print("\n載入資料...")
    df_txn = pd.read_csv('data/acct_transaction.csv')
    df_alert = pd.read_csv('data/acct_alert.csv')
    df_test = pd.read_csv('data/acct_predict.csv')
    
    print(f"交易資料: {len(df_txn):,} 筆")
    print(f"警示帳戶: {len(df_alert):,} 個")
    print(f"待預測帳戶: {len(df_test):,} 個")
    
    return df_txn, df_alert, df_test


def preprocess_and_label_data(df_txn, df_alert):
    """
    Preprocess transaction data and add alert account indicators.
    
    Creates binary indicators for whether each transaction involves
    alert accounts as sender, receiver, or either.
    
    Args:
        df_txn (pd.DataFrame): Raw transaction data
        df_alert (pd.DataFrame): Known alert accounts
    
    Returns:
        tuple: (df_txn, alert_accts)
            - df_txn (pd.DataFrame): Transaction data with added columns:
                from_is_alert, to_is_alert, any_alert
            - alert_accts (set): Set of alert account IDs for quick lookup
    
    Added Columns:
        - from_is_alert (int): 1 if sender is alert account, 0 otherwise
        - to_is_alert (int): 1 if receiver is alert account, 0 otherwise
        - any_alert (int): 1 if either sender or receiver is alert account
    
    Statistics Printed:
        - Number and percentage of transactions involving alert accounts
    """
    print("\n" + "="*80)
    print("資料預處理")
    print("="*80)
    
    # Build alert account set for fast lookup
    alert_accts = set(df_alert['acct'].unique())
    
    # Add alert indicators
    df_txn['from_is_alert'] = df_txn['from_acct'].isin(alert_accts).astype(int)
    df_txn['to_is_alert'] = df_txn['to_acct'].isin(alert_accts).astype(int)
    df_txn['any_alert'] = ((df_txn['from_is_alert'] == 1) | 
                           (df_txn['to_is_alert'] == 1)).astype(int)
    
    print(f"涉及警示帳戶的交易: {df_txn['any_alert'].sum():,} 筆 "
          f"({df_txn['any_alert'].sum()/len(df_txn)*100:.2f}%)")
    
    return df_txn, alert_accts


def generate_amount_distribution_chart(df_txn):
    """
    Generate transaction amount distribution visualization.
    
    Creates a 3-panel chart showing:
    1. TWD transaction amount distribution (linear scale)
    2. TWD transaction amount distribution (log scale)
    3. Foreign exchange transaction distribution (log scale)
    
    Args:
        df_txn (pd.DataFrame): Transaction data with columns:
            txn_amt, currency_type
    
    Output:
        Saves chart to: visualizations/01_amount_distribution.png
    
    Chart Features:
        - 50 bins for histograms
        - Log scale for better visualization of wide range
        - Separate analysis for TWD and foreign currencies
    
    Notes:
        Most transactions are in TWD (~99%), foreign exchange analyzed
        separately due to different scale and characteristics.
    """
    print("\n生成圖1: 交易金額分佈...")
    
    # Filter TWD transactions
    df_txn_twd = df_txn[df_txn['currency_type'] == 'TWD']
    print(f"新台幣交易: {len(df_txn_twd):,} 筆 ({len(df_txn_twd)/len(df_txn)*100:.2f}%)")
    
    # Filter forex transactions
    df_txn_forex = df_txn[df_txn['currency_type'] != 'TWD']
    print(f"外匯交易: {len(df_txn_forex):,} 筆 ({len(df_txn_forex)/len(df_txn)*100:.2f}%)")
    if len(df_txn_forex) > 0:
        print(f"外匯幣別: {df_txn_forex['currency_type'].unique()}")
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    
    # Subplot 1: TWD raw distribution
    axes[0].hist(df_txn_twd['txn_amt'], bins=50, color='steelblue', 
                 edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Transaction Amount (TWD)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('TWD Transaction Amount Distribution', 
                      fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Subplot 2: TWD log scale distribution
    axes[1].hist(np.log10(df_txn_twd['txn_amt'] + 1), bins=50, color='coral', 
                 edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Log10(Transaction Amount in TWD)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('TWD Amount Distribution (Log Scale)', 
                      fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # Subplot 3: Forex distribution (log scale)
    if len(df_txn_forex) > 0:
        axes[2].hist(np.log10(df_txn_forex['txn_amt'] + 1), bins=50, 
                     color='lightgreen', edgecolor='black', alpha=0.7)
        axes[2].set_xlabel('Log10(Transaction Amount)', fontsize=12)
        axes[2].set_ylabel('Frequency', fontsize=12)
        axes[2].set_title('Foreign Exchange Amount Distribution (Log Scale)', 
                         fontsize=14, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, 'No Foreign Exchange Transactions', 
                    ha='center', va='center', fontsize=14)
        axes[2].set_title('Foreign Exchange Amount Distribution', 
                         fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('visualizations/01_amount_distribution.png', dpi=300, bbox_inches='tight')
    print("已儲存: visualizations/01_amount_distribution.png")
    plt.close()


def generate_categorical_distribution_chart(df_txn):
    """
    Generate categorical variable distribution visualization.
    
    Creates a 4-panel chart showing distributions of:
    1. Account type (E.SUN vs Other banks)
    2. Transaction channel (ATM, Branch, Mobile, etc.)
    3. Currency type (TWD, USD, etc.)
    4. Self-transaction ratio (pie chart)
    
    Args:
        df_txn (pd.DataFrame): Transaction data with columns:
            from_acct_type, channel_type, currency_type, is_self_txn
    
    Output:
        Saves chart to: visualizations/02_categorical_distribution.png
    
    Chart Types:
        - Bar charts for account type and currency
        - Horizontal bar chart for channel type
        - Pie chart for self-transaction ratio
    
    Notes:
        Channel types: 1=ATM, 2=Branch, 3=Mobile, 4=Web, 5=Voice, etc.
        Self-transactions indicate possible money laundering patterns.
    """
    print("\n生成圖2: 類別變數分佈...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 2.1 Account type distribution
    type_counts = df_txn['from_acct_type'].value_counts()
    axes[0, 0].bar(type_counts.index.astype(str), type_counts.values, 
                   color='steelblue', edgecolor='black')
    axes[0, 0].set_xlabel('Account Type', fontsize=12)
    axes[0, 0].set_ylabel('Count', fontsize=12)
    axes[0, 0].set_title('Account Type Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].set_xticks(range(len(type_counts)))
    axes[0, 0].set_xticklabels(['E.SUN (01)', 'Other (02)'])
    
    # 2.2 Channel distribution
    channel_counts = df_txn['channel_type'].value_counts()
    axes[0, 1].barh(range(len(channel_counts)), channel_counts.values, 
                    color='coral', edgecolor='black')
    axes[0, 1].set_yticks(range(len(channel_counts)))
    axes[0, 1].set_yticklabels(channel_counts.index)
    axes[0, 1].set_xlabel('Count', fontsize=12)
    axes[0, 1].set_title('Channel Type Distribution', fontsize=14, fontweight='bold')
    
    # 2.3 Currency distribution
    currency_counts = df_txn['currency_type'].value_counts()
    axes[1, 0].bar(range(len(currency_counts)), currency_counts.values, 
                   color='lightgreen', edgecolor='black')
    axes[1, 0].set_xlabel('Currency Type', fontsize=12)
    axes[1, 0].set_ylabel('Count', fontsize=12)
    axes[1, 0].set_title('Currency Distribution', fontsize=14, fontweight='bold')
    axes[1, 0].set_xticks(range(len(currency_counts)))
    axes[1, 0].set_xticklabels(currency_counts.index, rotation=45)
    
    # 2.4 Self-transaction ratio
    self_txn_counts = df_txn['is_self_txn'].value_counts()
    axes[1, 1].pie(self_txn_counts.values, labels=self_txn_counts.index, 
                   autopct='%1.1f%%',
                   colors=['#ff9999', '#66b3ff', '#99ff99'], startangle=90)
    axes[1, 1].set_title('Self Transaction Ratio', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('visualizations/02_categorical_distribution.png', dpi=300, bbox_inches='tight')
    print("已儲存: visualizations/02_categorical_distribution.png")
    plt.close()


def generate_time_distribution_chart(df_txn, df_alert):
    """
    Generate temporal distribution visualization.
    
    Creates a 2-panel chart showing:
    1. Daily transaction volume over time
    2. Daily alert account volume over time
    
    Args:
        df_txn (pd.DataFrame): Transaction data with txn_date column
        df_alert (pd.DataFrame): Alert data with event_date column
    
    Output:
        Saves chart to: visualizations/03_time_distribution.png
    
    Chart Features:
        - Line plots with filled areas
        - Transaction dates normalized (day 1 = first day)
        - Helps identify temporal patterns and seasonality
    
    Insights:
        - Transaction volumes may show weekly/monthly patterns
        - Alert account spikes may indicate fraud waves
    """
    print("\n生成圖3: 時間分佈...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # 3.1 Daily transaction distribution
    date_counts = df_txn['txn_date'].value_counts().sort_index()
    axes[0].plot(date_counts.index, date_counts.values, color='steelblue', 
                 linewidth=2, marker='o')
    axes[0].fill_between(date_counts.index, date_counts.values, alpha=0.3)
    axes[0].set_xlabel('Transaction Date (Day)', fontsize=12)
    axes[0].set_ylabel('Number of Transactions', fontsize=12)
    axes[0].set_title('Daily Transaction Volume', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # 3.2 Daily alert distribution
    alert_date_counts = df_alert['event_date'].value_counts().sort_index()
    axes[1].plot(alert_date_counts.index, alert_date_counts.values, color='red', 
                 linewidth=2, marker='s')
    axes[1].fill_between(alert_date_counts.index, alert_date_counts.values, 
                        alpha=0.3, color='red')
    axes[1].set_xlabel('Alert Date (Day)', fontsize=12)
    axes[1].set_ylabel('Number of Alerts', fontsize=12)
    axes[1].set_title('Daily Alert Account Volume', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('visualizations/03_time_distribution.png', dpi=300, bbox_inches='tight')
    print("已儲存: visualizations/03_time_distribution.png")
    plt.close()


def prepare_account_level_features(df_txn, alert_accts):
    """
    Prepare account-level aggregated features for comparison analysis.
    
    Aggregates transaction-level data to account level, computing statistics
    for both sending and receiving behavior.
    
    Args:
        df_txn (pd.DataFrame): Transaction data
        alert_accts (set): Set of alert account IDs
    
    Returns:
        pd.DataFrame: Account-level features with columns:
            - acct: Account ID
            - is_alert: Binary indicator (1=alert, 0=normal)
            - total_send, avg_send, std_send: Send amount statistics
            - send_count: Number of outgoing transactions
            - max_send, min_send: Send amount extremes
            - unique_recipients: Number of unique receiving accounts
            - date_range: Transaction date range (max - min)
            - total_recv, avg_recv: Receive amount statistics
            - recv_count: Number of incoming transactions
    
    Notes:
        Missing values filled with 0 for accounts with no activity
        in a particular direction (only send or only receive).
    """
    print("\n準備帳戶級別特徵...")
    
    # Sender statistics
    send_stats = df_txn.groupby('from_acct').agg({
        'txn_amt': ['sum', 'mean', 'std', 'count', 'max', 'min'],
        'to_acct': 'nunique',
        'txn_date': lambda x: x.max() - x.min()
    }).reset_index()
    send_stats.columns = ['acct', 'total_send', 'avg_send', 'std_send', 'send_count', 
                          'max_send', 'min_send', 'unique_recipients', 'date_range']
    send_stats['is_alert'] = send_stats['acct'].isin(alert_accts).astype(int)
    
    # Receiver statistics
    recv_stats = df_txn.groupby('to_acct').agg({
        'txn_amt': ['sum', 'mean', 'count'],
    }).reset_index()
    recv_stats.columns = ['acct', 'total_recv', 'avg_recv', 'recv_count']
    recv_stats['is_alert'] = recv_stats['acct'].isin(alert_accts).astype(int)
    
    # Merge
    df_acct = send_stats.merge(recv_stats, on=['acct', 'is_alert'], how='outer').fillna(0)
    
    print(f"總帳戶數: {len(df_acct):,}")
    print(f"警示帳戶: {df_acct['is_alert'].sum():,}")
    print(f"非警示帳戶: {len(df_acct) - df_acct['is_alert'].sum():,}")
    
    return df_acct


def generate_alert_comparison_chart(df_acct):
    """
    Generate alert vs normal account comparison visualization.
    
    Creates a 4-panel box plot comparing key features between alert
    and normal accounts:
    1. Average send amount
    2. Total send amount
    3. Average receive amount
    4. Number of transactions
    
    Args:
        df_acct (pd.DataFrame): Account-level features with is_alert column
    
    Output:
        Saves chart to: visualizations/04_alert_vs_normal_comparison.png
    
    Chart Features:
        - Box plots showing distribution differences
        - Outliers displayed as individual points
        - Median values annotated on charts
        - Alert accounts in coral, normal in light blue
    
    Expected Findings:
        Alert accounts typically have:
        - Higher transaction amounts
        - More frequent transactions
        - Wider variance in behavior
    
    Notes:
        Zero values removed for better visualization.
        Skipped if no alert accounts found in transaction data.
    """
    print("\n生成圖4: 警示 vs 非警示金額對比...")
    
    if df_acct['is_alert'].sum() == 0:
        print("警告: 交易資料中沒有警示帳戶，跳過對比圖")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    features_to_compare = [
        ('avg_send', 'Average Send Amount'),
        ('total_send', 'Total Send Amount'),
        ('avg_recv', 'Average Receive Amount'),
        ('send_count', 'Number of Transactions')
    ]
    
    for idx, (feature, title) in enumerate(features_to_compare):
        ax = axes[idx // 2, idx % 2]
        
        alert_data = df_acct[df_acct['is_alert'] == 1][feature]
        normal_data = df_acct[df_acct['is_alert'] == 0][feature]
        
        # Remove zeros for better visualization
        alert_data = alert_data[alert_data > 0]
        normal_data = normal_data[normal_data > 0]
        
        if len(alert_data) > 0 and len(normal_data) > 0:
            data = [normal_data, alert_data]
            bp = ax.boxplot(data, labels=['Normal', 'Alert'], patch_artist=True,
                           showfliers=True, widths=0.6)
            
            # Set colors
            colors = ['lightblue', 'lightcoral']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            
            ax.set_ylabel('Value', fontsize=12)
            ax.set_title(f'{title}\n(Alert vs Normal)', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add statistics
            alert_median = alert_data.median()
            normal_median = normal_data.median()
            ax.text(0.02, 0.98, 
                   f'Normal Median: {normal_median:,.0f}\nAlert Median: {alert_median:,.0f}',
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=9)
    
    plt.tight_layout()
    plt.savefig('visualizations/04_alert_vs_normal_comparison.png', dpi=300, bbox_inches='tight')
    print("已儲存: visualizations/04_alert_vs_normal_comparison.png")
    plt.close()


def perform_statistical_tests(df_acct):
    """
    Perform Mann-Whitney U tests comparing alert vs normal accounts.
    
    Conducts non-parametric statistical tests to determine if alert and
    normal accounts have significantly different distributions for key
    behavioral features.
    
    Args:
        df_acct (pd.DataFrame): Account-level features with is_alert column
    
    Output:
        Prints test results to console with:
        - Median and mean values for each group
        - P-values with significance levels:
            *** p < 0.001 (highly significant)
            **  p < 0.01  (very significant)
            *   p < 0.05  (significant)
            n.s. p >= 0.05 (not significant)
    
    Features Tested:
        - avg_send: Average send amount
        - total_send: Total send amount
        - send_count: Transaction count
        - avg_recv: Average receive amount
        - total_recv: Total receive amount
    
    Statistical Method:
        Mann-Whitney U test (non-parametric) chosen because:
        - Transaction amounts are not normally distributed
        - Robust to outliers
        - Suitable for comparing two independent samples
    
    Expected Results:
        Alert accounts should show significantly different patterns
        (p < 0.001) for most features, validating their utility for
        fraud detection.
    
    Notes:
        Only tests accounts with positive values (removes zeros).
        Requires scipy.stats module.
    """
    print("\n" + "="*80)
    print("統計檢驗 (Mann-Whitney U Test)")
    print("="*80)
    
    if df_acct['is_alert'].sum() == 0:
        print("警告: 無警示帳戶，跳過統計檢驗")
        return
    
    from scipy import stats
    
    features_to_test = ['avg_send', 'total_send', 'send_count', 'avg_recv', 'total_recv']
    
    for feature in features_to_test:
        alert_data = df_acct[df_acct['is_alert'] == 1][feature]
        normal_data = df_acct[df_acct['is_alert'] == 0][feature]
        
        # Remove zeros
        alert_data = alert_data[alert_data > 0]
        normal_data = normal_data[normal_data > 0]
        
        if len(alert_data) > 0 and len(normal_data) > 0:
            statistic, p_value = stats.mannwhitneyu(alert_data, normal_data, 
                                                    alternative='two-sided')
            
            significance = ("***" if p_value < 0.001 else 
                          "**" if p_value < 0.01 else 
                          "*" if p_value < 0.05 else "n.s.")
            
            print(f"\n{feature}:")
            print(f"  Alert: median={alert_data.median():.2f}, mean={alert_data.mean():.2f}")
            print(f"  Normal: median={normal_data.median():.2f}, mean={normal_data.mean():.2f}")
            print(f"  p-value: {p_value:.4f} {significance}")


def generate_time_series_chart(df_txn):
    """
    Generate time series trend visualization.
    
    Creates a 3-panel chart showing daily trends of:
    1. Transaction count
    2. Total transaction amount
    3. Average transaction amount
    
    Args:
        df_txn (pd.DataFrame): Transaction data with txn_date and txn_amt
    
    Output:
        Saves chart to: visualizations/05_time_series_trends.png
    
    Aggregation:
        Groups by txn_date and calculates:
        - count: Number of transactions per day
        - sum: Total amount per day
        - mean: Average amount per day
    
    Chart Features:
        - Line plots with filled areas
        - Grid for easier reading
        - Different colors for each metric
        - Markers on data points
    
    Use Cases:
        - Identify weekly/monthly patterns
        - Detect anomalies or sudden changes
        - Validate data quality
    """
    print("\n生成圖5: 時間序列趨勢...")
    
    daily_stats = df_txn.groupby('txn_date').agg({
        'txn_amt': ['sum', 'mean', 'count']
    }).reset_index()
    daily_stats.columns = ['date', 'total_amount', 'avg_amount', 'count']
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    # Daily transaction count
    axes[0].plot(daily_stats['date'], daily_stats['count'], color='steelblue', 
                 linewidth=2, marker='o')
    axes[0].fill_between(daily_stats['date'], daily_stats['count'], alpha=0.3)
    axes[0].set_xlabel('Date (Day)', fontsize=12)
    axes[0].set_ylabel('Transaction Count', fontsize=12)
    axes[0].set_title('Daily Transaction Volume', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Daily total amount
    axes[1].plot(daily_stats['date'], daily_stats['total_amount'], color='green', 
                 linewidth=2, marker='s')
    axes[1].fill_between(daily_stats['date'], daily_stats['total_amount'], 
                        alpha=0.3, color='green')
    axes[1].set_xlabel('Date (Day)', fontsize=12)
    axes[1].set_ylabel('Total Amount', fontsize=12)
    axes[1].set_title('Daily Total Transaction Amount', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # Daily average amount
    axes[2].plot(daily_stats['date'], daily_stats['avg_amount'], color='coral', 
                 linewidth=2, marker='^')
    axes[2].fill_between(daily_stats['date'], daily_stats['avg_amount'], 
                        alpha=0.3, color='coral')
    axes[2].set_xlabel('Date (Day)', fontsize=12)
    axes[2].set_ylabel('Average Amount', fontsize=12)
    axes[2].set_title('Daily Average Transaction Amount', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('visualizations/05_time_series_trends.png', dpi=300, bbox_inches='tight')
    print("已儲存: visualizations/05_time_series_trends.png")
    plt.close()


def generate_hourly_distribution_chart(df_txn):
    """
    Generate hourly transaction pattern visualization.
    
    Shows distribution of transactions across 24 hours of the day,
    helping identify peak hours and unusual patterns.
    
    Args:
        df_txn (pd.DataFrame): Transaction data with txn_time column
    
    Output:
        Saves chart to: visualizations/06_hourly_distribution.png
    
    Time Parsing:
        Extracts hour from txn_time (assumes format with ':' separator)
        Falls back to 0 if parsing fails
    
    Chart Features:
        - Bar chart for 24 hours (0-23)
        - Red dashed line marks peak hour
        - Grid for easier reading
        - Legend showing peak hour
    
    Fraud Indicators:
        - High late-night activity (22:00-06:00) may indicate fraud
        - Normal accounts typically transact during business hours
        - Alert accounts often show abnormal time patterns
    
    Notes:
        Peak hour identification helps understand customer behavior
        and detect anomalous patterns.
    """
    print("\n生成圖6: 交易時間分佈...")
    
    # Parse hour from time
    df_txn['hour'] = df_txn['txn_time'].apply(
        lambda x: int(str(x).split(':')[0]) if ':' in str(x) else 0
    )
    
    hour_stats = df_txn.groupby('hour').size().reset_index(name='count')
    
    fig, ax = plt.subplots(figsize=(15, 6))
    bars = ax.bar(hour_stats['hour'], hour_stats['count'], color='steelblue', 
                  edgecolor='black', alpha=0.7)
    
    # Mark peak hour
    if len(hour_stats) > 0:
        peak_hour = hour_stats.loc[hour_stats['count'].idxmax(), 'hour']
        ax.axvline(x=peak_hour, color='red', linestyle='--', linewidth=2, 
                  label=f'Peak Hour: {peak_hour}:00')
    
    ax.set_xlabel('Hour of Day', fontsize=12)
    ax.set_ylabel('Transaction Count', fontsize=12)
    ax.set_title('Transaction Distribution by Hour', fontsize=14, fontweight='bold')
    ax.set_xticks(range(24))
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('visualizations/06_hourly_distribution.png', dpi=300, bbox_inches='tight')
    print("已儲存: visualizations/06_hourly_distribution.png")
    plt.close()


def detect_outliers_and_visualize(df_txn):
    """
    Detect and visualize transaction amount outliers.
    
    Uses IQR (Interquartile Range) method to identify outliers and
    generates visualization showing their distribution.
    
    Args:
        df_txn (pd.DataFrame): Transaction data with txn_amt and txn_date
    
    Returns:
        tuple: (outliers, lower_bound, upper_bound)
            - outliers (pd.DataFrame): Outlier transactions
            - lower_bound (float): Lower outlier threshold
            - upper_bound (float): Upper outlier threshold
    
    Output:
        Saves chart to: visualizations/07_outlier_detection.png
    
    Outlier Detection Method:
        IQR = Q3 - Q1
        Lower bound = Q1 - 1.5 * IQR
        Upper bound = Q3 + 1.5 * IQR
        Outliers are values outside these bounds
    
    Chart Features:
        Panel 1: Box plot on log scale showing outliers as red dots
        Panel 2: Scatter plot over time highlighting outliers
    
    Statistics Printed:
        - IQR range
        - Number and percentage of outliers
        - Outlier amount statistics
        - Number involving alert accounts
    
    Notes:
        IQR method is robust to extreme values and widely used
        for outlier detection in financial data.
    """
    print("\n" + "="*80)
    print("Part 4: 異常值檢測")
    print("="*80)
    print("\n金額異常值檢測 (IQR方法)...")
    
    Q1 = df_txn['txn_amt'].quantile(0.25)
    Q3 = df_txn['txn_amt'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df_txn[(df_txn['txn_amt'] < lower_bound) | 
                      (df_txn['txn_amt'] > upper_bound)]
    
    print(f"IQR 範圍: [{lower_bound:,.0f}, {upper_bound:,.0f}]")
    print(f"異常值數量: {len(outliers):,} ({len(outliers)/len(df_txn)*100:.2f}%)")
    
    if len(outliers) > 0:
        print(f"\n異常交易統計:")
        print(f"  最大金額: {outliers['txn_amt'].max():,.0f}")
        print(f"  最小金額: {outliers['txn_amt'].min():,.0f}")
        print(f"  平均金額: {outliers['txn_amt'].mean():,.0f}")
        print(f"  涉及警示帳戶: {outliers['any_alert'].sum():,} 筆")
    
    # Generate visualization
    print("\n生成圖7: 異常值視覺化...")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Box plot
    axes[0].boxplot(np.log10(df_txn['txn_amt'] + 1), vert=True, patch_artist=True,
                    boxprops=dict(facecolor='lightblue', color='black'),
                    medianprops=dict(color='red', linewidth=2),
                    flierprops=dict(marker='o', markerfacecolor='red', 
                                   markersize=5, alpha=0.5))
    axes[0].set_ylabel('Log10(Transaction Amount)', fontsize=12)
    axes[0].set_title('Log10(Transaction Amount) Box Plot\n(Red dots = Outliers)', 
                     fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Scatter plot over time
    normal_txn = df_txn[~((df_txn['txn_amt'] < lower_bound) | 
                          (df_txn['txn_amt'] > upper_bound))]
    axes[1].scatter(normal_txn['txn_date'], normal_txn['txn_amt'], 
                    c='steelblue', alpha=0.5, s=30, label='Normal')
    if len(outliers) > 0:
        axes[1].scatter(outliers['txn_date'], outliers['txn_amt'], 
                        c='red', alpha=0.7, s=50, marker='^', label='Outlier')
    axes[1].set_xlabel('Transaction Date', fontsize=12)
    axes[1].set_ylabel('Transaction Amount', fontsize=12)
    axes[1].set_title('Transaction Amount over Time\n(Outliers Highlighted)', 
                     fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('visualizations/07_outlier_detection.png', dpi=300, bbox_inches='tight')
    print("已儲存: visualizations/07_outlier_detection.png")
    plt.close()
    
    return outliers, lower_bound, upper_bound


def detect_high_frequency_accounts(df_txn, alert_accts):
    """
    Detect accounts with abnormally high transaction frequency.
    
    Identifies accounts with transaction counts exceeding mean + 2*std,
    which may indicate automated trading, money laundering, or fraud.
    
    Args:
        df_txn (pd.DataFrame): Transaction data
        alert_accts (set): Set of alert account IDs
    
    Output:
        Prints statistics about high-frequency accounts to console
    
    Detection Threshold:
        mean + 2*std (covers ~95% of normal distribution)
        Accounts above this threshold are flagged as high-frequency
    
    Statistics Printed:
        - Average transaction count
        - High-frequency threshold
        - Number of high-frequency accounts
        - Number of high-frequency accounts that are alerts
    
    Use Cases:
        - Identify bot accounts or automated systems
        - Detect potential money laundering patterns
        - Validate data quality (test accounts, etc.)
    
    Notes:
        High frequency alone doesn't indicate fraud, but combined
        with other features (amount patterns, timing) can be indicative.
    """
    print("\n異常頻繁交易檢測...")
    
    acct_freq = df_txn.groupby('from_acct').size().reset_index(name='txn_count')
    acct_freq['is_alert'] = acct_freq['from_acct'].isin(alert_accts).astype(int)
    
    mean_freq = acct_freq['txn_count'].mean()
    std_freq = acct_freq['txn_count'].std()
    high_freq_threshold = mean_freq + 2 * std_freq
    
    high_freq_accts = acct_freq[acct_freq['txn_count'] > high_freq_threshold]
    
    print(f"平均交易次數: {mean_freq:.2f}")
    print(f"高頻交易門檻: {high_freq_threshold:.2f}")
    print(f"高頻交易帳戶: {len(high_freq_accts):,}")
    
    if len(high_freq_accts) > 0:
        print(f"其中警示帳戶: {high_freq_accts['is_alert'].sum():,}")


def perform_network_analysis(df_txn, alert_accts):
    """
    Perform transaction network analysis and visualization.
    
    Constructs directed transaction network and calculates:
    - Degree centrality (in/out degree)
    - PageRank scores
    - Network visualization of top nodes
    - Community detection (optional, for smaller networks)
    
    Args:
        df_txn (pd.DataFrame): Transaction data with from_acct, to_acct, txn_amt
        alert_accts (set): Set of alert account IDs
    
    Outputs:
        - visualizations/08_network_visualization.png: Network diagram
        - features/pagerank_scores.csv: PageRank scores for all accounts
        - Console output with network statistics
    
    Network Construction:
        - Nodes: Accounts
        - Edges: Transactions (from_acct -> to_acct)
        - Edge weights: Transaction amounts
    
    Analyses Performed:
        1. Basic statistics (node/edge counts)
        2. Degree centrality (top 5 senders/receivers)
        3. PageRank (top 10 important accounts)
        4. Network visualization (top 20 active nodes)
        5. Community detection (if network < 100k nodes)
    
    PageRank:
        - Measures account importance in network
        - alpha=0.85 (standard damping factor)
        - Weighted by transaction amounts
        - Saved to features/pagerank_scores.csv for use in modeling
    
    Visualization:
        - Red nodes: Alert accounts
        - Blue nodes: Normal accounts
        - Node size: Proportional to degree
        - Shows top 20 active accounts
        - Uses spring layout algorithm
    
    Notes:
        - Network analysis can be time-consuming for large graphs
        - Community detection skipped for networks > 100k nodes
        - Requires networkx package
    """
    print("\n" + "="*80)
    print("Part 5: 交易網絡分析")
    print("="*80)
    
    try:
        import networkx as nx
        
        print("\n建立交易網絡...")
        G = nx.DiGraph()
        
        # Build network
        for _, row in df_txn.iterrows():
            G.add_edge(row['from_acct'], row['to_acct'], weight=row['txn_amt'])
        
        print(f"節點數 (帳戶): {G.number_of_nodes():,}")
        print(f"邊數 (交易): {G.number_of_edges():,}")
        
        # Basic network statistics
        print("\n網絡基本統計:")
        
        in_degree = dict(G.in_degree())
        out_degree = dict(G.out_degree())
        
        top_in = sorted(in_degree.items(), key=lambda x: x[1], reverse=True)[:5]
        top_out = sorted(out_degree.items(), key=lambda x: x[1], reverse=True)[:5]
        
        print("\nTop 5 收款帳戶 (最多入度):")
        for acct, degree in top_in:
            is_alert = "警示" if acct in alert_accts else ""
            print(f"  {acct[:16]}... : {degree} 筆 {is_alert}")
        
        print("\nTop 5 匯款帳戶 (最多出度):")
        for acct, degree in top_out:
            is_alert = "警示" if acct in alert_accts else ""
            print(f"  {acct[:16]}... : {degree} 筆 {is_alert}")
        
        # PageRank analysis
        print("\nPageRank 分析...")
        try:
            pagerank = nx.pagerank(G, weight='weight')
            top_pagerank = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:10]
            
            print("\nTop 10 重要帳戶 (PageRank):")
            for idx, (acct, score) in enumerate(top_pagerank, 1):
                is_alert = "警示" if acct in alert_accts else ""
                print(f"  {idx}. {acct[:16]}... : {score:.6f} {is_alert}")
            
            # Save PageRank scores
            print("\n儲存 PageRank 結果...")
            pagerank_data = []
            for acct in G.nodes():
                pagerank_data.append({
                    'acct': acct,
                    'pagerank': pagerank.get(acct, 0),
                    'in_degree': G.in_degree(acct),
                    'out_degree': G.out_degree(acct)
                })
            
            pagerank_df = pd.DataFrame(pagerank_data)
            import os
            os.makedirs('features', exist_ok=True)
            pagerank_df.to_csv('features/pagerank_scores.csv', index=False)
            print(f"PageRank 已儲存: features/pagerank_scores.csv ({len(pagerank_df):,} 個帳戶)")
        
        except:
            print("PageRank 計算失敗 (可能網絡太小)")
        
        # Network visualization
        print("\n生成圖8: 網絡視覺化...")
        
        # Visualize smaller subgraph
        if G.number_of_nodes() > 20:
            top_nodes = ([acct for acct, _ in top_out[:10]] + 
                        [acct for acct, _ in top_in[:10]])
            G_sub = G.subgraph(top_nodes)
        else:
            G_sub = G
        
        fig, ax = plt.subplots(figsize=(15, 12))
        
        # Node colors and sizes
        node_colors = ['red' if node in alert_accts else 'lightblue' 
                      for node in G_sub.nodes()]
        node_sizes = [G_sub.degree(node) * 100 + 100 for node in G_sub.nodes()]
        
        # Layout
        pos = nx.spring_layout(G_sub, k=2, iterations=50, seed=42)
        
        # Draw network
        nx.draw_networkx_nodes(G_sub, pos, node_color=node_colors, 
                              node_size=node_sizes, alpha=0.7, ax=ax)
        nx.draw_networkx_edges(G_sub, pos, alpha=0.3, arrows=True, 
                              arrowsize=10, width=1, ax=ax)
        
        # Labels
        labels = {node: node[:8]+"..." for node in G_sub.nodes()}
        nx.draw_networkx_labels(G_sub, pos, labels, font_size=8, ax=ax)
        
        ax.set_title('Transaction Network\n(Red = Alert Accounts, Size = Activity Level)', 
                    fontsize=14, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('visualizations/08_network_visualization.png', dpi=300, 
                   bbox_inches='tight')
        print("已儲存: visualizations/08_network_visualization.png")
        plt.close()
        
        # Community detection
        print("\n社群檢測...")
        G_undirected = G.to_undirected()
        
        if G_undirected.number_of_nodes() > 100000:
            print(f"網絡規模過大 ({G_undirected.number_of_nodes():,} 節點)")
            print("跳過社群檢測")
        else:
            try:
                from networkx.algorithms import community
                communities = community.greedy_modularity_communities(G_undirected)
                
                print(f"檢測到 {len(communities)} 個社群")
                
                sorted_communities = sorted(communities, key=len, reverse=True)
                for idx, comm in enumerate(sorted_communities[:5], 1):
                    alert_count = sum(1 for node in comm if node in alert_accts)
                    print(f"  社群 {idx}: {len(comm)} 個帳戶 (警示: {alert_count})")
            except:
                print("社群檢測失敗")
    
    except ImportError:
        print("NetworkX 未安裝，跳過網絡分析")
        print("安裝指令: pip install networkx")


def generate_analysis_report(df_txn, df_alert, df_test, outliers, lower_bound, upper_bound):
    """
    Generate comprehensive EDA report in Markdown format.
    
    Creates a summary report containing key statistics and findings
    from the exploratory data analysis.
    
    Args:
        df_txn (pd.DataFrame): Transaction data
        df_alert (pd.DataFrame): Alert account data
        df_test (pd.DataFrame): Test data
        outliers (pd.DataFrame): Detected outlier transactions
        lower_bound (float): Lower outlier threshold
        upper_bound (float): Upper outlier threshold
    
    Output:
        Saves report to: visualizations/EDA_REPORT.md
    
    Report Sections:
        1. Data Overview: Counts and percentages
        2. Transaction Amount Statistics: Mean, median, std, etc.
        3. Outlier Detection: IQR range and outlier counts
        4. Visualizations: List of generated charts
        5. Key Findings: (placeholder for manual insights)
        6. Feature Engineering Directions: Suggested features
    
    Format:
        Markdown with headers, bullet points, and formatted numbers
        
    Notes:
        This report serves as documentation of the EDA process
        and guides subsequent feature engineering steps.
    """
    print("\n" + "="*80)
    print("生成分析報告")
    print("="*80)
    
    report = f"""
# 深度探索性資料分析報告
生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 1. 資料概覽
- 交易資料: {len(df_txn):,} 筆
- 警示帳戶: {len(df_alert):,} 個
- 待預測帳戶: {len(df_test):,} 個
- 涉及警示帳戶的交易: {df_txn['any_alert'].sum():,} 筆 ({df_txn['any_alert'].sum()/len(df_txn)*100:.2f}%)

## 2. 交易金額統計
- 平均交易金額: {df_txn['txn_amt'].mean():,.2f}
- 中位數交易金額: {df_txn['txn_amt'].median():,.2f}
- 最大交易金額: {df_txn['txn_amt'].max():,.2f}
- 最小交易金額: {df_txn['txn_amt'].min():,.2f}
- 標準差: {df_txn['txn_amt'].std():,.2f}

## 3. 異常值檢測
- IQR 範圍: [{lower_bound:,.0f}, {upper_bound:,.0f}]
- 異常值數量: {len(outliers):,} ({len(outliers)/len(df_txn)*100:.2f}%)

## 4. 視覺化圖表
已生成 8 張圖表於 visualizations/ 資料夾:
1. 交易金額分佈
2. 類別變數分佈
3. 時間分佈
4. 警示 vs 非警示對比
5. 時間序列趨勢
6. 交易時間分佈
7. 異常值視覺化
8. 交易網絡視覺化

## 5. 關鍵發現
(待補充: 根據實際資料的發現)

## 6. 建議的特徵工程方向
- 金額特徵: 總額、平均、最大、最小、標準差、變異係數
- 頻率特徵: 交易次數、日均交易次數
- 時間特徵: 交易日期範圍、夜間交易比例、週末交易比例
- 網絡特徵: PageRank、度中心性、社群歸屬
- 異常特徵: 是否為異常金額、是否為高頻交易
"""
    
    with open('visualizations/EDA_REPORT.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("分析報告已儲存: visualizations/EDA_REPORT.md")


def main():
    """
    Main execution function for EDA pipeline.
    
    Orchestrates the complete exploratory data analysis process:
    1. Load and preprocess data
    2. Generate 8 visualization charts
    3. Perform statistical tests
    4. Conduct network analysis
    5. Generate comprehensive report
    
    Execution Steps:
        - Data loading and labeling
        - Amount distribution analysis
        - Categorical variable analysis
        - Temporal pattern analysis
        - Alert vs normal comparison
        - Time series trends
        - Hourly patterns
        - Outlier detection
        - High-frequency account detection
        - Network analysis (optional)
        - Report generation
    
    Outputs:
        visualizations/*.png: 8 analysis charts
        visualizations/EDA_REPORT.md: Summary report
        features/pagerank_scores.csv: PageRank features (if networkx available)
    
    Execution Time:
        Approximately 2-3 minutes for standard dataset size
        May take longer for network analysis on large graphs
    
    Notes:
        This is an optional step. The generated insights inform
        feature engineering but EDA itself is not required for
        model training.
    """
    print("="*80)
    print("Step 1: 深度探索性資料分析")
    print("="*80)
    
    # 1. Load data
    df_txn, df_alert, df_test = load_transaction_data()
    
    # 2. Preprocess and label
    df_txn, alert_accts = preprocess_and_label_data(df_txn, df_alert)
    
    # 3. Create visualization directory
    import os
    os.makedirs('visualizations', exist_ok=True)
    
    # 4. Generate visualizations
    print("\n" + "="*80)
    print("Part 1: 視覺化分析")
    print("="*80)
    
    generate_amount_distribution_chart(df_txn)
    generate_categorical_distribution_chart(df_txn)
    generate_time_distribution_chart(df_txn, df_alert)
    
    # 5. Alert vs normal comparison
    print("\n" + "="*80)
    print("Part 2: 警示 vs 非警示帳戶對比分析")
    print("="*80)
    
    df_acct = prepare_account_level_features(df_txn, alert_accts)
    generate_alert_comparison_chart(df_acct)
    perform_statistical_tests(df_acct)
    
    # 6. Time series analysis
    print("\n" + "="*80)
    print("Part 3: 時間序列分析")
    print("="*80)
    
    generate_time_series_chart(df_txn)
    generate_hourly_distribution_chart(df_txn)
    
    # 7. Outlier detection
    outliers, lower_bound, upper_bound = detect_outliers_and_visualize(df_txn)
    detect_high_frequency_accounts(df_txn, alert_accts)
    
    # 8. Network analysis
    perform_network_analysis(df_txn, alert_accts)
    
    # 9. Generate report
    generate_analysis_report(df_txn, df_alert, df_test, outliers, 
                            lower_bound, upper_bound)
    
    # 10. Summary
    print("\n" + "="*80)
    print("深度 EDA 完成!")
    print("="*80)
    
    print(f"\n已生成:")
    print(f"  ✓ 8 張視覺化圖表 (visualizations/*.png)")
    print(f"  ✓ 1 份分析報告 (visualizations/EDA_REPORT.md)")
    
    print(f"\n關鍵發現:")
    print(f"  1. 交易資料中涉及警示帳戶的比例: {df_txn['any_alert'].sum()/len(df_txn)*100:.2f}%")
    print(f"  2. 異常交易比例: {len(outliers)/len(df_txn)*100:.2f}%")
    print(f"  3. 已識別網絡結構特徵")
    
    print(f"\n下一步:")
    print(f"  1. 查看 visualizations/ 資料夾中的圖表")
    print(f"  2. 根據 EDA 發現設計特徵")
    print(f"  3. 執行 Preprocess/feature_engineering.py")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()