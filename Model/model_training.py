"""
Model Training Module for E.SUN AI Challenge 2025.

This module implements the complete model training pipeline including:
    1. Hard Negative Mining for handling class imbalance
    2. Optimized XGBoost model training
    3. Quantile-based threshold optimization
    4. Multiple threshold submission generation
    5. Feature importance analysis

The training strategy uses Hard Negative Mining to improve model's ability
to distinguish difficult negative samples, combined with carefully tuned
XGBoost parameters and dynamic threshold selection.

Usage:
    $ python Model/model_training.py

Input:
    features/features.csv: Complete feature set from feature engineering
    data/acct_alert.csv: Known alert accounts for training
    data/acct_predict.csv: Accounts to predict

Output:
    submissions/submission_YYYYMMDD_HHMMSS.csv: Main prediction file (Q93)
    submissions/submission_q{91,92,94,95}_YYYYMMDD_HHMMSS.csv: Alternative thresholds
    submissions/feature_importance.csv: Feature importance rankings

Expected Performance:
    Local Validation: F1-Score ~0.22-0.25
    Public Leaderboard: F1-Score ~0.28-0.32
    Private Leaderboard: F1-Score ~0.30-0.32

Author: E.SUN AI Challenge Team
Date: 2025
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
import os
import warnings
warnings.filterwarnings('ignore')


def select_confident_negatives(neg_data):
    """
    Select high-confidence negative samples for Hard Negative Mining.
    
    Filters negative samples based on transaction behavior patterns to identify
    accounts that are clearly non-fraudulent. These serve as the pool for
    selecting hard and easy negatives.
    
    Filtering criteria:
        - Transaction count: 5 <= count <= 1000 (active but not excessive)
        - Average amount: 0 < amount < 100,000 (reasonable range)
        - Active days: >= 10 (sustained activity)
    
    Args:
        neg_data (pd.DataFrame): All negative samples with features:
            - out_cnt, in_cnt: Transaction counts
            - out_amt_mean: Average transaction amount
            - active_days: Number of active days
    
    Returns:
        pd.DataFrame: Filtered high-confidence negative samples
    
    Notes:
        Confidence is based on behavioral consistency patterns identified
        through EDA showing normal accounts have stable transaction patterns.
    """
    neg_data['total_txn'] = neg_data.get('out_cnt', 0) + neg_data.get('in_cnt', 0)
    
    confident_mask = (
        (neg_data['total_txn'] >= 5) &
        (neg_data['total_txn'] <= 1000) &
        (neg_data.get('out_amt_mean', 1) > 0) &
        (neg_data.get('out_amt_mean', 1) < 100000) &
        (neg_data.get('active_days', 0) >= 10)
    )
    
    return neg_data[confident_mask].copy()


def perform_hard_negative_mining(pos_data, neg_confident_data, feature_cols):
    """
    Perform Hard Negative Mining to identify difficult negative samples.
    
    Uses a preliminary XGBoost model to score negative samples and classify
    them into:
    - Hard Negatives: Samples with moderate prediction probability (0.10-0.40)
                     that are easily confused with positives
    - Easy Negatives: Samples with very low probability (<0.05) that are
                     clearly negative
    
    This technique improves model's discrimination ability on borderline cases.
    
    Args:
        pos_data (pd.DataFrame): All positive (alert) samples
        neg_confident_data (pd.DataFrame): High-confidence negative samples
        feature_cols (list): List of feature column names to use
    
    Returns:
        tuple: (neg_hard, neg_easy)
            - neg_hard (pd.DataFrame): Hard negative samples with pred_prob
            - neg_easy (pd.DataFrame): Easy negative samples with pred_prob
    
    Model Parameters:
        - n_estimators: 300 (moderate complexity)
        - learning_rate: 0.05
        - max_depth: 6 (prevents overfitting)
        - Random sampling for balanced training
    
    Notes:
        Probability thresholds (0.10-0.40 for hard, <0.05 for easy) are
        empirically determined to maximize training effectiveness.
    """
    print("\n尋找Hard Negatives...")
    
    # Sample negatives for preliminary model
    neg_sample = neg_confident_data.sample(
        n=min(len(neg_confident_data), len(pos_data)*10), 
        random_state=42
    )
    
    temp_train = pd.concat([pos_data, neg_sample]).sample(frac=1.0, random_state=42)
    
    X_temp = temp_train[feature_cols]
    y_temp = temp_train['label']
    
    # Train preliminary model
    temp_model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        eval_metric='logloss'
    )
    temp_model.fit(X_temp, y_temp)
    
    # Predict probabilities for all confident negatives
    X_neg = neg_confident_data[feature_cols]
    neg_confident_data['pred_prob'] = temp_model.predict_proba(X_neg)[:, 1]
    
    # Classify into hard and easy negatives
    neg_hard = neg_confident_data[
        (neg_confident_data['pred_prob'] >= 0.10) & 
        (neg_confident_data['pred_prob'] <= 0.40)
    ].copy()
    
    neg_easy = neg_confident_data[
        neg_confident_data['pred_prob'] < 0.05
    ].copy()
    
    return neg_hard, neg_easy


def build_augmented_training_set(pos_data, neg_hard, neg_easy):
    """
    Build augmented training set with balanced hard and easy negatives.
    
    Constructs final training set using:
    - All positive samples (100%)
    - Hard negatives (70% of positive count) - Focus on difficult cases
    - Easy negatives (30% of positive count) - Maintain clear boundaries
    
    This ratio is optimized to improve model's discrimination ability while
    maintaining robustness.
    
    Args:
        pos_data (pd.DataFrame): All positive samples
        neg_hard (pd.DataFrame): Hard negative samples
        neg_easy (pd.DataFrame): Easy negative samples
    
    Returns:
        pd.DataFrame: Augmented training set with shuffled samples
    
    Sampling Strategy:
        - Uses replacement if insufficient samples available
        - Random shuffling to prevent order bias
        - Fixed random seed (42) for reproducibility
    
    Notes:
        The 70:30 hard:easy ratio is based on experiments showing this
        balance maximizes F1-score on validation sets.
    """
    n_pos = len(pos_data)
    target_hard = int(n_pos * 7)  # 70% hard negatives
    target_easy = int(n_pos * 3)  # 30% easy negatives
    
    # Sample with replacement if needed
    neg_hard_sample = neg_hard.sample(
        n=target_hard, 
        replace=(len(neg_hard) < target_hard), 
        random_state=42
    )
    neg_easy_sample = neg_easy.sample(
        n=target_easy, 
        replace=(len(neg_easy) < target_easy), 
        random_state=42
    )
    
    # Combine and shuffle
    train_final = pd.concat([
        pos_data, 
        neg_hard_sample, 
        neg_easy_sample
    ]).sample(frac=1.0, random_state=42)
    
    return train_final


def train_xgboost_model(X_train, y_train):
    """
    Train optimized XGBoost model for fraud detection.
    
    Trains XGBoost classifier with carefully tuned hyperparameters optimized
    for this imbalanced fraud detection task. Parameters are selected to
    balance model capacity, generalization, and training efficiency.
    
    Args:
        X_train (pd.DataFrame): Training features (~53 features)
        y_train (pd.Series): Training labels (0=normal, 1=alert)
    
    Returns:
        XGBClassifier: Trained XGBoost model
    
    Hyperparameters:
        n_estimators: 800 - High tree count for sufficient capacity
        learning_rate: 0.03 - Low rate to prevent overfitting
        max_depth: 9 - Deep trees to capture complex interactions
        min_child_weight: 3 - Prevents overfitting on small samples
        subsample: 0.8 - Row sampling for robustness
        colsample_bytree: 0.8 - Column sampling per tree
        colsample_bylevel: 0.8 - Column sampling per level
        gamma: 0.1 - Minimum loss reduction for split
        reg_alpha: 0.1 - L1 regularization (Lasso)
        reg_lambda: 1.0 - L2 regularization (Ridge)
        random_state: 42 - For reproducibility
    
    Training Time:
        Approximately 3-5 minutes on 4-core CPU
    
    Notes:
        Regularization parameters (gamma, alpha, lambda) are crucial for
        preventing overfitting given the high feature dimensionality.
    """
    XGBOOST_PARAMS = {
        'n_estimators': 800,
        'learning_rate': 0.03,
        'max_depth': 9,
        'min_child_weight': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'colsample_bylevel': 0.8,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0,
        'eval_metric': 'logloss'
    }
    
    print("\n參數設定:")
    for key, value in XGBOOST_PARAMS.items():
        print(f"  {key}: {value}")
    
    print("\n訓練中...")
    model = XGBClassifier(**XGBOOST_PARAMS)
    model.fit(X_train, y_train)
    print("訓練完成")
    
    return model


def predict_with_quantile_threshold(model, X_test, test_feat, quantile=0.93):
    """
    Make predictions using quantile-based threshold optimization.
    
    Instead of using fixed threshold (0.5), this method dynamically selects
    threshold based on test set probability distribution. This adapts better
    to the actual distribution and improves F1-score.
    
    Args:
        model: Trained XGBoost model
        X_test (pd.DataFrame): Test features
        test_feat (pd.DataFrame): Test feature data with account IDs
        quantile (float): Quantile value for threshold selection.
                         Default 0.93 means ~7% predicted as positive.
    
    Returns:
        tuple: (test_prob, test_pred, threshold)
            - test_prob (np.array): Predicted probabilities
            - test_pred (np.array): Binary predictions (0/1)
            - threshold (float): Computed threshold value
    
    Threshold Selection:
        - Quantile 0.93 -> ~7% positive (matches training imbalance)
        - Higher quantile -> Lower recall, higher precision
        - Lower quantile -> Higher recall, lower precision
    
    Notes:
        Multiple quantile values (0.91-0.97) are tested and saved
        separately for A/B testing on leaderboard.
    """
    test_prob = model.predict_proba(X_test)[:, 1]
    
    threshold = float(np.quantile(test_prob, quantile))
    test_pred = (test_prob >= threshold).astype(int)
    
    return test_prob, test_pred, threshold


def save_predictions(df_test, test_feat, test_pred, timestamp, suffix=''):
    """
    Save predictions in competition submission format.
    
    Args:
        df_test (pd.DataFrame): Original test data with account IDs
        test_feat (pd.DataFrame): Test features with account IDs
        test_pred (np.array): Binary predictions
        timestamp (str): Timestamp string for filename
        suffix (str): Optional suffix for filename (e.g., '_q91')
    
    Returns:
        str: Path to saved file
    
    File Format:
        CSV with columns: acct, label
        - acct: Account ID (string)
        - label: Prediction (0 or 1)
        - Encoding: utf-8-sig (for compatibility)
    """
    pred_map = pd.Series(test_pred, index=test_feat['acct'])
    df_test['label'] = df_test['acct'].map(pred_map).fillna(0).astype(int)
    
    os.makedirs('submissions', exist_ok=True)
    output_file = f'submissions/submission{suffix}_{timestamp}.csv'
    df_test[['acct', 'label']].to_csv(output_file, index=False, encoding='utf-8-sig')
    
    return output_file


def analyze_feature_importance(model, feature_cols, timestamp):
    """
    Analyze and save feature importance rankings.
    
    Calculates feature importance based on XGBoost's gain metric, which
    measures the improvement in accuracy brought by each feature.
    
    Args:
        model: Trained XGBoost model with feature_importances_ attribute
        feature_cols (list): List of feature names
        timestamp (str): Timestamp for filename
    
    Returns:
        pd.DataFrame: Feature importance dataframe sorted by importance
    
    Saved Files:
        submissions/feature_importance.csv: Complete ranking with columns:
            - feature: Feature name
            - importance: Importance score (higher = more important)
    
    Special Analysis:
        Also calculates total contribution of graph features as a group
        to assess the value of network-based features.
    
    Notes:
        Top features typically include:
        - pagerank (graph importance)
        - date_similarity (temporal regularity)
        - comm_alert_ratio (community alert density)
    """
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n" + "="*80)
    print("特徵重要性 Top 30")
    print("="*80)
    print(importance_df.head(30).to_string(index=False))
    
    # Analyze graph feature contribution
    graph_feature_names = [
        'in_degree', 'out_degree', 'total_degree', 'pagerank', 
        'clustering_coef', 'betweenness', 'weighted_in_degree', 
        'weighted_out_degree', 'k_core', 'community_id', 'comm_alert_ratio'
    ]
    
    graph_importance = importance_df[
        importance_df['feature'].isin(graph_feature_names)
    ]['importance'].sum()
    total_importance = importance_df['importance'].sum()
    
    print(f"\n圖特徵總貢獻: {graph_importance/total_importance*100:.1f}%")
    
    # Save to file
    importance_df.to_csv('submissions/feature_importance.csv', index=False)
    
    return importance_df


def main():
    """
    Main execution function for model training pipeline.
    
    Orchestrates the complete training process:
    1. Load features and labels
    2. Perform Hard Negative Mining
    3. Build augmented training set
    4. Train optimized XGBoost model
    5. Generate predictions with multiple thresholds
    6. Analyze feature importance
    
    Outputs:
        - Main submission file (Q93 threshold)
        - Alternative submissions (Q91, Q92, Q94, Q95)
        - Feature importance analysis
    
    Performance Expectations:
        Training Time: 3-5 minutes
        Memory Usage: 2-4 GB
        F1-Score: 0.22+ on local validation
    
    Raises:
        SystemExit: If feature file not found
    """
    print("="*80)
    print("Step 3: Model Training")
    print("="*80)
    
    # ========================================================================
    # Load Data
    # ========================================================================
    if os.path.exists('/mnt/project/acct_alert.csv'):
        DATA_PATH = '/mnt/project'
    elif os.path.exists('data/acct_alert.csv'):
        DATA_PATH = 'data'
    else:
        DATA_PATH = '.'
    
    df_alert = pd.read_csv(f'{DATA_PATH}/acct_alert.csv')
    df_test = pd.read_csv(f'{DATA_PATH}/acct_predict.csv')
    
    # Load features
    if os.path.exists('features/features.csv'):
        feat = pd.read_csv('features/features.csv')
        print(f"\n特徵來源: features.csv")
    else:
        print("錯誤: 找不到特徵檔案")
        print("請先執行: python Preprocess/feature_engineering.py")
        exit(1)
    
    feat = feat.fillna(0).replace([np.inf, -np.inf], 0)
    feat['label'] = feat['acct'].isin(df_alert['acct']).astype(int)
    
    print(f"特徵維度: {feat.shape}")
    print(f"總帳戶: {len(feat):,}")
    print(f"警示帳戶: {feat['label'].sum():,}")
    
    # ========================================================================
    # Part 1: Hard Negative Mining
    # ========================================================================
    print("\n" + "="*80)
    print("Part 1: 負樣本增強")
    print("="*80)
    
    pos = feat[feat['label'] == 1].copy()
    neg = feat[feat['label'] == 0].copy()
    
    # Select confident negatives
    neg_confident = select_confident_negatives(neg)
    print(f"高置信度負樣本: {len(neg_confident):,} ({len(neg_confident)/len(neg)*100:.1f}%)")
    
    # Perform hard negative mining
    temp_cols = [c for c in feat.columns if c not in ['acct', 'label', 'total_txn']]
    neg_hard, neg_easy = perform_hard_negative_mining(pos, neg_confident, temp_cols)
    
    print(f"Hard Negative: {len(neg_hard):,}")
    print(f"Easy Negative: {len(neg_easy):,}")
    
    # Build augmented training set
    train_final = build_augmented_training_set(pos, neg_hard, neg_easy)
    
    print(f"\n最終訓練集: {len(train_final):,}")
    print(f"正樣本比例: {(train_final['label']==1).mean():.4f}")
    
    # ========================================================================
    # Part 2: Train XGBoost
    # ========================================================================
    print("\n" + "="*80)
    print("Part 2: 訓練XGBoost")
    print("="*80)
    
    train_cols = [c for c in train_final.columns if c not in ['acct', 'label', 'total_txn', 'pred_prob']]
    X_train = train_final[train_cols]
    y_train = train_final['label']
    
    print(f"訓練特徵數: {len(train_cols)}")
    
    model = train_xgboost_model(X_train, y_train)
    
    # ========================================================================
    # Part 3: Test Set Prediction
    # ========================================================================
    print("\n" + "="*80)
    print("Part 3: 測試集預測")
    print("="*80)
    
    test_feat = feat[feat['acct'].isin(df_test['acct'])].copy()
    X_test = test_feat[train_cols]
    
    test_prob, _, _ = predict_with_quantile_threshold(model, X_test, test_feat)
    
    print(f"測試集大小: {len(test_prob)}")
    print(f"機率範圍: [{test_prob.min():.6f}, {test_prob.max():.6f}]")
    print(f"機率中位數: {np.median(test_prob):.6f}")
    
    # Test multiple thresholds
    print("\n閾值測試:")
    print("-" * 60)
    thresholds_test = [0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97]
    
    for quantile in thresholds_test:
        thr = float(np.quantile(test_prob, quantile))
        pred = (test_prob >= thr).astype(int)
        rate = pred.sum() / len(pred)
        print(f"Quantile {quantile:.2f} | 閾值:{thr:.6f} | 預測:{pred.sum():4d} | 比例:{rate*100:.2f}%")
    
    # Final prediction with Q93
    FINAL_QUANTILE = 0.93
    _, test_pred, FINAL_THR = predict_with_quantile_threshold(
        model, X_test, test_feat, FINAL_QUANTILE
    )
    
    print("\n" + "="*80)
    print("最終選擇")
    print("="*80)
    print(f"Quantile: {FINAL_QUANTILE}")
    print(f"閾值: {FINAL_THR:.6f}")
    print(f"預測陽性: {test_pred.sum()}/{len(test_pred)} ({test_pred.sum()/len(test_pred)*100:.2f}%)")
    
    # ========================================================================
    # Part 4: Save Results
    # ========================================================================
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    
    # Save main submission
    output_file = save_predictions(df_test, test_feat, test_pred, timestamp)
    print(f"\n已儲存: {output_file}")
    
    # Save alternative thresholds
    print("\n儲存多個閾值版本...")
    for quantile in [0.91, 0.92, 0.94, 0.95]:
        _, pred, _ = predict_with_quantile_threshold(model, X_test, test_feat, quantile)
        file_name = save_predictions(
            df_test.copy(), test_feat, pred, timestamp, 
            suffix=f'_q{int(quantile*100)}'
        )
        print(f"  Q{quantile:.2f} ({pred.sum():4d}): {file_name}")
    
    # ========================================================================
    # Part 5: Feature Importance Analysis
    # ========================================================================
    analyze_feature_importance(model, train_cols, timestamp)
    
    print("\n" + "="*80)
    print("訓練完成")
    print("="*80)


if __name__ == "__main__":
    main()