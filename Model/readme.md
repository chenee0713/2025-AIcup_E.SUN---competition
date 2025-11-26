# Model - 模型訓練模組

## 功能說明
本模組負責模型訓練、負樣本增強、預測與結果輸出。

## 主要程式

### model_training.py
完整模型訓練Pipeline，包含Hard Negative Mining和XGBoost訓練。

**輸入**:
- `features_enhanced/features_enhanced_v1.csv`: 完整特徵集
- `data/acct_alert.csv`: 警示帳戶清單
- `data/acct_predict.csv`: 待預測帳戶清單

**輸出**:
- `submissions/submission_YYYYMMDD_HHMMSS.csv`: 主要預測結果
- `submissions/submission_q{91,92,94,95}_YYYYMMDD_HHMMSS.csv`: 不同閾值版本
- `submissions/feature_importance.csv`: 特徵重要性排序

**執行**:
```bash
python Model/model_training.py
```

## 訓練策略詳細說明

### 1. Hard Negative Mining

**目的**: 提升模型對難分辨負樣本的識別能力

**步驟**:
1. **篩選高置信度負樣本**:
   - 條件:
     - 5 ≤ 交易次數 ≤ 1000
     - 金額 < 100,000
     - 活躍天數 ≥ 10
   
2. **訓練初步模型**:
```python
   XGBClassifier(
       n_estimators=300,
       learning_rate=0.05,
       max_depth=6,
       subsample=0.8,
       colsample_bytree=0.8
   )
```

3. **分類負樣本**:
   - **Hard Negative** (pred_prob: 0.10~0.40): 容易誤判的負樣本
   - **Easy Negative** (pred_prob < 0.05): 明顯的負樣本

4. **構建增強訓練集**:
   - 所有正樣本 (100%)
   - Hard Negative (70%)
   - Easy Negative (30%)

### 2. XGBoost模型

#### 超參數設定
```python
XGBOOST_PARAMS = {
    'n_estimators': 800,        # 樹的數量（增加以提升性能）
    'learning_rate': 0.03,      # 學習率（較小以防止過擬合）
    'max_depth': 9,             # 最大深度（增加模型複雜度）
    'min_child_weight': 3,      # 最小葉子節點樣本權重
    'subsample': 0.8,           # 樣本子採樣比例
    'colsample_bytree': 0.8,    # 特徵子採樣比例
    'colsample_bylevel': 0.8,   # 層級特徵子採樣
    'gamma': 0.1,               # 最小損失減少（剪枝參數）
    'reg_alpha': 0.1,           # L1正則化（Lasso）
    'reg_lambda': 1.0,          # L2正則化（Ridge）
    'random_state': 42,         # 隨機種子（確保可復現）
    'n_jobs': -1,               # 使用所有CPU核心
    'verbosity': 0,             # 關閉訓練日誌
    'eval_metric': 'logloss'    # 評估指標
}
```

#### 參數選擇理由
- **高樹數量(800)**: 提供足夠的模型容量
- **低學習率(0.03)**: 配合高樹數，防止過擬合
- **較大深度(9)**: 捕捉複雜的特徵交互
- **正則化(alpha=0.1, lambda=1.0)**: 控制模型複雜度
- **子採樣(0.8)**: 增加模型泛化能力

### 3. 閾值優化策略

#### Quantile方法
不使用固定閾值(0.5)，而是基於測試集機率分佈的分位數：
```python
FINAL_QUANTILE = 0.93  # 使用第93百分位數
FINAL_THR = np.quantile(test_prob, FINAL_QUANTILE)
```

**測試的Quantile值**:
- 0.90 (~10% 預測為陽性)
- 0.91 (~9% 預測為陽性)
- 0.92 (~8% 預測為陽性)
- **0.93** (~7% 預測為陽性) ← 主要提交版本
- 0.94 (~6% 預測為陽性)
- 0.95 (~5% 預測為陽性)

**為何選擇0.93**:
- 與訓練集警示帳戶比例(約0.31%)一致
- 平衡Precision和Recall
- 經驗證在Public Leaderboard表現最佳

## 訓練流程
```
1. 載入特徵 → 2. Hard Negative Mining → 3. 構建訓練集
                                              ↓
6. 輸出多版本 ← 5. Quantile閾值 ← 4. XGBoost訓練
```

## 輸出檔案說明

### 1. submission_YYYYMMDD_HHMMSS.csv
主要提交檔案，使用Quantile=0.93

格式:
```csv
acct,label
fcf31c5113d3424d8c7c3c42f6e8685a,0
e21dfa45e9904a2e8f0b8f5d4e5c9f8b,1
...
```

### 2. submission_q{91,92,94,95}_YYYYMMDD_HHMMSS.csv
不同閾值的替代版本，用於比較和A/B測試

### 3. feature_importance.csv
特徵重要性排序

格式:
```csv
feature,importance
pagerank,0.0823
date_similarity,0.0756
comm_alert_ratio,0.0645
...
```

## 模型性能

### 訓練效率
- 特徵數: 約50個
- 訓練樣本: 約10,000個（增強後）
- 訓練時間: 3-5分鐘
- 記憶體使用: 約2-4GB

### 預期F1-Score
- Local Validation: 0.22-0.25
- Public Leaderboard: 0.28-0.32
- Private Leaderboard: 0.30-0.32

## 可復現性保證

所有隨機過程使用固定種子:
```python
random_state=42
```

確保在相同環境下執行結果完全一致。

## 注意事項

1. **執行順序**: 必須先執行 `Preprocess/feature_engineering.py`
2. **計算資源**: 建議4核心CPU，16GB記憶體
3. **執行時間**: 約3-5分鐘（視硬體而定）