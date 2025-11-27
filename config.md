## 超參數設定

### 1. 特徵工程參數

#### 1.1 時序切分
```python
CUTOFF_DATE = 121  # 預設使用全部資料；可通過命令列參數調整
```

#### 1.2 特徵定義閾值
```python
LARGE_TXN_QUANTILE = 0.75    # 大額交易定義（第75百分位數）
NIGHT_HOURS_START = 22       # 夜間交易開始時間（22:00）
NIGHT_HOURS_END = 6          # 夜間交易結束時間（06:00）
WEEKEND_DAYS = [0, 6]        # 週末定義（0=週一, 6=週日）
```

#### 1.3 圖特徵生成參數
```python
# PageRank 計算
PAGERANK_ALPHA = 0.85               # 阻尼係數
PAGERANK_MAX_ITER = 100             # 最大迭代次數

# Betweenness Centrality 採樣
BETWEENNESS_SAMPLE_THRESHOLD = 5000 # 節點數 > 5000 時進行採樣

# 社群檢測
COMMUNITY_DETECTION_MAX_NODES = 50000  # 節點數上限
```

---

### 2. Hard Negative Mining 參數

#### 2.1 高置信度負樣本篩選條件
```python
MIN_TXN_COUNT = 5          # 最小交易次數
MAX_TXN_COUNT = 1000       # 最大交易次數
MIN_AVG_AMOUNT = 0         # 最小平均交易金額
MAX_AVG_AMOUNT = 100000    # 最大平均交易金額（10萬）
MIN_ACTIVE_DAYS = 10       # 最小活躍天數
```

#### 2.2 Hard/Easy Negative 分類閾值
```python
HARD_NEG_PROB_MIN = 0.10   # Hard Negative 最低機率
HARD_NEG_PROB_MAX = 0.40   # Hard Negative 最高機率
EASY_NEG_PROB_MAX = 0.05   # Easy Negative 最高機率
```

**分類邏輯：**
- **Hard Negative**: `0.10 ≤ prob ≤ 0.40` （容易誤判的負樣本）
- **Easy Negative**: `prob < 0.05` （明顯的負樣本）

#### 2.3 負樣本採樣比例
```python
HARD_NEG_RATIO = 0.70      # Hard Negative 佔 70%
EASY_NEG_RATIO = 0.30      # Easy Negative 佔 30%
```

**實際採樣數量（相對於正樣本數量 n_pos）：**
- `target_hard = n_pos × 7`
- `target_easy = n_pos × 3`

#### 2.4 初步模型參數（用於識別 Hard Negatives）
```python
TEMP_MODEL_PARAMS = {
    'n_estimators': 300,
    'learning_rate': 0.05,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': 0,
    'eval_metric': 'logloss'
}
```

---

### 3. XGBoost 模型參數

#### 3.1 完整參數設定
```python
XGBOOST_PARAMS = {
    # 樹結構參數
    'n_estimators': 800,           # 樹的數量
    'learning_rate': 0.03,         # 學習率
    'max_depth': 9,                # 樹的最大深度
    'min_child_weight': 3,         # 最小葉子節點樣本權重
    
    # 採樣參數
    'subsample': 0.8,              # 樣本子採樣比例（每棵樹）
    'colsample_bytree': 0.8,       # 特徵子採樣比例（每棵樹）
    'colsample_bylevel': 0.8,      # 特徵子採樣比例（每層）
    
    # 正則化參數
    'gamma': 0.1,                  # 最小損失減少
    'reg_alpha': 0.1,              # L1 正則化（Lasso）
    'reg_lambda': 1.0,             # L2 正則化（Ridge）
    
    # 其他參數
    'random_state': 42,            # 隨機種子
    'n_jobs': -1,                  # 使用所有 CPU 核心
    'verbosity': 0,                # 不輸出訓練日誌
    'eval_metric': 'logloss'       # 評估指標
}
```

#### 3.2 參數選擇理由

| 參數 | 值 | 理由 |
|------|-----|------|
| `n_estimators` | 800 | 高樹數量搭配低學習率，提升泛化能力 |
| `learning_rate` | 0.03 | 降低學習率防止過擬合 |
| `max_depth` | 9 | 較深的樹捕捉複雜特徵交互（圖特徵、網絡關係） |
| `min_child_weight` | 3 | 防止在小樣本上過擬合 |
| `subsample` | 0.8 | 樣本採樣增加泛化能力 |
| `colsample_bytree` | 0.8 | 特徵採樣減少特徵間共線性 |
| `colsample_bylevel` | 0.8 | 層級特徵採樣進一步防止過擬合 |
| `gamma` | 0.1 | 控制樹的分裂 |
| `reg_alpha` | 0.1 | L1 正則化促進稀疏解 |
| `reg_lambda` | 1.0 | L2 正則化控制權重大小 |

---

### 4. 閾值優化參數

#### 4.1 Quantile 測試範圍
```python
QUANTILE_TEST_RANGE = [0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97]
```

#### 4.2 最終選擇
```python
FINAL_QUANTILE = 0.93                        # 主要提交版本（約預測 7% 為陽性）
ALTERNATIVE_QUANTILES = [0.91, 0.92, 0.94, 0.95]  # 額外提交版本
```

#### 4.3 Quantile 對應效果

| Quantile | 預測陽性比例 | 特性 |
|----------|-------------|------|
| 0.90 | ~10% | 高召回率、低精確率 |
| 0.91 | ~9% | - |
| 0.92 | ~8% | - |
| **0.93** | **~7%** | **主要提交（平衡點）** |
| 0.94 | ~6% | - |
| 0.95 | ~5% | 低召回率、高精確率 |
| 0.96 | ~4% | - |
| 0.97 | ~3% | - |

---

### 5. 系統級參數
```python
RANDOM_STATE = 42    # 隨機種子（確保可復現性）
N_JOBS = -1          # 使用所有可用 CPU 核心
VERBOSITY = 0        # 不輸出詳細訓練日誌
```

---

### 6. 特徵統計

#### 6.1 特徵數量分佈
```
總計：約 53 個特徵

1. 基礎統計特徵（~20 個）
   - 轉出特徵：9 個
   - 轉入特徵：8 個
   - 衍生特徵：5 個

2. 增強特徵（~15 個）
   - 時序特徵：7 個
   - 金額趨勢：4 個
   - 異常時段：2 個
   - 高階交互：3 個

3. 圖特徵（~12 個）
   - 度中心性：3 個
   - 重要性指標：1 個
   - 聚類係數：1 個
   - 介數中心性：1 個
   - 加權度：2 個
   - 核心數：1 個
   - 社群特徵：2 個

4. 警示交互特徵（~6 個）
   - 二元指標：2 個
   - 計數特徵：2 個
   - 比例特徵：2 個
```

---

### 7. 訓練流程總結
```
完整 Pipeline：
┌─────────────────────────────────────────────────────────────┐
│ 1. 資料載入 + 時序過濾（cutoff_date ≤ 121）                    │
├─────────────────────────────────────────────────────────────┤
│ 2. 特徵工程                                                   │
│    ├─ 基礎統計特徵（20 個）                                    │
│    ├─ 增強特徵（15 個）                                        │
│    ├─ 圖特徵（12 個）                                          │
│    └─ 警示交互特徵（6 個）                                      │
├─────────────────────────────────────────────────────────────┤
│ 3. Hard Negative Mining                                      │
│    ├─ 篩選高置信度負樣本                                       │
│    │   └─ 條件：5 ≤ txn ≤ 1000, amt < 100k, days ≥ 10      │
│    ├─ 訓練初步模型                                            │
│    │   └─ 參數：300 trees, lr=0.05, depth=6                 │
│    ├─ 分類 Hard/Easy Negatives                               │
│    │   ├─ Hard: 0.10 ≤ prob ≤ 0.40                          │
│    │   └─ Easy: prob < 0.05                                 │
│    └─ 採樣：70% Hard + 30% Easy                              │
├─────────────────────────────────────────────────────────────┤
│ 4. 最終模型訓練                                               │
│    └─ XGBoost：800 trees, lr=0.03, depth=9, 強正則化        │
├─────────────────────────────────────────────────────────────┤
│ 5. 閾值優化                                                   │
│    ├─ 測試 Quantile：0.90 ~ 0.97                            │
│    └─ 選擇：0.93（約 7% 陽性）                                │
├─────────────────────────────────────────────────────────────┤
│ 6. 生成提交檔案                                               │
│    ├─ 主要版本：Q93                                          │
│    └─ 額外版本：Q91, Q92, Q94, Q95                           │
└─────────────────────────────────────────────────────────────┘
```

---

### 8. 預期效能

#### 8.1 執行時間
```
- 特徵工程：5-10 分鐘
- Hard Negative Mining：2-3 分鐘
- 模型訓練：3-5 分鐘
- 總計：10-18 分鐘
```

#### 8.2 資源需求
```
- CPU：建議 4 核心以上
- 記憶體：建議 8 GB 以上
  - 特徵工程：2-4 GB
  - 模型訓練：3-5 GB
```

#### 8.3 預期 F1-Score
```
- Local Validation: 0.22-0.25
- Public Leaderboard: 0.28-0.32
- Private Leaderboard: 0.30-0.32
```

---
### 9. 注意事項


-  `random_state=42` 確保結果可復現
-  修改超參數後建議重新訓練並驗證效能
-  建議創建 `config.py` 集中管理、調整超參數
