# Preprocess - 資料前處理模組

## 功能說明
本模組負責從原始交易資料中提取、工程化特徵，生成用於模型訓練的完整特徵集。

## 主要程式

### feature_engineering.py
完整特徵工程Pipeline，生成約50個特徵。

**輸入**:
- `data/acct_transaction.csv`: 原始交易資料
- `data/acct_alert.csv`: 警示帳戶清單
- `data/acct_predict.csv`: 待預測帳戶清單

**輸出**:
- `features_enhanced/features_enhanced_v1.csv`: 完整特徵集（約50個特徵）

**執行**:
```bash
python Preprocess/feature_engineering.py
```

## 特徵工程詳細說明

### 1. 基礎統計特徵（~20個）
- 轉出/轉入金額統計: sum, mean, std, max, min
- 交易次數: out_cnt, in_cnt
- 交易對手數: unique_to_acct, unique_from_acct
- 跨行交易比例: cross_bank_out_ratio, cross_bank_in_ratio
- 自我交易比例: self_txn_ratio
- 衍生比例: out_in_cnt_ratio, out_in_amt_ratio, Connection_Ratio

### 2. 增強特徵（~15個）
- **date_similarity**: 交易時間規律性（關鍵特徵）
- **時序統計**: 
  - active_days: 活躍天數
  - activity_shift: 早期/晚期活動變化
- **金額趨勢**:
  - out_amt_cv: 金額變異係數
  - out_amt_range_ratio: 金額極差比
- **異常時段**:
  - night_txn_ratio_out: 深夜交易比例（22:00-06:00）
  - weekend_txn_ratio: 週末交易比例
- **高階交互**:
  - avg_txn_per_partner: 每交易對手平均交易數
  - large_txn_ratio: 大額交易比例（>Q3）

### 3. 圖特徵（~12個）
基於交易網絡的結構特徵：
- **度中心性**: in_degree, out_degree, total_degree
- **PageRank**: 帳戶重要性評分
- **聚類係數**: clustering_coef（局部密集度）
- **介數中心性**: betweenness（橋接能力）
- **加權度**: weighted_in_degree, weighted_out_degree
- **核心數**: k_core（網絡核心程度）
- **社群特徵**:
  - community_id: 所屬社群ID
  - comm_alert_ratio: 社群內警示帳戶比例

### 4. 警示交互特徵（~6個）
與已知警示帳戶的交互關係：
- has_sent_to_alert: 是否曾轉帳給警示帳戶
- has_recv_from_alert: 是否曾收到警示帳戶轉帳
- alert_send_count: 轉帳給警示帳戶的次數
- alert_recv_count: 收到警示帳戶轉帳的次數
- ratio_alert_send: 警示轉出比例
- ratio_alert_recv: 警示轉入比例

## 時間複雜度

- 基礎特徵: O(n)，n為交易數
- 圖特徵: O(n + m*log(m))，m為帳戶數
- 總執行時間: 約5-10分鐘（取決於硬體）

## 注意事項

1. **記憶體需求**: 圖特徵生成需要約8-12GB記憶體
2. **時序正確性**: 所有特徵確保不使用未來資訊（避免data leakage）
3. **缺失值處理**: 所有NaN填充為0，Inf/-Inf替換為0

## 參數說明

- `CUTOFF_DATE`: 時間切點，預設121（使用全部資料）
- 可通過命令列參數調整: `python feature_engineering.py 100`