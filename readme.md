# 2025 AIcup - 玉山人工智慧挑戰賽

## 競賽說明
- 預測玉山帳戶是否為詐欺警示帳戶
- 為二元分類任務
- 評分指標：F1-score，最低門檻為 0.2
- 官方 Github ：https://github.com/esun-ai/esun-ai-competition-2025-tutorial
- slack 討論區：https://join.slack.com/t/esunaiopencompetition/shared_invite/zt-3elbcozz6-rcYMgU9lDpKt2lMHEkKtTQ

## 最佳成績
private leaderboard：35 / 790
Test Set：F1-score 	0.317135

## 專案結構
```
2025 AIcup_E.SUN - competition/
├── Preprocess/                          # 資料前處理
│   ├── feature_engineering.py           # 特徵工程主程式
│   └── README.md                        # 前處理說明文件
├── Model/                               # 模型訓練
│   ├── model_training.py                # 模型訓練主程式
│   └── README.md                        # 模型說明文件
├── EDA/                                 # 探索性資料分析
│   ├── exploratory_analysis.py          # EDA分析程式
│   └── README.md                        # EDA說明文件
├── data/                                # 資料檔案（不上傳至Git）
│   ├── acct_transaction.csv             # 交易資料
│   ├── acct_alert.csv                   # 警示帳戶清單
│   └── acct_predict.csv                 # 待預測帳戶清單
├── features/                            # 生成的特徵檔案
│   ├── pagerank_scores.csv              # pagerank 的計算結果
│   └── features.csv                     # 完整特徵集
├── submissions/                         # 提交檔案
│   ├── feature_importance.csv           # 各個 features 重要性計算結果
│   └── submission_YYYYMMDD_HHMMSS.csv   # 預測結果
│
├── visualizations/                      # 存放 EDA 結果
├── main.py                              # 主執行程式
├── requirements.txt                     # Python套件需求
└── README.md                            # 本說明文件
```


## 執行指令

###  方法一
```bash
# Step 1: EDA
python EDA/exploratory_analysis.py
# Step 2: Feature engineering
python Preprocess/feature_engineering.py
# Step 3: Training Model
python Model/model_training.py
```

### 方法二
```bash
python main.py
```
## 建模概述

### 整體策略
採用三階段Pipeline：
1. **特徵工程**: 生成基礎統計、圖特徵、時序特徵等約50個特徵
2. **負樣本增強**: Hard Negative Mining 提升模型區分能力
3. **模型訓練**: 優化的XGBoost + Quantile閾值策略

### 特徵工程（Preprocess）
生成以下特徵組：
- **基礎統計特徵（~20個）**: 交易金額、次數、帳戶類型等
- **增強特徵（~15個）**: date_similarity、時序統計、金額趨勢
- **圖特徵（~12個）**: PageRank、度中心性、聚類係數、社群檢測
- **警示交互特徵（~6個）**: 與已知警示帳戶的交互關係

**關鍵特徵**:
- `date_similarity`: 交易時間規律性
- `pagerank`: 帳戶在交易網絡中的重要性
- `comm_alert_ratio`: 所屬社群中警示帳戶比例

### 模型訓練（Model）

#### Hard Negative Mining
1. 篩選高置信度負樣本
2. 訓練初步模型識別難分負樣本
3. 構建增強訓練集（Hard 70% + Easy 30%）

## 輸出檔案

執行完成後會生成：

1. **特徵檔案**:
   - `features_enhanced/features_enhanced_v1.csv`

2. **提交檔案** (多個閾值版本):
   - `submissions/submission_YYYYMMDD_HHMMSS.csv` (主要版本)
   - `submissions/submission_q91_YYYYMMDD_HHMMSS.csv`
   - `submissions/submission_q92_YYYYMMDD_HHMMSS.csv`
   - `submissions/submission_q94_YYYYMMDD_HHMMSS.csv`
   - `submissions/submission_q95_YYYYMMDD_HHMMSS.csv`

3. **特徵重要性**:
   - `submissions/feature_importance.csv`

## 注意事項

1. **資料檔案**: 需將競賽資料放置於 `data/` 資料夾
2. **計算資源**: 圖特徵生成需要較多記憶體和時間
3. **隨機種子**: 所有隨機過程使用 `random_state=42` 確保可復現性

## 參考資料

- [官方GitHub](https://github.com/esun-ai/esun-ai-competition-2025-tutorial)
- [Slack討論區](https://join.slack.com/t/esunaiopencompetition/shared_invite/zt-3elbcozz6-rcYMgU9lDpKt2lMHEkKtTQ)

