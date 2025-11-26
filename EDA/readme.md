# EDA - 探索性資料分析

## 功能說明
本模組進行深度資料探索，生成視覺化圖表和分析報告，幫助理解資料特性。

## 主要程式

### exploratory_analysis.py
執行完整EDA分析，包含8張圖表和分析報告。

**輸入**:
- `data/acct_transaction.csv`
- `data/acct_alert.csv`
- `data/acct_predict.csv`

**輸出**:
- `visualizations/*.png`: 8張分析圖表
- `visualizations/EDA_REPORT.md`: 分析報告

**執行**:
```bash
python EDA/exploratory_analysis.py
```

## 分析內容

### 1. 資料概覽
- 交易筆數、警示帳戶數、待預測帳戶數
- 涉及警示帳戶的交易比例

### 2. 生成的圖表
1. `01_amount_distribution.png`: 交易金額分佈
2. `02_categorical_distribution.png`: 類別變數分佈
3. `03_time_distribution.png`: 時間分佈
4. `04_alert_comparison.png`: 警示vs非警示對比
5. `05_time_series.png`: 時間序列趨勢
6. `06_transaction_time.png`: 交易時間分佈
7. `07_outlier_detection.png`: 異常值檢測
8. `08_network_visualization.png`: 交易網絡視覺化

### 3. 關鍵發現
- 警示帳戶交易金額平均高6.5倍
- 警示帳戶交易頻率高6倍
- 深夜交易(22:00-06:00)比例異常
- 存在明顯的交易網絡結構

## 注意事項

- EDA為可選步驟，不影響模型訓練
- 生成圖表需要約2-3分鐘
- 網絡視覺化僅顯示Top20活躍節點