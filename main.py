import subprocess
import sys
import os
import time
from datetime import datetime

print("="*80)
print("E.SUN AI Cup 2025 - 完整執行流程")
print("="*80)
print(f"開始時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================================
# Step 1: 檢查環境
# ============================================================================
print("\n" + "="*80)
print("Step 1: 檢查環境")
print("="*80)

required_files = [
    'step2.py',
    'step3.py'
]

data_files = []
for path in ['/mnt/project', 'data', '.']:
    if os.path.exists(f'{path}/acct_transaction.csv'):
        data_files = [
            f'{path}/acct_transaction.csv',
            f'{path}/acct_alert.csv',
            f'{path}/acct_predict.csv'
        ]
        break

missing_files = []
for f in required_files:
    if not os.path.exists(f):
        missing_files.append(f)

if missing_files:
    print("缺少必要檔案:")
    for f in missing_files:
        print(f"   - {f}")
    sys.exit(1)

if not data_files:
    print("找不到資料檔案")
    sys.exit(1)

print("環境檢查通過")
print(f"資料檔案: {len(data_files)} 個")

# ============================================================================
# Step 2: 特徵工程
# ============================================================================
print("\n" + "="*80)
print("Step 2: 特徵工程 (含圖特徵)")
print("="*80)

feature_file = 'features_enhanced/features_enhanced_v1.csv'

if os.path.exists(feature_file):
    print(f"特徵檔案已存在: {feature_file}")
    response = input("是否重新生成? (y/n): ").lower()
    if response != 'y':
        print("跳過特徵生成")
    else:
        print("\n執行 step2.py...")
        start_time = time.time()
        result = subprocess.run(['python', 'step2.py'], capture_output=False)
        elapsed = time.time() - start_time
        
        if result.returncode != 0:
            print("step2.py 執行失敗")
            sys.exit(1)
        
        print(f"✓ step2.py 完成 (耗時: {elapsed/60:.1f} 分鐘)")
else:
    print("\n執行 step2.py...")
    start_time = time.time()
    result = subprocess.run(['python', 'step2.py'], capture_output=False)
    elapsed = time.time() - start_time
    
    if result.returncode != 0:
        print("step2.py 執行失敗")
        sys.exit(1)
    
    print(f"✓ step2.py 完成 (耗時: {elapsed/60:.1f} 分鐘)")

# 檢查特徵檔案
if not os.path.exists(feature_file):
    print(f"特徵檔案未生成: {feature_file}")
    sys.exit(1)

file_size_mb = os.path.getsize(feature_file) / (1024 * 1024)
print(f"特徵檔案: {feature_file} ({file_size_mb:.1f} MB)")

# ============================================================================
# Step 3: 模型訓練與預測
# ============================================================================
print("\n" + "="*80)
print("Step 3: 模型訓練與預測")
print("="*80)

print("\n執行 step3.py...")
start_time = time.time()
result = subprocess.run(['python', 'step3.py'], capture_output=False)
elapsed = time.time() - start_time

if result.returncode != 0:
    print("step3.py 執行失敗")
    sys.exit(1)

print(f"✓ step3.py 完成 (耗時: {elapsed/60:.1f} 分鐘)")

# ============================================================================
# 總結
# ============================================================================
print("\n" + "="*80)
print("執行完成")
print("="*80)

# 列出生成的提交檔案
if os.path.exists('submissions'):
    submission_files = sorted([f for f in os.listdir('submissions') if f.endswith('.csv')])
    if submission_files:
        print(f"\n生成的提交檔案 ({len(submission_files)} 個):")
        for f in submission_files[-5:]:  # 顯示最新的5個
            filepath = f'submissions/{f}'
            size_kb = os.path.getsize(filepath) / 1024
            print(f"  - {f} ({size_kb:.1f} KB)")
        
        latest = submission_files[-1]
        print(f"\n最新提交檔案: submissions/{latest}")

print(f"\n結束時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

print("\n" + "="*80)
