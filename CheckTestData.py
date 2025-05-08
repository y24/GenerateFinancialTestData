import pandas as pd
import os
from pathlib import Path

def load_account_master():
    """勘定科目マスタを読み込む"""
    df = pd.read_csv('account_master.csv')
    # カラム名を英語に変換
    df = df.rename(columns={
        '勘定科目コード': 'account_code',
        '勘定科目名称': 'account_name',
        '貸借区分': 'balance_type',
        '必須': 'required'
    })
    return df

def load_test_data():
    """テストデータを読み込む"""
    output_dir = Path('output')
    test_data = []
    
    for file in output_dir.glob('*.txt'):
        # タブ区切りのテキストファイルを読み込み
        df = pd.read_csv(file, sep='\t', header=None, names=['account_code', 'amount'])
        test_data.append(df)
    
    return pd.concat(test_data, ignore_index=True)

def check_balance(test_data, account_master):
    """貸借バランスをチェックする"""
    # 勘定科目マスタとテストデータを結合
    merged_data = pd.merge(
        test_data,
        account_master,
        on='account_code',
        how='left'
    )
    
    # 借方・貸方の合計を計算（金額が正の場合は借方、負の場合は貸方）
    debit_total = merged_data[merged_data['amount'] > 0]['amount'].sum()
    credit_total = abs(merged_data[merged_data['amount'] < 0]['amount'].sum())
    
    # バランスチェック
    is_balanced = abs(debit_total - credit_total) < 0.01  # 小数点以下の誤差を許容
    
    return {
        'is_balanced': is_balanced,
        'debit_total': debit_total,
        'credit_total': credit_total,
        'difference': debit_total - credit_total
    }

def main():
    try:
        # データ読み込み
        account_master = load_account_master()
        test_data = load_test_data()
        
        # バランスチェック実行
        result = check_balance(test_data, account_master)
        
        # 結果表示
        print('=== 貸借バランスチェック結果 ===')
        print(f'借方合計: {result["debit_total"]:,.0f}円')
        print(f'貸方合計: {result["credit_total"]:,.0f}円')
        print(f'差額: {result["difference"]:,.0f}円')
        print(f'バランス状態: {"OK" if result["is_balanced"] else "NG"}')
        
    except Exception as e:
        print(f'エラーが発生しました: {str(e)}')

if __name__ == '__main__':
    main()
