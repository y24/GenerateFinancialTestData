import pandas as pd
import os
from pathlib import Path

def load_account_master():
    """勘定科目マスタを読み込む"""
    df = pd.read_csv('master/account_master.csv')
    # カラム名を英語に変換
    df = df.rename(columns={
        '勘定科目コード': 'account_code',
        '勘定科目名称': 'account_name',
        '貸借区分': 'balance_type',
        '必須': 'required'
    })
    # account_codeを文字列型に変換
    df['account_code'] = df['account_code'].astype(str)
    return df

def load_single_file(file_path):
    """単一のテストデータファイルを読み込む"""
    df = pd.read_csv(file_path, sep='\t', header=None, names=['account_code', 'amount'])
    df['account_code'] = df['account_code'].astype(str)
    return df

def check_balance(test_data, account_master):
    """貸借バランスをチェックする"""
    # 勘定科目マスタとテストデータを結合
    merged_data = pd.merge(
        test_data,
        account_master,
        on='account_code',
        how='left'
    )
    
    # 貸借区分に基づいて金額を計算
    debit_entries = merged_data[merged_data['balance_type'] == 1]
    credit_entries = merged_data[merged_data['balance_type'] == -1]
    
    debit_total = debit_entries['amount'].sum()
    credit_total = credit_entries['amount'].sum()
    
    # バランスチェック
    is_balanced = abs(debit_total - credit_total) < 0.01  # 小数点以下の誤差を許容
    
    return {
        'is_balanced': is_balanced,
        'debit_total': debit_total,
        'credit_total': credit_total,
        'difference': debit_total - credit_total,
        'details': merged_data
    }

def print_balance_report(file_name, result, account_master_dict):
    """バランスチェック結果を表示する"""
    print(f'\n========= {file_name} =========')
    print(f'科目数: {len(result["details"])}')
    print(f'借方合計: {result["debit_total"]:,.0f}円')
    print(f'貸方合計: {result["credit_total"]:,.0f}円')
    print(f'差額: {result["difference"]:,.0f}円')
    print(f'バランス状態: {"OK" if result["is_balanced"] else "NG"}')
    print('\n--- 科目別集計 ---')
    grouped = result['details'].groupby('account_code')['amount'].sum()
    
    for code, amount in grouped.items():
        name = account_master_dict.get(code, '不明')
        print(f'{code} ({name}): {amount:,.0f}円')

def print_summary_report(file_results):
    """ファイルごとのバランス状態を一覧表示する"""
    print('')
    # ファイル名の長さを取得して、見やすく整形
    max_filename_length = max(len(f) for f in file_results.keys())
    
    for file_name, result in sorted(file_results.items()):
        status = "OK" if result['is_balanced'] else "NG"
        # ファイル名を左寄せで、一定の幅で表示
        print(f"{file_name:<{max_filename_length}} ... {status}")

def main():
    try:
        # 勘定科目マスタの読み込み
        account_master = load_account_master()
        account_master_dict = dict(zip(account_master['account_code'], account_master['account_name']))
        
        # outputディレクトリ内の全ファイルを処理
        output_dir = Path('output')
        all_balanced = True
        file_results = {}  # ファイルごとの結果を保存
        
        for file_path in sorted(output_dir.glob('*.txt')):
            # ファイルごとにデータを読み込んでチェック
            test_data = load_single_file(file_path)
            result = check_balance(test_data, account_master)
            
            # 結果を保存
            file_results[file_path.name] = result
            
            # 詳細な結果を表示
            print_balance_report(file_path.name, result, account_master_dict)
            
            if not result['is_balanced']:
                all_balanced = False
        
        # 全体の判定とファイル別一覧を表示
        print('\n=== チェック結果 ===')
        print(f'判定結果: {"OK" if all_balanced else "NG (バランスの合わないファイルが存在します)"}')
        
        # ファイル別一覧を表示
        print_summary_report(file_results)
        
    except Exception as e:
        print(f'エラーが発生しました: {str(e)}')

if __name__ == '__main__':
    main()

# Usage:
# python CheckTestData.py
