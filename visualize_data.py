import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re
import traceback
import sys
import japanize_matplotlib

def load_company_data(output_dir):
    """
    出力ディレクトリから会社別のデータを読み込む
    """
    output_dir = Path(output_dir)
    company_data = {}
    
    # ファイル名から会社コード、会社名、期間を抽出する正規表現
    pattern = r'FST01(\d{8})(\d{5})(\d{3})_([^-]+)-([^.]+)\.txt'
    
    # 出力ディレクトリの存在確認
    if not output_dir.exists():
        raise FileNotFoundError(f"出力ディレクトリが見つかりません: {output_dir}")
    
    # ファイル一覧を表示
    files = list(output_dir.glob('FST01*.txt'))
    print(f"見つかったファイル数: {len(files)}")
    
    for file_path in files:
        print(f"処理中のファイル: {file_path}")
        match = re.match(pattern, file_path.name)
        if match:
            fiscal_date = match.group(1)
            company_code = match.group(2)
            segment_code = match.group(3)
            company_name = match.group(4)
            period = match.group(5)
            
            # データを読み込む
            df = pd.read_csv(file_path, sep='\t', header=None, names=['勘定科目コード', '金額'])
            
            if company_code not in company_data:
                company_data[company_code] = {
                    'name': company_name,
                    'periods': {}
                }
            
            company_data[company_code]['periods'][period] = df['金額'].sum()
            print(f"会社コード {company_code}（{company_name}）の期間 {period} の合計金額: {company_data[company_code]['periods'][period]:,}")
    
    if not company_data:
        raise ValueError("データが見つかりませんでした。")
    
    return company_data

def plot_company_trends(company_data, output_dir):
    """
    会社別のトレンドを折れ線グラフで表示
    """
    plt.figure(figsize=(12, 6))
    
    for company_code, data in company_data.items():
        company_name = data['name']
        periods = data['periods']
        
        # 期間と金額のリストを作成
        x = sorted(periods.keys())
        y = [periods[p] for p in x]
        
        # 折れ線グラフを描画
        plt.plot(x, y, marker='o', label=f'{company_name}（{company_code}）')
    
    plt.title('会社別の期間推移')
    plt.xlabel('期間')
    plt.ylabel('合計金額')
    plt.grid(True)
    
    # 横軸の目盛りを設定
    plt.xticks(rotation=45)
    
    # 縦軸の目盛りを整数のみに設定
    y_ticks = plt.yticks()[0]
    plt.yticks(y_ticks, [int(y) for y in y_ticks])
    
    # 凡例が空でないことを確認
    if plt.gca().get_legend_handles_labels()[0]:
        plt.legend()
    else:
        print("警告: 凡例に表示するデータが見つかりませんでした。")
    
    # グラフを保存
    plt.tight_layout()  # ラベルが切れないように調整
    plt.savefig(f'{output_dir}/company_trends.png')
    plt.close()

def main():
    try:
        print("start")
        sys.stdout.flush()
        # 出力ディレクトリのパスを指定
        output_dir = 'output'
        
        # データを読み込む
        company_data = load_company_data(output_dir)
        
        # グラフを描画
        plot_company_trends(company_data, output_dir)
        print('グラフを company_trends.png として保存しました。')
        
    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")
        traceback.print_exc()

if __name__ == '__main__':
    main() 

# Usage:
# python visualize_data.py