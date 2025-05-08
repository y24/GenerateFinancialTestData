import pandas as pd
import numpy as np
from pathlib import Path

def load_master(master_csv):
    """
    マスタCSVから勘定科目コード、貸借区分、必須フラグを読み込む
    """
    df = pd.read_csv(
        master_csv,
        dtype={'勘定科目コード': str, '勘定科目名称': str, '貸借区分': int},
        keep_default_na=False
    )
    # 必須列がなければ0で初期化、空白や欠損は0扱い
    if '必須' not in df.columns:
        df['必須'] = 0
    else:
        df['必須'] = df['必須'].replace('', '0').astype(int)
    return df[['勘定科目コード', '貸借区分', '必須']]

def load_company_master(company_master_csv):
    """
    会社マスタCSVから会社コードを読み込む
    重複チェックも行う
    """
    df = pd.read_csv(
        company_master_csv,
        dtype={'会社コード': str},
        keep_default_na=False
    )
    
    # 重複チェック
    duplicates = df['会社コード'].duplicated()
    if duplicates.any():
        duplicate_codes = df[duplicates]['会社コード'].unique()
        raise ValueError(f"会社コードが重複しています: {', '.join(duplicate_codes)}")
    
    return df['会社コード'].tolist()

# サンプリング比率をタプルに変換
def parse_ratio_range(ratio_str: str) -> tuple[float, float]:
    """カンマ区切りの文字列を比率範囲のタプルに変換する
    
    Args:
        ratio_str: カンマ区切りの文字列 (例: "0.7,0.9")
        
    Returns:
        比率範囲のタプル (例: (0.7, 0.9))
    """
    return tuple(map(float, ratio_str.split(',')))

def sample_accounts(master_df, sample_ratio_range=(0.8, 1.0), min_accounts=2):
    """
    必須フラグONの勘定科目は必ず含め、残りをランダムサンプリングして返す
    sample_ratio_range: 全体母数に対するサンプリング比の範囲
    """
    required = master_df[master_df['必須'] == 1]
    optional = master_df[master_df['必須'] != 1]
    ratio = np.random.uniform(*sample_ratio_range)
    total_sample = max(int(ratio * len(master_df)), min_accounts)
    optional_n = max(total_sample - len(required), 0)
    sampled_optional = optional.sample(optional_n) if optional_n > 0 else pd.DataFrame(columns=master_df.columns)
    sampled = pd.concat([required, sampled_optional], ignore_index=True)
    sampled = sampled.sample(frac=1).reset_index(drop=True)
    return sampled[['勘定科目コード', '貸借区分']]

def generate_amounts(accounts_df, prev_abs=None, noise_level=0.1,
                     min_amount=1000, max_amount=10000000, rounding_unit=1,
                     allow_negative=False):
    count = len(accounts_df)
    drcr = accounts_df['貸借区分'].values
    # 絶対値生成（対数スケールで生成）
    if prev_abs is None:
        # 対数スケールで一様分布を生成
        log_min = np.log10(min_amount)
        log_max = np.log10(max_amount)
        log_vals = np.random.uniform(log_min, log_max, size=count)
        abs_vals = np.power(10, log_vals)
    else:
        factors = np.random.normal(1, noise_level, size=count)
        abs_vals = prev_abs * factors
        abs_vals = np.clip(abs_vals, min_amount, max_amount)
    # 貸借バランス調整
    debit_mask = drcr == 1
    credit_mask = drcr == -1
    debit_vals = abs_vals[debit_mask]
    credit_vals = abs_vals[credit_mask]
    total_debit = debit_vals.sum()
    total_credit = credit_vals.sum()
    scaled_credit = credit_vals * (total_debit / total_credit) if total_credit > 0 else np.zeros_like(credit_vals)
    debit_rounded = np.round(debit_vals)
    credit_rounded = np.round(scaled_credit)
    diff = total_debit - credit_rounded.sum()
    if credit_rounded.size > 0:
        credit_rounded[0] += diff
    balanced_abs = np.empty_like(abs_vals)
    balanced_abs[debit_mask] = debit_rounded
    balanced_abs[credit_mask] = credit_rounded
    # 指定単位で丸め
    if rounding_unit and rounding_unit > 1:
        # 借方と貸方を別々に丸める
        rounded_abs = np.empty_like(balanced_abs)
        rounded_abs[debit_mask] = np.round(balanced_abs[debit_mask] / rounding_unit) * rounding_unit
        rounded_abs[credit_mask] = np.round(balanced_abs[credit_mask] / rounding_unit) * rounding_unit
        
        # 丸め後の貸借差額を計算
        rd_debit = rounded_abs[debit_mask]
        rd_credit = rounded_abs[credit_mask]
        td = rd_debit.sum()
        tc = rd_credit.sum()
        delta = td - tc
        
        # 差分を指定単位で丸めて調整
        if credit_mask.sum() > 0:
            idx0 = np.where(credit_mask)[0][0]
            adjustment = np.round(delta / rounding_unit) * rounding_unit
            rounded_abs[idx0] += adjustment
        
        final_abs = rounded_abs
    else:
        final_abs = balanced_abs
    
    if allow_negative:
        # 貸借区分に基づいて符号を決定し、ランダムに反転
        base_signs = drcr.copy()
        flip_mask = np.random.choice([-1, 1], size=count)
        signed = final_abs * base_signs * flip_mask
    else:
        # 常に正の数を生成
        signed = final_abs
    return signed.astype(int), np.abs(final_abs)

def generate_company_data(accounts_df, periods, noise_level=0.1, rounding_unit=1, allow_negative=False):
    results = []
    prev_abs = None
    for _ in range(periods):
        signed, prev_abs = generate_amounts(
            accounts_df, prev_abs, noise_level, rounding_unit=rounding_unit,
            allow_negative=allow_negative)
        results.append(signed)
    return results

def main(master_csv, company_master_csv, output_dir, periods, noise_level=0.1, rounding_unit=1,
         parent_ratio_range=(0.9, 1.0), child_ratio_range=(0.7, 0.9), allow_negative=False):
    master_df = load_master(master_csv)
    company_codes = load_company_master(company_master_csv)
    
    if len(company_codes) < 2:
        raise ValueError("会社マスタには少なくとも親会社と1つの子会社が必要です")
    
    parent_code = company_codes[0]
    child_codes = company_codes[1:]
    num_companies = len(company_codes)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, company_code in enumerate(company_codes):
        # 親会社は多め、子会社は少なめ
        if idx == 0:
            ratio_range = parent_ratio_range
        else:
            ratio_range = child_ratio_range

        # サンプリング
        accounts = sample_accounts(master_df, sample_ratio_range=ratio_range)
        # データ生成
        period_data = generate_company_data(accounts, periods, noise_level, rounding_unit, allow_negative)

        # ファイル出力
        for p, amounts in enumerate(period_data, start=1):
            df_out = pd.DataFrame({
                '勘定科目コード': accounts['勘定科目コード'].values,
                '金額': amounts
            })
            file_path = output_dir / f"{company_code}_X{p}.txt"
            df_out.to_csv(file_path, sep='\t', index=False, header=False)

    print(f"Generated data for {num_companies} companies over {periods} periods in {output_dir}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate test data for consolidated accounting system')
    parser.add_argument('--master', required=True, help='account master CSV path')
    parser.add_argument('--company-master', required=True, help='company master CSV path')
    parser.add_argument('--output', type=str, default='output', help='output directory')
    parser.add_argument('--periods', type=int, default=2, help='number of periods')
    parser.add_argument('--noise', type=float, default=0.1, help='noise level for trends')
    parser.add_argument('--rounding', type=int, default=100, help='rounding unit (e.g.100,1000)')
    parser.add_argument('--parent-ratio', type=str, default='0.9,1.0',
                        help='Parent sample ratio range, e.g. "0.9,1.0"')
    parser.add_argument('--child-ratio', type=str, default='0.7,0.9',
                        help='Child sample ratio range, e.g. "0.7,0.9"')
    parser.add_argument('--allow-negative', action='store_true',
                        help='Allow negative amounts in generated data')
    args = parser.parse_args()

    parent_ratio = parse_ratio_range(args.parent_ratio)
    child_ratio = parse_ratio_range(args.child_ratio)

    main(args.master, args.company_master, args.output, args.periods, args.noise, args.rounding,
         parent_ratio, child_ratio, args.allow_negative)

    # Usage:
    # python GenerateTestData.py --master account_master.csv --company-master company_master.csv