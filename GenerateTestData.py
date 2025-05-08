import pandas as pd
import numpy as np
from pathlib import Path

def load_master(master_csv):
    """
    マスタCSVから勘定科目コード、貸借区分、必須フラグを読み込む
    """
    df = pd.read_csv(
        master_csv,
        dtype={
            '勘定科目コード': str,
            '勘定科目名称': str,
            '貸借区分': int,
            '必須': int
        }
    )
    return df[['勘定科目コード', '貸借区分', '必須']]

def sample_accounts(master_df, sample_ratio_range=(0.8, 1.0), min_accounts=2):
    """
    必須フラグONの勘定科目は必ず含め、残りをランダムサンプリングして返す
    """
    # 必須勘定科目
    required = master_df[master_df['必須'] == 1]
    optional = master_df[master_df['必須'] != 1]

    # サンプリング数の決定
    ratio = np.random.uniform(*sample_ratio_range)
    total_sample = max(int(ratio * len(master_df)), min_accounts)
    optional_n = max(total_sample - len(required), 0)

    # 任意勘定科目のサンプリング
    sampled_optional = optional.sample(optional_n) if optional_n > 0 else pd.DataFrame(columns=master_df.columns)

    # 結合してシャッフル
    sampled = pd.concat([required, sampled_optional], ignore_index=True)
    sampled = sampled.sample(frac=1).reset_index(drop=True)
    return sampled[['勘定科目コード', '貸借区分']]

def generate_amounts(accounts_df, prev_abs=None, noise_level=0.1,
                     min_amount=100, max_amount=1000000, rounding_unit=1):
    """
    各勘定科目の金額を生成し、貸借を一致させ、指定桁で丸める
    - prev_abs: 前期の絶対値金額（numpy array、Noneなら初期生成）
    - rounding_unit: 丸め単位（例:100,1000）
    """
    count = len(accounts_df)
    drcr = accounts_df['貸借区分'].values

    # 絶対値の初期生成またはノイズ付与
    if prev_abs is None:
        abs_vals = np.random.randint(min_amount, max_amount + 1, size=count).astype(float)
    else:
        factors = np.random.normal(1, noise_level, size=count)
        abs_vals = prev_abs * factors
        abs_vals = np.clip(abs_vals, 0, None)

    # 貸借グループ分離とバランス調整
    debit_mask = drcr == 1
    credit_mask = drcr == -1
    debit_vals = abs_vals[debit_mask]
    credit_vals = abs_vals[credit_mask]
    total_debit = debit_vals.sum()
    total_credit = credit_vals.sum()
    if total_credit > 0:
        scaled_credit = credit_vals * (total_debit / total_credit)
    else:
        scaled_credit = np.zeros_like(credit_vals)
    debit_rounded = np.round(debit_vals)
    credit_rounded = np.round(scaled_credit)
    diff = total_debit - credit_rounded.sum()
    if credit_rounded.size > 0:
        credit_rounded[0] += diff
    balanced_abs = np.empty_like(abs_vals)
    balanced_abs[debit_mask] = debit_rounded
    balanced_abs[credit_mask] = credit_rounded

    # 丸め処理（下位切り捨て）
    if rounding_unit and rounding_unit > 1:
        rounded_abs = np.floor(balanced_abs / rounding_unit) * rounding_unit
        rd_debit = rounded_abs[debit_mask]
        rd_credit = rounded_abs[credit_mask]
        td = rd_debit.sum()
        tc = rd_credit.sum()
        delta = td - tc
        if credit_mask.sum() > 0:
            idx0 = np.where(credit_mask)[0][0]
            rounded_abs[idx0] += delta
        final_abs = rounded_abs
    else:
        final_abs = balanced_abs

    signed = final_abs * drcr
    return signed.astype(int), np.abs(final_abs)

def generate_company_data(accounts_df, periods, noise_level=0.1, rounding_unit=1):
    """
    指定期間分のデータを生成する
    """
    results = []
    prev_abs = None
    for _ in range(periods):
        signed, prev_abs = generate_amounts(
            accounts_df, prev_abs, noise_level,
            rounding_unit=rounding_unit
        )
        results.append(signed)
    return results

def main(master_csv, output_dir, num_child, periods, noise_level=0.1, rounding_unit=1):
    master_df = load_master(master_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_companies = 1 + num_child
    for idx in range(num_companies):
        name = 'parent' if idx == 0 else f'child{idx}'
        accounts = sample_accounts(master_df)
        period_data = generate_company_data(
            accounts, periods, noise_level, rounding_unit
        )

        for p, amounts in enumerate(period_data, start=1):
            df_out = pd.DataFrame({
                '勘定科目コード': accounts['勘定科目コード'].values,
                '金額': amounts
            })
            file_path = output_dir / f"{name}_period_{p}.txt"
            df_out.to_csv(file_path, sep='\t', index=False, header=False)

    print(f"Generated data for {num_companies} companies over {periods} periods in {output_dir}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate test data for consolidated accounting system')
    parser.add_argument('--master', required=True, help='account master CSV path')
    parser.add_argument('--output', required=True, help='output directory')
    parser.add_argument('--child', type=int, default=1, help='number of child companies')
    parser.add_argument('--periods', type=int, default=2, help='number of periods')
    parser.add_argument('--noise', type=float, default=0.1, help='noise level for trends')
    parser.add_argument('--rounding', type=int, default=100, help='rounding unit (e.g.100,1000)')
    args = parser.parse_args()
    main(args.master, args.output, args.child, args.periods, args.noise, args.rounding)

    # python GenerateTestData.py --master account_master.csv --output output