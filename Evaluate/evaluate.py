#!/usr/bin/env python3
"""
evaluate.py

Local execution script for Bayesian cumulative logit mixed model (CLMM) using Bambi.
Reads 'merge_base_rater.csv', fits a cumulative link mixed model via bambi, and writes summary to CSV.
"""
import argparse
import pandas as pd
import bambi as bmb
import arviz as az

def run_clmm(data_path: str, output_path: str):
    # 1. データ読み込み
    df = pd.read_csv(data_path)

    # 2. 順序カテゴリへの変換（1.0, 1.5, …, 5.0）
    score_levels = [1.0 + 0.5 * i for i in range(9)]
    df['target'] = pd.Categorical(
        df['target'].astype(float),
        categories=score_levels,
        ordered=True
    )

    # 3. モデル式の定義
    formula = 'target ~ hearing_status + gender + (1|speaker) + (1|rater)'

    # 4. モデル構築と推定
    model = bmb.Model(formula, df, family='cumulative')
    trace = model.fit(draws=1000, tune=1000, cores=4, random_seed=42)

    # 5. 結果サマリー取得とCSV出力
    summary = az.summary(
        trace,
        var_names=['hearing_status', 'gender', '~speaker', '~rater']
    )
    summary.to_csv(output_path)
    print(f"Saved CLMM summary to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fit CLMM with Bambi')
    parser.add_argument(
        '--data', default='../merge_base_rater.csv',
        help='Path to input CSV file'
    )
    parser.add_argument(
        '--out', default='clmm_summary.csv',
        help='Path for output summary CSV'
    )
    args = parser.parse_args()
    run_clmm(args.data, args.out)
