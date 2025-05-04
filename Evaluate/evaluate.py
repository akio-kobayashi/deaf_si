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
    # 1. Load data
    df = pd.read_csv(data_path)

    # 2. Convert to ordered categorical target
    score_levels = [1.0 + 0.5 * i for i in range(9)]
    df['target'] = pd.Categorical(
        df['target'].astype(float),
        categories=score_levels,
        ordered=True
    )

    # 3. Define formula
    formula = 'target ~ hearing_status + gender + (1|speaker) + (1|rater)'

    # 4. Fit model with Bambi (single chain to avoid RNG spawn issues)
    model = bmb.Model(formula, df, family='cumulative')
    trace = model.fit(
        draws=1000,
        tune=1000,
        chains=1,
        cores=1,
        random_seed=None  # Avoid numpy RNG spawn issue
    )

    # 5. Extract summary and save to CSV
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
