import pandas as pd
from glob import glob
import os

# 1. 評定者ごとのCSVが格納されたディレクトリ
evaluator_dir = './csv/'  # 例: '/content/evaluators'

# 2. 統合結果の入力CSVパス
input_csv = './merge_base_asr_si_new_ctc.csv'      # 例: 'merged.csv'

# 3. 出力CSVパス
output_csv = './merge_base_rater.csv'

# ——— 評定者ごとのCSVを読み込み、key→rater_idマッピングを作成 ———
mapping = {}
for idx, filepath in enumerate(sorted(glob(os.path.join(evaluator_dir, '*.csv'))), start=1):
    df_eval = pd.read_csv(filepath)
    # 'key' 列から一意のキーを取得してマッピング
    keys = df_eval['key'].unique()
    for key in keys:
        mapping[key] = idx

# ——— 統合CSVを読み込み、rater列を追加 ———
df = pd.read_csv(input_csv)
df['rater'] = df['key'].map(mapping)

# ——— 出力CSVに保存 ———
df.to_csv(output_csv, index=False)
print(f"Rater列を追加したデータを '{output_csv}' に保存しました。")
