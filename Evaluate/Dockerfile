# Dockerfile for Python-based CLMM Evaluation Script

# 1. ベースイメージ選択 (Python 3.10 slim)
FROM python:3.10-slim

# 2. 作業ディレクトリを作成・設定
WORKDIR /app

# 3. 必要なシステムパッケージのインストール（軽量化のため最小限）
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# 4. Pythonライブラリのインストール
#    - pandas, numpy: データ操作
#    - scikit-learn, scipy: 指標計算
#    - gdown: Google Drive からのファイルダウンロード（必要に応じて）
#    - bambi, arviz: CLMM 推定用ライブラリ
# Install Python packages with compatible numpy for Bambi
RUN pip install --no-cache-dir pandas numpy==1.26.4 scikit-learn scipy gdown bambi arviz

# 5. スクリプトをコピー
#    - evaluate.py: 評価スクリプト（CSVを読み込み評価し、結果をCSV出力）
COPY evaluate.py /app/evaluate.py

# 6. 実行コマンド
#    コンテナ起動時に評価スクリプトを実行
CMD ["bash"]  # Start a bash shell by default
