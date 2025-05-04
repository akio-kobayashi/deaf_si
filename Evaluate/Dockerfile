# Dockerfile for Python-based CLMM Evaluation Script

# 1. ベースイメージ選択 (Python 3.10 slim)
FROM python:3.10-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl g++ \
    && rm -rf /var/lib/apt/lists/*

# ---- Python パッケージ ----
#  * NumPy 1.23.5 で固定   （spawn あり／2.0 系を回避）
#  * PyMC 5.9.0            （bambi 0.12 と公式互換）
#  * Bambi 0.12.0
#  * ArviZ 0.16.1
RUN pip install --no-cache-dir \
      numpy==1.23.5 \
      pandas \
      scikit-learn \
      scipy \
      gdown \
      pymc==5.9.0 \
      bambi==0.12.0 \
      arviz==0.16.1

# 6. 実行コマンド
#    コンテナ起動時に評価スクリプトを実行
CMD ["bash"]  # Start a bash shell by default
