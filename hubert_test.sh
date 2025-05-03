#!/bin/sh

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="/home/akio/deaf_si:$PYTHONPATH"

# HubertOrdinalRegressionModel HubertCornModel AttentionHubertOrdinalRegressionModel AttentionHubertCornModel
export MODEL=HubertOrdinalRegressionModel

target_speaker=BF026
export SPEAKER="$target_speaker"
export TARGET="hubert_orm" 
echo "=== Training for $SPEAKER ==="
python3 hubert_train.py --config hubert.yaml
