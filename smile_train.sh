#!/bin/sh

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="/home/akio/deaf_si:$PYTHONPATH"

#AttentionOrdinalRegressionModel CornModel AttentionCornModel:
#export MODEL=AttentionOrdinalRegressionModel
#export MODEL=AttentionCornModel
#export MODEL=OrdinalRegressionModel
export MODEL=CornModel
for target_speaker in BF026  BF070  BM082 F002  F005  F008  F013  F016  F020  M003  M006  M009  M012 \
				 M015  M018  M025 BF027  BM046  BM083  F003  F006  F009  F014  F018  M001 \
				 M004  M007  M010 M013  M016  M023  M028 BF069  BM047 F001   F004  F007  \
    				 F010  F015  F019  M002  M005  M008  M011  M014  M017  M024;
do
    export SPEAKER="$target_speaker"
    export TARGET="mfcc_lstm_corn"
    export EXCLUDE_SMILE_CATS='[]'
    export USE_MFCC=true
    export USE_SMILE=false    
    
    echo "=== Training for $SPEAKER ==="
    python3 smile_train.py --config smile.yaml
done
exit 

export MODEL=AttentionOrdinalRegressionModel
for target_speaker in BF026  BF070  BM082 F002  F005  F008  F013  F016  F020  M003  M006  M009  M012 \
			     M015  M018  M025 BF027  BM046  BM083  F003  F006  F009  F014  F018  M001 \
			     M004  M007  M010 M013  M016  M023  M028 BF069  BM047 F001   F004  F007  \
			     F010  F015  F019  M002  M005  M008  M011  M014  M017  M024;
do
    export SPEAKER="$target_speaker"
    export TARGET="mfcc_lstm_att_coral"
    export EXCLUDE_SMILE_CATS='[]'
    export USE_MFCC=true
    export USE_SMILE=false    
    
    echo "=== Training for $SPEAKER ==="
    python3 smile_train.py --config smile.yaml
done
