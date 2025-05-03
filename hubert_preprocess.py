import os
import argparse
import pandas as pd
import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, HubertModel
from tqdm import tqdm


def extract_and_save(
    input_csv: str,
    target_dir: str,
    output_csv: str,
    model_name: str = "rinna/japanese-hubert-base",
    layer: int = -1
) -> None:
    """
    input_csv:   入力メタデータCSV (sourceカラム, featureカラムを含む)
    target_dir:  HuBERT特徴量(.pt)の保存先ディレクトリ
    output_csv:  更新後のCSV出力先
    model_name:  HuggingFace の日本語HuBERTモデル名
    layer:       抽出する hidden_states のレイヤーインデックス
    """
    os.makedirs(target_dir, exist_ok=True)
    df = pd.read_csv(input_csv)

    # feature extractor とモデルのロード
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model = HubertModel.from_pretrained(model_name)
    model.eval()

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting HuBERT features"):
        wav_path = row['source']
        # WAV読み込み
        speech, sr = torchaudio.load(wav_path)
        if speech.size(0) > 1:
            speech = speech.mean(dim=0, keepdim=True)
        speech = speech.squeeze(0).numpy()

        # 特徴抽出
        inputs = extractor(speech, sampling_rate=sr, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model(
                inputs.input_values,
                attention_mask=inputs.attention_mask,
                output_hidden_states=True
            )

        # 指定レイヤーの hidden_states
        hs = outputs.hidden_states[layer]  # (1, seq_len, hidden_size)
        hubert_feats = hs.squeeze(0)       # (seq_len, hidden_size)

        # 保存
        base = os.path.splitext(os.path.basename(wav_path))[0]
        feat_path = os.path.join(target_dir, f"{base}_hubert_layer{layer}.pt")
        torch.save({'hubert_feats': hubert_feats}, feat_path)

        # CSV更新
        df.at[idx, 'feature'] = feat_path

    # 更新後保存
    df.to_csv(output_csv, index=False)


def main():
    parser = argparse.ArgumentParser(description="Extract HuBERT features from specified layer")
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--target_dir", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--model_name", default="rinna/japanese-hubert-base")
    parser.add_argument("--layer", type=int, default=-1)
    args = parser.parse_args()

    extract_and_save(
        input_csv=args.input_csv,
        target_dir=args.target_dir,
        output_csv=args.output_csv,
        model_name=args.model_name,
        layer=args.layer
    )

if __name__ == '__main__':
    main()
