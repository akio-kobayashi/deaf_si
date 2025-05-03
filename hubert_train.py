import os
import torch
import pytorch_lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
import torch.utils.data as data
from argparse import ArgumentParser
import pandas as pd
import yaml
from string import Template
import warnings
warnings.filterwarnings('ignore')

# HuBERT用モジュール
from hubert_dataset import HubertDataset, data_processing
from lit_hubert_solver import LitHubert


def load_config(path: str) -> dict:
    """
    YAML設定を読み込み、環境変数のプレースホルダを展開して返す
    """
    raw = open(path, 'r', encoding='utf-8').read()
    rendered = Template(raw).substitute(
        **os.environ
    )
    cfg = yaml.safe_load(rendered)
    if 'config' in cfg:
        cfg = cfg['config']
    return cfg


def main(args, config: dict):
    # 1) モデルとデータローダー準備
    model = LitHubert(config)

    train_dataset = HubertDataset(path=config['train_path'])
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['process']['num_workers'],
        pin_memory=True,
        collate_fn=data_processing
    )

    valid_dataset = HubertDataset(path=config['valid_path'])
    valid_loader = data.DataLoader(
        valid_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['process']['num_workers'],
        pin_memory=True,
        collate_fn=data_processing
    )

    # 2) コールバック・ロガー
    checkpoint_cb = pl.callbacks.ModelCheckpoint(**config['checkpoint'])
    logger = TensorBoardLogger(**config['logger'])

    # 3) Trainer 作成・学習実行
    trainer = pl.Trainer(
        callbacks=[checkpoint_cb],
        logger=logger,
        num_sanity_val_steps=0,
        **config['trainer']
    )
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
        ckpt_path=args.checkpoint
    )

    # 4) ベストモデルによる予測・CSV出力
    best_path = checkpoint_cb.best_model_path
    if best_path:
        best = LitHubert.load_from_checkpoint(best_path, config=config)
        best.eval()

        df = pd.read_csv(config['valid_path'])
        preds = []
        for huberts, labels, ranks, lengths in valid_loader:
            huberts = huberts.to(best.device)
            with torch.no_grad():
                logits = best.model(huberts)
                probs = torch.sigmoid(logits)
                batch_preds = (probs > 0.5).sum(dim=1) + 1
            preds.extend(batch_preds.cpu().tolist())

        # スコアに戻す
        scores = [1.0 + (r-1)*0.5 for r in preds]
        df['predict'] = scores
        df.to_csv(config['output_csv'], index=False)
        print(f"Saved predictions to {config['output_csv']}")
        # 精度計算
        correct = (df['intelligibility'] == df['predict']).sum()
        acc = correct / len(df) if len(df)>0 else 0.0
        print(f"Validation accuracy: {acc:.4f} ({correct}/{len(df)})")
    else:
        print("No best model checkpoint found, skipping prediction.")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()
    torch.set_float32_matmul_precision('high')
    config = load_config(args.config)
    main(args, config)
