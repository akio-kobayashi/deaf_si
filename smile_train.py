import torch
import pytorch_lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
import torch.utils.data as data
from smile_solver import LitOrd
import pandas as pd
import torch.multiprocessing as mp
from smile_dataset import SmileDataset
import smile_dataset as S
from argparse import ArgumentParser
import yaml
from string import Template
import os
import warnings
warnings.filterwarnings('ignore')


def main(args, config:dict):
    # 1) モデルとデータローダーの準備
    model = LitOrd(config)

    # SmileDataset は ablation 用に exclude_smile_cats のみを受け取る
    train_dataset = SmileDataset(
        path=config['train_path'],
        stat_path=config['stat_path'],
        exclude_smile_cats=config.get('exclude_smile_cats', [])
    )
    train_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=config['batch_size'],
        num_workers=config['process']['num_workers'],
        pin_memory=False,
        shuffle=True,
        collate_fn=S.data_processing
    )

    valid_dataset = SmileDataset(
        path=config['valid_path'],
        stat_path=config['stat_path'],
        exclude_smile_cats=config.get('exclude_smile_cats', [])
    )
    valid_loader = data.DataLoader(
        dataset=valid_dataset,
        batch_size=config['batch_size'],
        num_workers=config['process']['num_workers'],
        pin_memory=False,
        shuffle=False,
        collate_fn=S.data_processing
    )

    # 2) コールバックとロガー
    checkpoint_cb = pl.callbacks.ModelCheckpoint(**config['checkpoint'])
    callbacks = [checkpoint_cb]
    logger = TensorBoardLogger(**config['logger'])

    # 3) Trainer の作成と学習
    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=logger,
        num_sanity_val_steps=0,
        **config['trainer']
    )
    trainer.fit(
        model=model,
        ckpt_path=args.checkpoint,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader
    )

    # 4) ベストモデルで検証データを予測し CSV 出力
    best_path = checkpoint_cb.best_model_path
    if best_path:
        best_model = LitOrd.load_from_checkpoint(best_path, config=config)
        best_model.eval()

        df = pd.read_csv(config['valid_path'])
        predictions = []
        for mfcc, smile, labels, ranks, lengths in valid_loader:
            mfcc = mfcc.to(best_model.device)
            smile = smile.to(best_model.device)
            lengths = lengths.to(best_model.device)
            with torch.no_grad():
                preds = best_model.model.predict(mfcc, lengths, smile)
            predictions.extend(preds.cpu().tolist())

        scores = [1.0+(r-1)*0.5 for r in predictions]
        df['predict'] = scores
        df.to_csv(config['output_csv'], index=False)
        print(f"Saved predictions to {config['output_csv']}")
        # compute accuracy
        correct = (df['intelligibility'] == df['predict']).sum()
        total = len(df)
        accuracy = correct / total if total > 0 else 0.0
        print(f"Validation accuracy: {accuracy:.4f} ({correct}/{total})")
    else:
        print("No best model checkpoint found, skipping prediction.")


def load_config(path: str)-> dict:
    raw = open(path, 'r', encoding='utf-8').read()
    rendered = Template(raw).substitute(
        SPEAKER=os.getenv('SPEAKER', ''),
        TARGET=os.getenv('TARGET', ''),
        EXCLUDE_SMILE_CATS=os.getenv('EXCLUDE_SMILE_CATS', '[]'),
        USE_SMILE=os.getenv('USE_SMILE', 'true'),
        USE_MFCC=os.getenv('USE_MFCC', 'true')
    )
    cfg = yaml.safe_load(rendered)
    if not isinstance(cfg, dict):
        raise RuntimeError(f"Expected dict after YAML load, got {type(cfg)}")
    if 'config' in cfg:
        cfg = cfg['config']
    return cfg

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn', force=True)
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()

    torch.set_float32_matmul_precision('high')

    # 設定ファイルを読み込む前にプレースホルダを展開
    config = load_config(args.config)
    main(args, config)
