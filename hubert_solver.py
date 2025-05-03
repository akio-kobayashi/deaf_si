import os
import torch
from torch import Tensor
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau

import coral_loss
import corn_loss

from hubert_model import (
    HubertOrdinalRegressionModel,
    AttentionHubertOrdinalRegressionModel,
    HubertCornModel,
    AttentionHubertCornModel
)


class LitHubert(pl.LightningModule):
    """
    PyTorch Lightning solver for HuBERT-based ordinal regression (CORAL/CORN).
    Expects batch: (huberts, labels, ranks, lengths)
    """
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config

        # モデル選択
        model_cfg = config['model'].copy()
        class_name = model_cfg.pop('class_name', 'HubertOrdinalRegressionModel')
        model_map = {
            'HubertOrdinalRegressionModel': HubertOrdinalRegressionModel,
            'AttentionHubertOrdinalRegressionModel': AttentionHubertOrdinalRegressionModel,
            'HubertCornModel': HubertCornModel,
            'AttentionHubertCornModel': AttentionHubertCornModel,
        }
        ModelClass = model_map[class_name]
        self.model = ModelClass(**model_cfg)

        self.save_hyperparameters()
        self.num_correct = 0
        self.num_total = 0

    def forward(self, hubert_feats: Tensor) -> Tensor:
        """
        Predict logits for given HuBERT embeddings.
        """
        return self.model(hubert_feats)

    def training_step(self, batch, batch_idx: int) -> Tensor:
        huberts, labels, ranks, lengths = batch
        logits = self.forward(huberts)
        # CORN or CORAL の損失を選択
        if isinstance(self.model, (HubertCornModel, AttentionHubertCornModel)):
            loss = corn_loss.corn_loss(logits, labels)
        else:
            loss = coral_loss.coral_loss(logits, labels)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx: int) -> None:
        huberts, labels, ranks, lengths = batch
        logits = self.forward(huberts)
        if isinstance(self.model, (HubertCornModel, AttentionHubertCornModel)):
            loss = corn_loss.corn_loss(logits, labels)
        else:
            loss = coral_loss.coral_loss(logits, labels)
        self.log('val_loss', loss, prog_bar=True)

        # 精度計算
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).sum(dim=1) + 1
        self.num_correct += (preds == ranks).sum().item()
        self.num_total += ranks.size(0)

    def on_validation_epoch_end(self) -> None:
        acc = self.num_correct / self.num_total if self.num_total > 0 else 0.0
        self.log('val_acc', acc, prog_bar=True)
        self.num_correct = 0
        self.num_total = 0

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), **self.config['optimizer'])
        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, **self.config['scheduler']),
            'monitor': 'val_loss'
        }
        return [optimizer], [scheduler]

    def save_model(self) -> None:
        save_dir = os.path.join(
            self.config['logger']['save_dir'],
            self.config['logger']['name'],
            f"version_{self.config['logger']['version']}"
        )
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, self.config['output_path'])
        torch.save(self.model.cpu().state_dict(), path)
