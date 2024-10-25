import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Tuple
from einops import rearrange
from model import SIModel

class LightningSolver(pl.LightningModule):
    def __init__(self, config:dict) -> None:
        super().__init__()

        self.optim_config=config['optimizer']

        self.model = SIModel(config['model'])
        #self.loss = nn.MSELoss()
        self.loss = nn.HuberLoss(delta=1.0)
        self.save_hyperparameters()

    def forward(self, wave:Tensor, lengths:list) -> Tensor:
        return self.model(wave, lengths)

    def compute_loss(self, estimates, targets, valid=False):
        d={}
        _loss = self.loss(estimates, targets)
        if valid:
            d['valid_loss'] = _loss
        else:
            d['train_loss'] = _loss
       
        self.log_dict(d)

        return _loss

    def training_step(self, batch, batch_idx:int) -> Tensor:
        waves, lengths, targets = batch
        estimates = self.forward(waves, lengths)
        _loss = self.compute_loss(estimates, targets, valid=False)

        return _loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        waves, lengths, targets = batch
        estimates = self.forward(waves, lengths)
        _loss = self.compute_loss(estimates, targets, valid=True)

        return _loss


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                     **self.optim_config)
        return optimizer
