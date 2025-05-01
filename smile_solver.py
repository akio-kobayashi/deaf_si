import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Tuple
from torch.optim.lr_scheduler import ReduceLROnPlateau
from einops import rearrange
from smile_model import OrdinalRegressionModel
import smile_model
import math
import coral_loss
import os, sys

'''
 PyTorch Lightning 用 solver
'''
class LitOrd(pl.LightningModule):
    def __init__(self, config:dict) -> None:        
        super().__init__()
        self.config = config

        self.model = OrdinalRegressionModel(**config['model'])
        self.save_hyperparameters()
        self.num_correct = self.num_total = 0
        
    def forward(self, mfcc:Tensor, smile:Tensor, lengths:Tensor) -> Tensor:
        return self.model(mfcc, lengths, smile)

    def training_step(self, batch, batch_idx:int) -> Tensor:
        self.model.train()
        mfcc, smile, labels, ranks, lengths = batch

        logits = self.forward(mfcc, smile, lengths)
        loss = coral_loss.coral_loss(logits, labels)
        self.log_dict({'loss': loss.item()})

        return loss
    
    def validation_step(self, batch, batch_idx: int):
        self.model.eval()
        mfcc, smile, labels, ranks, lengths = batch

        with torch.no_grad():
            logits = self.forward(mfcc, smile, lengths)
            loss = coral_loss.coral_loss(logits, labels)
            self.log_dict({'valid_loss': loss})
            predicted = smile_model.logits_to_label(logits)
            self.num_correct += (predicted == ranks).sum().item()
            self.num_total += len(labels)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     **self.config['optimizer'])
        scheduler ={
            "scheduler": ReduceLROnPlateau(optimizer, **self.config['scheduler']),
            "monitor": "valid_loss"
        }
        return [optimizer], [scheduler]

    def on_validation_epoch_end(self):
        self.log_dict({'valid_acc': self.num_correct/self.num_total})
        self.num_correct = self.num_total = 0

    def save_model(self):
        full_path = os.path.join(self.config['logger']['save_dir'],
                                 self.config['logger']['name'],
                                 'version_' + str(self.config['logger']['version']),
                                 self.config['output_path']
                                 )
        torch.save(self.model.to('cpu').state_dict(), full_path)
