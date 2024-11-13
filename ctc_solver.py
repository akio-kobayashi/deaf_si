import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Tuple
from einops import rearrange
from ctc_model import SIModelCTC
from loss_func import WeightedKappaLoss

class LightningSolverCTC(pl.LightningModule):
    def __init__(self, config:dict) -> None:
        super().__init__()

        self.optim_config=config['optimizer']

        if config['loss']['type'] == 'kappa':
            self.model = SIModelCTC(config['model'], 9)
            self.loss = WeightedKappaLoss(num_classes=9, mode=config['loss']['mode'])
        else:
            self.model = SIModelCTC(config['model'])
            if config['loss']['type'] == 'mse':
             self.loss = nn.MSELoss()
            else:
                self.loss = nn.HuberLoss(config['loss']['delta'])

        self.ctc_loss = nn.CTCLoss()
        self.weight = config['ctc_weight']
        self.return_lengths = config['return_length']
        
        self.save_hyperparameters()
        
    def forward(self, wave:Tensor, lengths:list, ctc:bool) -> Tensor:
        return self.model(wave, lengths, ctc)

    def compute_loss(self, estimates, values, ctc_estimates, labels, lengths_ctc, lengths_label, valid=False):
        d={}
        ctc_estimates = rearrange(ctc_estimates, 'b t c -> t b c')
        _ctc_loss = self.ctc_loss(ctc_estimates, labels, lengths_ctc, lengths_label)
        _si_loss = self.loss(estimates, values)
        _loss = self.weight * _ctc_loss + _si_loss
        
        if valid:
            d['valid_loss'] = _loss
            d['valid_ctc_loss'] = _ctc_loss
            d['valid_si_loss'] = _si_loss
        else:
            d['train_loss'] = _loss
            d['train_ctc_loss'] = _ctc_loss
            d['train_si_loss'] = _si_loss
       
        self.log_dict(d)

        return _loss

    def training_step(self, batch, batch_idx:int) -> Tensor:
        # (waves_si, values, lengths_si_tensor, lengths_si), (waves_ctc, labels, lengths_ctc_tensor, lengths_ctc, label_lengths_tensor, label_lengths)
        batch1, batch2  = batch
        # si
        waves_si, values, lengths_si_tensor, _ = batch1
        if self.return_lengths is False:
            lengths_si_tensor = None
        si_estimates = self.forward(waves_si, lengths_si_tensor, ctc=False)
        # ctc
        waves_ctc, labels, lengths_ctc_tensor, _, label_lengths_tensor, _ = batch2
        ctc_estimates = self.forward(waves_ctc, None, ctc=True)
        valid_lengths_ctc_tensor = self.model.valid_ctc_lengths(lengths_ctc_tensor)
        ctc_estimates = F.log_softmax(ctc_estimates)
        
        _loss = self.compute_loss(si_estimates, values, ctc_estimates, labels, valid_lengths_ctc_tensor, label_lengths_tensor, valid=False)

        return _loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        batch1, batch2  = batch
        # si
        waves_si, values, lengths_si_tensor, _ = batch1
        if self.return_lengths is False:
            lengths_si_tensor = None
        si_estimates = self.forward(waves_si, lengths_si_tensor, ctc=False)
        # ctc
        waves_ctc, labels, lengths_ctc_tensor, _, label_lengths_tensor, _ = batch2
        ctc_estimates = self.forward(waves_ctc, None, ctc=True)
        valid_lengths_ctc_tensor = self.model.valid_ctc_lengths(lengths_ctc_tensor)
        ctc_estimates = F.log_softmax(ctc_estimates)
        
        _loss = self.compute_loss(si_estimates, values, ctc_estimates, labels, valid_lengths_ctc_tensor, label_lengths_tensor, valid=True)

        return _loss


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                     **self.optim_config)
        return optimizer
