import torch
import pytorch_lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
import torch.utils.data as data
from smile_solver import LitOrd
import torch.utils.data as dat
import torch.multiprocessing as mp
from smile_dataset import SmileDataset
import smile_dataset as S
from callback import SaveEveryNEpochs
from argparse import ArgumentParser
import yaml
import warnings
warnings.filterwarnings('ignore')

def main(args, config:dict):

    model = LitOrd(config)

    train_dataset = SmileDataset(path=config['train_path'], 
                                  stat_path=config['stat_path']
    )
    train_loader = data.DataLoader(dataset=train_dataset,
                                   **config['process'],
                                   pin_memory=False,
                                   shuffle=True, 
                                   collate_fn=S.data_processing
    )

    valid_dataset = SmileDataset(path=config['valid_path'], 
                                  stat_path=config['stat_path']
    )
    valid_loader = data.DataLoader(dataset=valid_dataset,
                                   **config['process'],
                                   pin_memory=False,
                                   shuffle=False, 
                                   collate_fn=S.data_processing
    )

    callbacks = [
        pl.callbacks.ModelCheckpoint( **config['checkpoint'])
    ]
    logger = TensorBoardLogger(**config['logger'])

    trainer = pl.Trainer( callbacks=callbacks,
                          logger=logger,
                          num_sanity_val_steps=0,
                          **config['trainer'] )
    
    trainer.fit(model=model, ckpt_path=args.checkpoint, 
                train_dataloaders=train_loader, val_dataloaders=valid_loader)

    model.save_model()


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn', force=True)
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--gpus', nargs='*', type=int)
    args=parser.parse_args()

    torch.set_float32_matmul_precision('high')
    with open(args.config, 'r') as yf:
        config = yaml.safe_load(yf)

    if 'config' in config.keys():
        config = config['config']
        
    main(args, config) 
