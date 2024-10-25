import torch
import pytorch_lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
import torch.utils.data as data
from solver import LightningSolver
from speech_dataset import SpeechDataset
from speech_dataset import data_processing
from model import SIModel
from argparse import ArgumentParser
import yaml
import warnings
import check_gpu
warnings.filterwarnings('ignore')

def train(config:dict, target_speaker):

    lite = LightningSolver(config)
       
    train_dataset = SpeechDataset(csv_path=config['csv'], target_speaker=target_speaker)
    train_loader = data.DataLoader(dataset=train_dataset,
                                   batch_size=config['batch_size'],
                                   num_workers=1,
                                   pin_memory=True,
                                   shuffle=True, 
                                   collate_fn=lambda x: data_processing(x))
    train_df = train_dataset.get_df()
    valid_dataset = SpeechDataset(csv_path=config['csv'], target_speaker=target_speaker, train_df=train_df)
    valid_loader = data.DataLoader(dataset=valid_dataset,
                                   batch_size=config['batch_size'],
                                   num_workers=1,
                                   pin_memory=True,
                                   shuffle=False, 
                                   collate_fn=lambda x: data_processing(x))
           
    callbacks = [
        pl.callbacks.ModelCheckpoint( **config['checkpoint'])
    ]
    logger = TensorBoardLogger(**config['logger'])

    trainer = pl.Trainer( callbacks=callbacks,
                          logger=logger,
                          **config['trainer'] )
    trainer.fit(model=lite, train_dataloaders=train_loader, val_dataloaders=valid_loader)
        
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--target_speaker', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    #parser.add_argument('--gpu', type=int, default=0)
    args=parser.parse_args()

    #print(torch.cuda.device_count())
    #print(torch.cuda.get_device_name())
    #print(torch.cuda.get_device_capability())
    best_gpu = check_gpu.get_free_gpu()
    #print(best_gpu)
    if best_gpu is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{best_gpu}")
        print(f"最も空きメモリの大きいGPUを選択: {device}")
    else:
        device = torch.device("cpu")
        print("GPUが見つからなかったため、CPUを使用します")
    torch.set_float32_matmul_precision('high')
    with open(args.config, 'r') as yf:
        config = yaml.safe_load(yf)

    if 'config' in config.keys():
        config = config['config']
    config['logger']['name'] = args.model_name
    train(config, args.target_speaker)
