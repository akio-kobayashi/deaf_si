from transformers import AutoModel
from peft import get_peft_model, LoraConfig, PeftConfig, PeftModel
import torch
import torch.nn as nn
from einops import rearrange
from argparse import ArgumentParser

class W2VModel(nn.Module):
    model_id = "facebook/wav2vec2-xls-r-300m"

    def __init__(self, config, save_dir=None):        
        super().__init__()
        self.model = None
        if save_dir is None:
            '''
                lora_attention_dim...attention dimension of LoRA
                lora_alpha...Scaleing factor of LoRA
                lora_dropout...Dropout factor of LoRA layers
            '''
            peft_config = LoraConfig(
                inference_mode=False,
                r=config['lora_attention_dim'],
                lora_alpha=config['lora_alpha'],
                lora_dropout=config['lora_dropout'],
                target_modules=["v_proj", "q_proj"],
            )
            model = AutoModel.from_pretrained(W2VModel.model_id)
            self.model = get_peft_model(model, peft_config)
        else:
            self.load(save_dir)

    def forward(self, x):
        return self.model(x, output_hidden_states=True)

    def save(self, save_dir):
        self.model.save_pretrained(save_dir)

    def load(self, save_dir):
        config = PeftConfig.from_pretrained(save_dir)
        model = AutoModel.from_pretrained(W2VModel.model_id)
        self.model = PeftModel.from_pretrained(model, save_dir)

class SIModel(nn.Module):
    def __init__(self, config, output_dim=1):
        super().__init__()
        self.w2v = W2VModel(config)
        self.linear1 = nn.Linear(config['input_dim'], config['output_dim'])
        self.lstm = nn.LSTM(config['output_dim'], config['hidden_dim'], num_layers=config['num_layers'],
                                batch_first=True, dropout=config['dropout'], bidirectional=True, proj_size=config['proj_dim'])
        self.linear2 = nn.Linear(config['proj_dim']*config['num_layers']*2, output_dim)

    def forward(self, x, lengths=None):
        y = self.w2v(x)
        z = [ rearrange(x, '(b c) t f -> b c t f', c=1) for x in y.hidden_states ]
        z = torch.cat(z, axis=1)
        z = torch.mean(self.linear1(z), axis=1) # b t f
        if lengths is not None:
            z = torch.nn.utils.rnn.pack_padded_sequence(z, lengths, batch_first=True, enforce_sorted=False)
        o, (z, _) = self.lstm(z)
        if lengths is not None:
            o, output_shape = torch.nn.utils.rnn.pad_packed_sequence(o, batch_first=True)
            states = torch.zeros(o.shape[0], o.shape[-1]).to(o.device)
            for i, lenghts in enumerate(lengths):
                states[i, o.shape[-1]//2:] = o[i, 0, o.shape[-1]//2:]
                states[i, :o.shape[-1]//2] = o[i, lengths[i]-1, :o.shape[-1]//2]
            z = states
        else:
            z = rearrange(z, 'd b f -> b (d f)')
        z = self.linear2(z)
        return z
    
if __name__ == '__main__':
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args=parser.parse_args()

    with open(args.config, 'r') as yf:
        config = yaml.safe_load(yf)
    model = SIModel(config['model'])

    x = torch.randn(4, 16_000 * 5)

    y = model(x)
    print(y.shape)
    
