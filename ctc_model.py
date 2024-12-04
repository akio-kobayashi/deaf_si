from transformers import AutoModel
from peft import get_peft_model, LoraConfig, PeftConfig, PeftModel
import torch
import torch.nn as nn
from einops import rearrange
from argparse import ArgumentParser
from model import W2VModel
from model import SIModel

class SIModelCTC(SIModel):
    def __init__(self, config, output_dim=1):
        super().__init__(config, output_dim)
        self.lstm_ctc = nn.LSTM(1024, 256, 1,
                                batch_first=True,
                                bidirectional=True)
        self.linear_ctc = nn.Linear(512, config['output_class'])
        self.downsample_ratio = self.w2v.model.config.conv_stride[-1]
        
    def forward(self, x, lengths=None, ctc=False):
        if isinstance(x, list):
            print(x)
            raise ValueError('type mismatch')
        
        y = self.w2v(x) # y = ()

        # CTC (b t f) -> (b t n), t=max valid lengths of input sequences
        if ctc is True:
            #print(y.last_hidden_state.shape)
            #print(y.extract_features.shape)
            #exit(1)
            v, _ = self.lstm_ctc(y.last_hidden_state)
            v = self.linear_ctc(v)
            return v
        else:
            z = [ rearrange(x, '(b c) t f -> b c t f', c=1) for x in y.hidden_states ]
            z = torch.cat(z, axis=1)
            z = torch.mean(self.linear1(z), axis=1) # b t f
            if lengths is not None:
                z = torch.nn.utils.rnn.pack_padded_sequence(z, lengths.cpu(), batch_first=True,
                                                            enforce_sorted=False)
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
        
    def valid_ctc_lengths(self, input_lengths):
        valid_lengths = torch.div(input_lengths, self.downsample_ratio, rounding_mode="floor")
        return valid_lengths
    
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
