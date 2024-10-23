import transformers import AutoModel
from peft import get_peft_model, LoraConfig, PeftConfig, PeftModel
import torch
import torch.nn as nn
from einops import rearrange

class W2VModel(nn.Module):
    model_id = "facebook/wav2vec2-xls-r-300m"

    def __init__(self, lora_attention_dim=8, lora_alpha=32, lora_dropout=0.1, save_dir=None):        
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
                r=lora_attention_dim,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["k_proj", "q_proj"],
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
    def __init__(self, config):
        self.w2v = W2VModel(config.lora_attention_dim, config.lora_alpha, config.lora_dropout)
        self.linear1 = nn.Linear(config.input_dim, config.output_dim)
        self.lstm = nn.LSTM(config.output_dim, config.hidden_dim, 1,
                                batch_first=True, dropout=config.dropout, bidirectional=True)
        self.linear2 = nn.Linear(config.hidden_dim, 1)

    def foward(self, x):
        y = self.w2v(x)
        z = [ rearrange(x, '(b c) f t -> b c t f', c=1) for x in y.hidden_states ]
        z = torch.cat(z, axis=1)
        z = torch.mean(self.linear1(z), axis=1) # b t f
        z = self.lstm(z)
        z = self.linear2(z)
        return z
    