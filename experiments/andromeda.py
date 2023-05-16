# import torch 
# from colt5_attention import (
#     ConditionalRoutedFeedForward,
#     ConditionalRoutedAttention,
#     ConditionalRoutedTransformerBlock
# )

# #mock input 
# tokens = torch.randn(2, 32768, 512)
# mask = torch.ones(2, 32768).bool() # can handle lenghted sequence


# #feedforward 
# ff = ConditionalRoutedFeedForward(
#     dim=512,
#     light_ff_mult=0.5, #hidden dimension ratio of light branch
#     heavy_ff_mult = 4, #hidden dimension ratio of heavy branch
#     num_heavy_tokens = 1024 # heavy branch recieves only 1024 routed tokens of 32768
# )

# ff_out = ff(tokens, mask=mask)

# #attention
# attn = ConditionalRoutedAttention(
#     dim=512, # attention head dimension of light branch
#     light_dim_head=64, #number of attention heads for light branch
#     light_heads=8, # local attention receptive field for light
#     heavy_dim_head=64, #attention head dimension of heavy branch
#     heavy_heads = 8, #number of attentiion heads for heavy branch
#     num_heavy_tokens_q = 1024, # heavy branch recieves only 1024 routed tokens of 32768
#     num_heavy_tokens_kv = 1024 # heavy branch recieves only 1024 routed tokens of 32768
# )

# attn_out = attn(tokens, mask=mask) # (2, 32768, 512) - light and heavy branch summed


# #both attention and feedofrward with residual

# block = ConditionalRoutedTransformerBlock(
#     dim=512,
#     light_dim_head=64,
#     light_heads=8,
#     light_window_size=128,
#     heavy_dim_head=64,
#     heavy_heads=8,
#     light_ff_mult=0.5,
#     heavy_ff_mult=4,
#     num_heavy_ff_tokens=1024,
#     num_heavy_attn_tokens_q = 1024,
#     num_heavy_attn_tokens_kv = 1024
# )

# block_out = block(tokens, mask=mask)




# import torch.nn as nn
# #mock input  - 8192 input
# tokens = torch.randn(2, 8192, 512).cuda()

# #attention
# attn = ConditionalRoutedAutoregressiveAttention(
#     dim=512,
#     light_dim_head=64, # attention head dimension of light branch
#     light_heads=8, #number of attention heads for light branch
#     light_window_size=128, #locat attention receptive field for light 
#     heavy_window_size= 128, #the windowing for the routed heavy attention by default will be = to the light window size be aware if this is any greater than the light window size there may be tokens that would be missed by attention
#     heavy_dim_head= 64, #attention head dimension of heavy branch
#     heavy_heads= 8, #number of attention heads for heavy branhc
#     num_heavy_tokens_q = 32, #heavy branch receives only 32 out of 128 of the windowed queries (1024 query tokens)
#     num_heavy_tokens_kv = 1024, # heavy branch receives only 1024 routed tokens for key values
#     num_routed_kv = 2, # one can split the attention heads so that groups of heads attend to diffeent sets of keys - values (2 routing tokens in this case)
#     use_triton = True, #will need to use Triton for this to be visible, otherwise it is too slow and efficient with the number of iterations
#     use_flash_attn=True  # use flash attention in heavy branch
# ).cuda()

# attn_out = attn(tokens) + tokens #(2, 8192, 512) - output of attention with residual(prenorm included)

import torch
from colt5_attention import ConditionalRoutedTransformerBlock
import torch.nn as nn
from transformers import ( PreTrainedTokenizer,
                          PreTrainedTokenizerFast)
from typing import Union
from transformers import T5Tokenizer
import math
from typing import Union
import bitsandbytes as bnb


class AndromedaTokenizer:
    def __init__(self):
        # Initialize the T5 tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(
            "t5-large",
            additional_special_tokens=[],
            extra_ids=0,
            model_max_length=8192
        )
        self.pad_token_id = self.tokenizer.pad_token_id
        self.vocab_size = len(self.tokenizer)

    def tokenize_texts(self, texts):
        return self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

    def pad_sequences(self, sequences, max_length):
        padded_sequences = []
        for seq in sequences:
            padding_length = max_length - len(seq)
            padding_tensor = torch.full((padding_length,), self.pad_token_id, dtype=torch.long)
            padded_seq = torch.cat((seq, padding_tensor), dim=0)
            padded_sequences.append(padded_seq)
        return torch.stack(padded_sequences)


    def tokenize(self, samples):
        if isinstance(samples, str):
            samples = [samples]
        elif isinstance(samples, list) and all(isinstance(sample, str) for sample in samples):
            pass
        else:
            raise ValueError("Invalid input type for samples.")
            
        tokens = self.tokenizer(samples, return_tensors="pt", padding=True, truncation=True)
        text_tokens = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]
        labels = text_tokens.clone()

        return {
            "text_tokens": text_tokens,
            "attention_mask": attention_mask,
            "labels": labels
        }


class Andromeda(nn.Module):
    def __init__(self, dim, num_layers, max_seq_len, device, **kwargs):
        super(Andromeda, self).__init__()

        self.dim = dim

        self.tokenizer = AndromedaTokenizer()

        self.vocab_size = self.tokenizer.vocab_size

        self.register_buffer("pos_encoding", self.positional_encoding(max_seq_len, dim, device))

        self.embedding = bnb.nn.modules.Embedding(self.vocab_size, dim)

        self.pos_encoding = self.positional_encoding(max_seq_len, dim, device)

        self.transformer_blocks = nn.ModuleList([
            ConditionalRoutedTransformerBlock(dim=dim, **kwargs)
            for _ in range(num_layers)
        ])

        self.fc_out = nn.Linear(dim, self.vocab_size)

    def forward(self, input_ids, mask=None):
        seq_len = input_ids.size(1)
        x = self.embedding(input_ids) * torch.sqrt(torch.tensor(self.dim, device=input_ids.device))
        x = x + self.pos_encoding[:x.size(0), :].unsqueeze(1)
        x = self.dropout(x)


        for block in self.transformer_blocks:
            x = block(x, mask=mask)

        x = self.fc_out(x)
        return x
        
    @staticmethod
    def positional_encoding(max_seq_len, dim, device):
        pos_encoding = torch.zeros(max_seq_len, dim, device=device)
        position = torch.arange(0, max_seq_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float, device=device) * -(math.log(10000.0) / dim))
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        return pos_encoding





