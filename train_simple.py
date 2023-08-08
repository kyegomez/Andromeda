import gzip
import random

import numpy as np
import torch
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader, Dataset

from Andromeda.model import Andromeda

from Andromeda.core.transformer import Decoder, Transformer
from Andromeda.core.autoregressive_wrapper import AutoregressiveWrapper
# constants

NUM_BATCHES = int(1e5)
BATCH_SIZE = 4
GRADIENT_ACCUMULATE_EVERY = 4
LEARNING_RATE = 1e-4
VALIDATE_EVERY  = 100
GENERATE_EVERY  = 500
GENERATE_LENGTH = 1024
SEQ_LEN = 1024

# helpers

def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))

# instantiate GPT-like decoder model

model = Transformer(
    num_tokens=50432,
    max_seq_len=8192,
    use_abs_pos_emb=False,
    # embedding_provider=embedding_provider,
    attn_layers=Decoder(
        dim=2560,
        depth=32,
        dim_head=128,
        heads=24,
        alibi_pos_bias=12,
        alibi_num_heads=True,
        rotary_xpos=True,
        attn_flash=True,
        # deepnorm=deepnorm,
        # shift_tokens=shift_tokens,
        attn_one_kv_head=True,
        qk_norm=True,
        attn_qk_norm=True,
        attn_qk_norm_dim_scale=True
    )
)

model = AutoregressiveWrapper(model)

model.cuda()

# prepare enwik8 data

with gzip.open('./data/enwik8.gz') as file:
    data = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()
    train_x, valid_x = np.split(data, [int(90e6)])
    data_train, data_val = torch.from_numpy(train_x), torch.from_numpy(valid_x)

class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len - 1, (1,))
        full_seq = self.data[rand_start: rand_start + self.seq_len + 1].long()
        return full_seq.cuda()

    def __len__(self):
        return self.data.size(0) // self.seq_len

train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
val_dataset   = TextSamplerDataset(data_val, SEQ_LEN)
train_loader  = cycle(DataLoader(train_dataset, batch_size = BATCH_SIZE, drop_last = True))
val_loader    = cycle(DataLoader(val_dataset, batch_size = BATCH_SIZE, drop_last = True))

# optimizer

optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# training

for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc='training'):
    model.train()

    for __ in range(GRADIENT_ACCUMULATE_EVERY):
        loss = model(next(train_loader))
        (loss / GRADIENT_ACCUMULATE_EVERY).backward()

    print(f'training loss: {loss.item()}')
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optim.step()
    optim.zero_grad()

    if i % VALIDATE_EVERY == 0:
        model.eval()
        with torch.no_grad():
            loss = model(next(val_loader))
            print(f'validation loss: {loss.item()}')
    
        #save the model weights
        torch.save(model.state_dict(), f"./model_{i}.pth")

    if i % GENERATE_EVERY == 0:
        model.eval()
        inp = random.choice(val_dataset)[:-1]
        prime = decode_tokens(inp)
        print('%s \n\n %s', (prime, '*' * 100))

        sample = model.generate(inp, GENERATE_LENGTH)
        output_str = decode_tokens(sample)
        print(output_str)