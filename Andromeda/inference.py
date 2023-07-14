import sys

sys.dont_write_bytecode = True

import os
import time

import tracemalloc

from pprint import pprint

import torch

from tqdm import tqdm

import numpy as np

from transformers import set_seed

from tokenizer import AndromedaTokenizer
from model import andromeda_model

from data_streaming import DatasetElement

class EvalAndromeda:
    def __init__(self, path, seed=42, device=None):
        self.path = path
        self.seed = seed
        
        self.device = device
        
        if self.device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        set_seed(self.seed)

        # Tokenizer

        self.tokenizer = AndromedaTokenizer()
        
        # Model
        
        self.model = andromeda_model
        
        # Checkpoint

        self.model.load_state_dict(torch.load(self.path))
        self.model.eval()
        
        # Device
        
        self.model = self.model.to(self.device)
        
        # Metrics
        
        self.metrics = {}
        self.reset_metrics()
        
    def reset_metrics(self):
        self.metrics = {
            'generation_steps': None,

            'times_forward': [],
            'time_forward_average': None,
            
            'memory_usages': [],
            'memory_usage_average': None,
            
            'time_end_to_end': None,
            'throughput': None
        }
        
    def get_num_params(self):
        num_params = sum(param.numel() for param in self.model.parameters() if param.requires_grad)

        return num_params

    def generate(self, prompt, generation_steps=32):
        # Make sure all of the metrics are reset at every generation
        
        self.reset_metrics()

        self.metrics['generation_steps'] = generation_steps
        
        tokens     = self.tokenizer.encode(prompt)
        tokens_new = []
        
        time_end_to_end_0 = time.time()
        
        # Generation loop
        
        for _ in range(generation_steps):
            tokens_tensor = torch.tensor([tokens], device=self.device)
            
            # Forward pass
            
            tracemalloc.start()

            time_forward_0 = time.time()

            logits = self.model(tokens_tensor, return_loss=False)[:, -1] # No loss, take the output for the last token

            time_forward_1 = time.time()
            
            _, memory_usage = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            self.metrics['memory_usages'].append(memory_usage)
            
            time_forward = time_forward_1 - time_forward_0
            self.metrics['times_forward'].append(time_forward)

            next_token = torch.argmax(logits).item()

            # Save the newly generated token
            
            tokens.append(next_token)
            tokens_new.append(next_token)

        time_end_to_end_1 = time.time()
        
        time_end_to_end                 = time_end_to_end_1 - time_end_to_end_0
        self.metrics['time_end_to_end'] = time_end_to_end
 
        decoded = self.tokenizer.decode(tokens)

        self.metrics['time_forward_average'] = np.mean(self.metrics['times_forward'])
        self.metrics['memory_usage_average'] = np.mean(self.metrics['memory_usages'])
        
        self.metrics['throughput'] = generation_steps / np.sum(self.metrics['times_forward'])
        
        return tokens_new, decoded

def main():
    prompt = 'My name is'

    andromeda = EvalAndromeda(path='checkpoints/step_44927_6656/pytorch_model.bin')
    
    num_params = andromeda.get_num_params()
    print(f'The model has {num_params} parameters')
    
    _, output = andromeda.generate(prompt)
    
    for metric, value in andromeda.metrics.items():
        print(f'{metric}: {value}\n')
    
    print('\n')
    
    print(output)
    
if __name__ == '__main__':
    main()
