import sys

sys.dont_write_bytecode = True

import random
import tqdm

import numpy as np

import torch

import torch.optim as optim
from torch.nn import functional as F

from transformers import AutoTokenizer, get_scheduler
from datasets import load_dataset

def sequence_slice(sequence, step_size): # This function slices the string into parts of length "step_size"
    for idx in range(0, len(sequence), step_size):
        yield sequence[idx:idx + step_size]

class DatasetElement: # This class contains information about a single dataset, multiple "DatasetElement" (s) could be combined and weighted to create a data mixture
    def __init__(self, dataset_name, dataset_data_column, tokenizer, dataset_split='train', sequence_length=32, batch_size=8):
        self.dataset_name        = dataset_name
        self.dataset_data_column = dataset_data_column

        self.dataset_split = dataset_split
        
        self.sequence_length = sequence_length
        self.batch_size      = batch_size

        self.tokenizer = tokenizer
        
        self.seed = 42

        self.dataset_skip_num = 0 # We would like to start from a specific dataset index when resuming the training run

        self.dataset_buffer_size = 100 # 100_000 # Avoiding spike losses when streaming=True, might create RAM-related issues
        
        self.dataset_pretokenized = False

        self.dataset     = iter(load_dataset(self.dataset_name, split=self.dataset_split, streaming=True).shuffle(buffer_size=self.dataset_buffer_size, seed=self.seed).skip(self.dataset_skip_num + 1))
        self.dataset_idx = 0

        self.data_stack = []
        
        self.max_ingestion_attempts = 128

    def sequence_slice(self, sequence, step_size): # This function creates slices from the sequence and pads the ones shorter than the specified sequence length
        for idx in range(0, len(sequence), step_size):
            chunk = sequence[idx:idx + step_size]

            if idx + step_size >= len(sequence): # Padding
                padding_length = step_size - len(chunk)
                chunk.extend([self.tokenizer.tokenizer.pad_token_id] * padding_length)

            yield chunk

    def ingest(self): # Ingest a new element from the dataset
        data             = next(self.dataset)[self.dataset_data_column]
        self.dataset_idx += 1

        tokens = data

        if not self.dataset_pretokenized:
            tokens = self.tokenizer.encode(data)

        chunks = self.sequence_slice(tokens, self.sequence_length)

        for chunk in chunks: # Create sequences from the chunks
            if chunk is not None:
                self.data_stack.append(chunk)

    def get_batch(self): # Get a batch for training
        batch_tokens     = []
        batch_embeddings = []

        batch_positions_idxs            = []
        batch_embeddings_positions_idxs = []

        max_length = 0
        
        while len(batch_tokens) < self.batch_size: # Append examples to the batch until you reach the desired batch size
            ingestion_attempts = 0
            
            while len(self.data_stack) == 0:
                if ingestion_attempts >= self.max_ingestion_attempts:
                    break
                
                self.ingest()
                
                ingestion_attempts += 1

            max_length = self.sequence_length

            tokens = self.data_stack[0]
            batch_tokens.append(tokens)

            del self.data_stack[0]

        return batch_tokens
