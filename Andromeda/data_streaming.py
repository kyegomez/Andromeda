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

class Block: # Base unit of data representation
    def __init__(self, content, holds_embeddings=False):
        assert isinstance(content, list), 'The provided content should be a list'

        self.content          = content
        self.holds_embeddings = holds_embeddings # Boolean
    
    def get_indices(self): # Get idxs (tokens)
        raise NotImplementedError()
    
    def get_embeddings(self): # Get embeddings (pre-computed by a different encoder, as Kosmos-X would)
        raise NotImplementedError()

class IndicesBlock(Block): # This block only holds idxs
    def __init__(self, content):
        super().__init__(content)

    def get_tokens(self):
        return self.content

class EmbeddingBlock(Block): # This block only holds embeddings
    def __init__(self, content):
        super().__init__(content, holds_embeddings=True)

    def get_tokens(self):
        raise NotImplementedError()

    def get_embeddings(self):
        return self.content

class Sequence: # This is a sequence of different blocks, it coule be used to represent multi-modal sequences
    def __init__(self, blocks=[]):
        self.blocks = blocks
    
    def add_block(self, block: Block):
        self.blocks.append(block)

    def get_data(self):
        tokens     = []
        embeddings = []

        positions_idxs            = []
        embeddings_positions_idxs = []

        for block in self.blocks:
            position_idx_offset    = max(positions_idxs + embeddings_positions_idxs) + 1 if len(positions_idxs) > 0 else 0 # We add 1 because range starts from 0
            current_positions_idxs = []

            if not block.holds_embeddings: # If the current block does not hold embeddings, you add indices, otherwise embeddings 
                current_tokens = block.get_tokens()
                tokens         += current_tokens

                for position_idx in range(len(current_tokens)):
                    current_positions_idxs.append(position_idx + position_idx_offset)

                positions_idxs += current_positions_idxs
            else:
                current_embeddings = block.get_embeddings()
                embeddings         += block.get_embeddings()

                for position_idx in range(len(current_embeddings)):
                    current_positions_idxs.append(position_idx + position_idx_offset)

                embeddings_positions_idxs += current_positions_idxs

        if len(tokens) < 2: # If there are not enough tokens, we set everything to None
            tokens     = None
            embeddings = None

            positions_idxs            = None
            embeddings_positions_idxs = None

        return tokens, embeddings, positions_idxs, embeddings_positions_idxs

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

        self.dataset_buffer_size = 100_000 # Avoiding spike losses when streaming=True, might create RAM-related issues
        
        self.dataset     = iter(load_dataset(self.dataset_name, split=self.dataset_split, streaming=True).shuffle(buffer_size=self.dataset_buffer_size, seed=self.seed).skip(self.dataset_skip_num + 1))
        self.dataset_idx = 0

        self.data_stack = []

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

        tokens = self.tokenizer.encode(data)

        chunks = self.sequence_slice(tokens, self.sequence_length)

        for chunk in chunks: # Create sequences from the chunks
            sequence = Sequence(blocks=[
                IndicesBlock(chunk)
            ])
            tokens, embeddings, positions_idxs, embeddings_positions_idxs = sequence.get_data()

            if tokens is not None:
                element = {
                    'tokens': tokens,
                    'embeddings': embeddings,
                    'positions_idxs': positions_idxs,
                    'embeddings_positions_idxs': embeddings_positions_idxs
                }

                self.data_stack.append(element)

    def get_batch(self): # Get a batch for training
        batch_tokens     = []
        batch_embeddings = []

        batch_positions_idxs            = []
        batch_embeddings_positions_idxs = []

        max_length = 0

        while len(batch_tokens) < self.batch_size: # Append examples to the batch until you reach the desired batch size
            if len(self.data_stack) == 0:
                self.ingest()

            element = self.data_stack[0]

            tokens     = element['tokens']
            embeddings = element['embeddings']

            positions_idxs            = element['positions_idxs']
            embeddings_positions_idxs = element['embeddings_positions_idxs']

            max_length = self.sequence_length

            batch_tokens.append(tokens)
            batch_embeddings.append(embeddings)

            batch_positions_idxs.append(positions_idxs)
            batch_embeddings_positions_idxs.append(embeddings_positions_idxs)

            del self.data_stack[0]

        return (
            batch_tokens,
            batch_embeddings,
            batch_positions_idxs,
            batch_embeddings_positions_idxs
        )
