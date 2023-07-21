import sys

sys.dont_write_bytecode = True

import time
import unittest

import torch

import torch.nn.functional as F

from nltk.translate.bleu_score import corpus_bleu

from sklearn.metrics import f1_score

from utils.stable_adamw import StableAdamWUnfused

from inference import EvalAndromeda

model_path = 'checkpoints/step_3450_512/pytorch_model.bin'

class AndromedaTest(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model_path = model_path
        self.eval       = EvalAndromeda(path=self.model_path, device=self.device)

        self.tokenizer = self.eval.tokenizer
        self.model     = self.eval.model

        self.optimizer = StableAdamWUnfused(self.model.parameters())

        self.input_tensor = torch.randint(0, 256, (1, 1024), device=self.device).long()

    def test_forward_pass(self):
        # Test if the models forward pass works
        logits = self.model(self.input_tensor, return_loss=False)
        
        self.assertEqual(logits.shape, (1, self.input_tensor.shape[1] - 1, len(self.tokenizer))) # Test if output shape is correct

    def test_backward_pass(self):
        # Test if the models backward pass works correctly
        self.optimizer.zero_grad()

        _, loss = self.model(self.input_tensor, return_loss=True)
        loss.backward()

        for name, parameter in self.model.named_parameters():
            self.assertFalse(torch.isnan(parameter.grad).any(), f'Gradient for {name} contains NaNs')
            self.assertFalse(torch.isinf(parameter.grad).any(), f'Gradient for {name} contains Infs')

    def test_optimizer_step(self):
        # Test if the optimizer steps correctly
        initial_params = [param.clone() for param in self.model.parameters()]

        self.optimizer.zero_grad()
        
        _, loss = self.model(self.input_tensor, return_loss=True)
 
        loss.backward()
        self.optimizer.step()

        for initial_param, param in zip(initial_params, self.model.parameters()):
            self.assertFalse(torch.equal(initial_param, param), 'Model parameters did not change after an optimizer step')

class FlopsBenchmark:
    def __init__(self, batch_size=4, num_heads=8, sequence_lengths=[128, 32, 64, 128, 256]):
        self.batch_size       = batch_size
        self.sequence_lengths = sequence_lengths # The first sequence length should just be there to let torch deal with the model and not fake further results

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model_path = model_path
        self.eval       = EvalAndromeda(path=self.model_path, device=self.device)

        self.tokenizer = self.eval.tokenizer
        self.model     = self.eval.model

        self.model_hidden_dim = self.model.net.emb_dim
        self.model_heads_num  = self.model.net.attn_layers.heads_num

    def benchmark(self):
        times_forward = []
        tflops_per_s  = []

        for seq_len in self.sequence_lengths:
            input_tensor = torch.randint(0, 256, (self.batch_size, seq_len), device=self.device).long()

            torch.cuda.synchronize()

            time_forward_0 = time.time()
            
            output = self.model(input_tensor)
            
            torch.cuda.synchronize()
            
            time_forward_1 = time.time()

            time_forward = time_forward_1 - time_forward_0
            times_forward.append(time_forward)
            
            total_flops = 4 * (seq_len ** 2) * (self.model_hidden_dim // self.model_heads_num) * self.model_heads_num
            tflops_per_s.append((total_flops / time_forward) / 1e12) # Convert to TFLOPs

        for seq_len, elapsed, tflops in zip(self.sequence_lengths, times_forward, tflops_per_s):
            print(f'Sequence length: {seq_len}, Time elapsed: {elapsed} (s), TFLOPs/s: {tflops}')

class AccuracyBenchmark:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model_path = model_path
        self.eval       = EvalAndromeda(path=self.model_path, device=self.device)

        self.tokenizer = self.eval.tokenizer
        self.model     = self.eval.model

    def compute_perplexity(self, data): # We cannot compare this against models with different tokenizers
        total_loss = 0

        with torch.no_grad():
            for text in data:
                tokens = torch.tensor(self.tokenizer.encode(text))
                tokens = torch.unsqueeze(tokens, dim=0)

                tokens = tokens.long()
                tokens = tokens.to(self.device)

                _, loss = self.model(tokens, return_loss=True)
                
                total_loss += loss.item()
        
            perplexity = torch.exp(torch.tensor(total_loss / len(data)))

        return perplexity
    
    def compute_bleu(self, references, hypotheses):
        bleu = corpus_bleu(references, hypotheses)
    
        return bleu
    
    def compute_f1(self, true_labels, pred_labels):
        return f1_score(true_labels, pred_labels, average='weighted')

def main():
    mode = 'benchmark' # 'test'

    if mode == 'test':
        unittest.main()
    elif mode == 'benchmark':
        flops_benchmark = FlopsBenchmark()

        flops_benchmark.benchmark()
        
        accuracy_benchmark = AccuracyBenchmark()

        perplexity = accuracy_benchmark.compute_perplexity(['This is Andromeda!'])
        print(f'Perplexity: {perplexity}')
    else:
        raise Exception(f'Unknown mode {mode}')

if __name__ == '__main__':
    main()
