import sys

sys.dont_write_bytecode = True

import time
import unittest

import torch

from utils.stable_adamw import StableAdamWUnfused

from inference import EvalAndromeda

class AndromedaTest(unittest.TestCase):
    def setUp(self):
        self.model_path = 'checkpoints/step_2802_512/pytorch_model.bin'

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.eval  = EvalAndromeda(path=self.model_path, device=self.device)

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

def main():
    unittest.main()

if __name__ == '__main__':
    main()
