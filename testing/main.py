import unittest
import torch
import time
from Andromeda.model import Andromeda
from Andromeda.utils.stable_adamw import StableAdamWUnfused

class AndromedaTest(unittest.TestCase):

    def setUp(self):
        self.model = Andromeda
        self.optimizer = StableAdamWUnfused()
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.test_input = torch.randint(0, 256, (1, 1024)).cuda()

    def test_forward_pass(self):
        #test if the models forward pass works
        output = self.model(self.input_tensor)
        self.assertEqual(output.shape, (1, 1024, 64007)) # test if output shape is correct
    
    def test_backward_pass(self):
        #test if the models backward pass works correctly
        self.optimizer.zero_grad()
        output = self.model(self.input_tensor)
        loss = self.loss_function(output, self.input_tensor)

        loss.backward()
        for name, parameter in self.model.named_parameters():
            self.assertFalse(torch.isnan(parameter.grad().any(), f"Gradient for {name} contains NaNs"))
            self.assertFalse(torch.isinf(parameter.grad().any(), f"Gradient for {name} contains Infs"))

    def test_optimizer_step(self):
        #test if the optimizer steps correctly
        initial_params = [param.clone() for param in self.model.parameters()]
        output = self.model(self.input_tensor)
        loss = self.loss_function(output, self.input_tensor)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer_step()
        for initial_param, param in zip(initial_params, self.model.parameters()):
            self.assertFalse(torch.equal(initial_param, param), 'Model parameters did not change after an optimizer step')

    # def test_prediction(self):
    #     start_time = time.time()
    #     prediction = self.model(self.test_input)
    #     latency = time.time() - start_time
        
    #     self.assertLess(latency, 1) # test if latency is less than 1 second
    #     self.assertEqual(prediction.shape, (1, 1024, 64007)) # test if output shape si correct


    # def test_memory_consumption(self):
    #     start_mem = torch.cuda.memory_allocated()
    #     prediction = self.model(self.test_input)
    #     end_mem = torch.cuda.memory_allocated()
    #     mem_diff = end_mem - start_mem
    #     self.assertLess(mem_diff, 2 * 1024**3) # memory diff should be less than 2gb

    # def test_throughput(self):
    #     start_time = time.time()
    #     for _ in range(100):
    #         prediction = self.model(self.test_input)
    #     end_time = time.time()
    #     throughput = 100 / (end_time - start_time)
    #     self.assertGreater(throughput, 10) # model should handle atleast at 10 inferences per second


if __name__ == "__main__":
    unittest.main()