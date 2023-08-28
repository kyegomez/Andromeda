import matplotlib.pyplot as plt
import time
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import tracemalloc

# from andromeda.model import Andromeda
from andromeda.model import Andromeda
from andromeda.utils.stable_adamw import StableAdamWUnfused

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AndromedaModelTest:
    def __init__(self):
        self.model = Andromeda
        self.optimizer = StableAdamWUnfused()
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.test_input = torch.randint(0, 256, (1, 1024)).cuda()

    def test_forward_pass(self):
        output = self.model(self.test_input)
        assert output.shape == (1, 1024, 64007), "Forward pass output shape mismatch"

    def test_backward_pass(self):
        self.optimizer.zero_grad()
        output = self.model(self.test_input)
        loss = self.loss_function(output, self.test_input)

        loss.backward()
        for name, parameter in self.model.named_parameters():
            assert not torch.isnan(parameter.grad().any()), f"Gradient for {name} contains NaNs"
            assert not torch.isinf(parameter.grad().any()), f"Gradient for {name} contains Infs"


    def test_optimizer_step(self):
        initial_params = [param.clone() for param in self.model_parameters()]
        output = self.model(self.test_input)
        loss = self.loss_function(output, self.test_input)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        for initial_param, param in zip(initial_params, self.model.parameters()):
            assert not torch.equal(initial_param, param), "Model Parameters did not change after an optimizer step"





class SpeedMetrics:
    def __init__(self, model):
        self.model = model.to(device)

    def forward_pass_time(self):
        start_time = time.time()
        self.model.decoder.forward(torch.randint(0, 50304, (1, 8192), device=device, dtype=torch.long))[0]
        end_time = time.time()
        return end_time - start_time
    
    def backward_pass_time(self):
        model_input = self.model.decoder.forward(torch.randint(0, 50304, (1, 8192), device=device, dtype=torch.long))[0]
        start_time = time.time()
        loss = torch.nn.CrossEntropyLoss()(model_input, torch.randint(0, 50304, (1, 8192), device=device, dtype=torch.long))
        loss.backward()
        end_time = time.time()
        return end_time - start_time
    
    def end_to_end_latency(self):
        start_time = time.time()
        self.model.forward(torch.randint(0, 50304, (1, 8192), device=device, dtype=torch.long))
        end_time = time.time()
        return end_time - start_time
    


class ScalabilityMetrics:
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset
        self.dataloader = DataLoader(dataset, batch_size=32)

    def throughput(self):
        start_time = time.time()
        for i, data in enumerate(self.dataloader, 0):
            self.model.forward(data)
        end_time = time.time()
        return len(self.dataset) / (end_time - start_time)


class ConsistencyMetrics:
    def __init__(self, model):
        self.model = model

    def consistency_over_time(self):
        consistency_times = []
        outputs_list = []
        for _ in range(10):
            start_time = time.time()
            outputs = self.model.forward(torch.randint(0, 50304, (1, 8192)))
            end_time = time.time()
            consistency_times.append(end_time - start_time)
            outputs_list.append(outputs.detach().numpy())

        initial_output = outputs_list[0]
        consistency_score = 0
        for output in outputs_list[1:]:
            if np.array_equal(initial_output, output):
                consistency_score += 1
        consistency_score = consistency_score / len(outputs_list) * 100

        return consistency_times, consistency_score


class MemoryMetrics:
    def __init__(self, model):
        self.model = model

    def memory_footprint(self):
        tracemalloc.start()
        self.model.forward(torch.randint(0, 50304, (1, 8192)))
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return current, peak


class SequenceMetrics:
    def __init__(self, model):
        self.model = model

    def sequence_length_impact(self):
        seq_lengths = [1024, 2048, 4096, 8192]
        seq_impact_times = []
        for length in seq_lengths:
            start_time = time.time()
            self.model.forward(torch.randint(0, 50304, (1, length)))
            end_time = time.time()
            seq_impact_times.append(end_time - start_time)
        return seq_lengths, seq_impact_times




class FlopsBenchmark:
    def __init__(self, model, bsz=32, d_model=1024, num_heads=8, sequence_lengths=list(range(500, 32001, 500))):
        self.bsz = bsz
        self.d_model = d_model
        self.num_heads = num_heads
        self.sequence_lengths = sequence_lengths
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype=torch.float32
        self.model = model.to(self.device)

    def benchmark(self):
        time_taken = []
        tflops_per_s = []

        for seq_len in self.sequence_lengths:
            x = torch.randn(self.bsz, seq_len, self.d_model).to(self.device).type(self.dtype)
            torch.cuda.synchronize()

            start = time.time()
            self.model(x)
            torch.cuda.synchronize()
            elapsed = time.time() - start

            time_taken.append(elapsed)
            total_flops = 4 * seq_len **2 * (self.d_model // self.num_heads) * self.num_heads
            tflops_per_s.append(total_flops / elapsed / 1e12)  # Convert to TFLOPs

        for seq_len, elapsed, tflops in zip(self.sequence_lengths, time_taken, tflops_per_s):
            print(f"Sequence length: {seq_len}, Time elapsed: {elapsed} s, TFLOPs/s: {tflops}")


#mock test dataset
test_dataset = datasets.FakeData(size=1000, transform=transforms.ToTensor())

#model
model = Andromeda(
    num_tokens=50304, 
    dim=1024,
    depth=24,
    dim_head=128,
    heads=8,
    alibi_num_heads=4
)


#speed test metrics test 
# speed test metrics test 
speed_metrics = SpeedMetrics(model)
forward_pass_time = speed_metrics.forward_pass_time()
backward_pass_time = speed_metrics.backward_pass_time()
end_to_end_latency = speed_metrics.end_to_end_latency()


#scalability metrics test
scalability_metrics = ScalabilityMetrics(model, test_dataset)
throughput = scalability_metrics.throughput()


#consistency metrucs test
consistency_metrics = ConsistencyMetrics(model)
consistency_times, consistency_score = consistency_metrics.consistency_over_time()


#memory metrics test
memory_metrics = MemoryMetrics(model)
current, peak = memory_metrics.memory_footprint()

#sequence metrics test
sequence_metrics = SequenceMetrics(model)
seq_lengths, seq_impact_times = sequence_metrics.sequence_length_impact()



#flops

flops_benchmark = FlopsBenchmark(model)
flops_benchmark.benchmark()

# Graphical Interface
fig, axs = plt.subplots(3)

axs[0].bar(["Forward Pass Time", "Backward Pass Time", "End-to-End Latency"], [forward_pass_time, backward_pass_time, end_to_end_latency])
axs[0].set_title('Speed Metrics')
axs[0].set_xlabel('Metrics')
axs[0].set_ylabel('Time (seconds)')

axs[1].bar(seq_lengths, seq_impact_times)
axs[1].set_title('Sequence Length Impact')
axs[1].set_xlabel('Sequence Length')
axs[1].set_ylabel('Time (seconds)')

axs[2].plot(list(range(1, 11)), consistency_times)
axs[2].set_title('Consistency Over Time')
axs[2].set_xlabel('Run Number')
axs[2].set_ylabel('Time (seconds)')

plt.tight_layout()
plt.show()

print(f"Throughput: {throughput} instances/second")
print(f"Memory used: {current / 10**6}MB; Peak: {peak / 10**6}MB")



# Add at the bottom of your file
if __name__ == "__main__":
    model_test = AndromedaModelTest()
    model_test.test_forward_pass()
    model_test.test_backward_pass()
    model_test.test_optimizer_step()