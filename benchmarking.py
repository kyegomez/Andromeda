import matplotlib.pyplot as plt
import time
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from memory_profiler import profile
import tracemalloc

from Andromeda.model import AndromedaClass

# from ..Andromeda.model import AndromedaClass



class SpeedMetrics:
    def __init__(self, model):
        self.model = model

    def forward_pass_time(self):
        start_time = time.time()
        model_input = self.model.decoder.forward_embedding(torch.randint(0, 640006, (1, 8192)))[0]
        end_time = time.time()
        return end_time - start_time
    
    def backward_pass_time(self):
        model_input = self.model.decoder.forward_embedding(torch.randint(0, 64006, (1, 8192)))[0]
        start_time = time.time()
        loss = torch.nn.CrossEntropyLoss()(model_input, torch.randint(0, 64006, (1, 8192)))
        loss.backward()
        end_time = time.time()
        return end_time - start_time
    
    def end_to_end_latency(self):
        start_time = time.time()
        self.model.forward(torch.randint(0, 64006, (1, 8192)))
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
            outputs = self.model.forward(torch.randint(0, 64006, (1, 8192)))
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
        self.model.forward(torch.randint(0, 64006, (1, 8192)))
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
            self.model.forward(torch.randint(0, 64006, (1, length)))
            end_time = time.time()
            seq_impact_times.append(end_time - start_time)
        return seq_lengths, seq_impact_times



#mock test dataset
test_dataset = datasets.FakeData(size=1000, transform=transforms.ToTensor())

#model
model = AndromedaClass()

#speed test metrics test 
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