import matplotlib.pyplot as plt
import time
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from memory_profiler import profile

from Andromeda.model import AndromedaClass

# Mock test dataset
test_dataset = datasets.FakeData(
    size=1000,
    transform=transforms.ToTensor(),
)

test_dataloader = DataLoader(test_dataset, batch_size=32)

# Model
model = AndromedaClass()

# Speed Metrics Tests

## Forward Pass Time
start_time = time.time()
model_input = model.decoder.forward_embedding(torch.randint(0, 64006, (1, 8192)))[0]
end_time = time.time()
forward_pass_time = end_time - start_time

## Backward Pass Time
start_time = time.time()
loss = torch.nn.CrossEntropyLoss()(model_input, torch.randint(0, 64006, (1, 8192)))
loss.backward()
end_time = time.time()
backward_pass_time = end_time - start_time

## End-to-End Latency
start_time = time.time()
outputs = model.forward(torch.randint(0, 64006, (1, 8192)))
end_time = time.time()
end_to_end_latency = end_time - start_time

# Scalability Metrics Tests

## Throughput
start_time = time.time()
for i, data in enumerate(test_dataloader, 0):
    outputs = model.forward(data)
end_time = time.time()
throughput = len(test_dataset)/(end_time - start_time)

# Graphical Interface
fig, ax = plt.subplots()

ax.bar(["Forward Pass Time", "Backward Pass Time", "End-to-End Latency"], [forward_pass_time, backward_pass_time, end_to_end_latency])
ax.set_title('Speed Metrics')
ax.set_xlabel('Metrics')
ax.set_ylabel('Time (seconds)')

plt.show()

print(f"Throughput: {throughput} instances/second")
