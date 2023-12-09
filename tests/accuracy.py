import time
import tracemalloc
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# from andromeda.model import Andromeda
from andromeda.model import Andromeda
from andromeda.utils.stable_adamw import StableAdamWUnfused

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AccuracyMetrics:
    def __init__(self):
        self.rouge = Rouge()

    def calculate_perplexity(self, model, data_loader):
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in data_loader:
                input_ids, labels = batch
                output = model(input_ids)
                loss = F.cross_entropy(
                    output.view(-1, output.size(-1)), labels.view(-1)
                )
                total_loss += loss.item()
        return torch.exp(torch.tensor(total_loss / len(data_loader)))

    def calculate_bleu(self, references, hypotheses):
        return corpus_bleu(references, hypotheses)

    def calculate_rouge(self, references, hypotheses):
        scores = self.rouge.get_scores(hypotheses, references, avg=True)
        return scores

    def calculate_f1(self, true_labels, pred_labels):
        return f1_score(true_labels, pred_labels, average="weighted")


# mock test dataset
test_dataset = datasets.FakeData(size=1000, transform=transforms.ToTensor())

# model
model = Andromeda(
    num_tokens=50304, dim=1024, depth=24, dim_head=128, heads=8, alibi_num_heads=4
)


# Usage:
accuracy_metrics = AccuracyMetrics()

# Calculate Perplexity
perplexity = accuracy_metrics.calculate_perplexity(model, data_loader)
print("Perplexity:", perplexity)

# Calculate BLEU
bleu = accuracy_metrics.calculate_bleu(references, hypotheses)
print("BLEU Score:", bleu)

# Calculate ROUGE
rouge_scores = accuracy_metrics.calculate_rouge(references, hypotheses)
print("ROUGE Scores:", rouge_scores)

# Calculate F1 Score
f1 = accuracy_metrics.calculate_f1(true_labels, pred_labels)
print("F1 Score:", f1)


# Add at the bottom of your file
if __name__ == "__main__":
    AccuracyMetrics()
