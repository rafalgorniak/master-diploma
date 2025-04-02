import torch.nn as nn
import torch

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super().__init__()
        self.margin = margin

    def forward(self, out1, out2, label):
        # Oblicz odległość euklidesową między embeddingami
        euclidean_distance = torch.nn.functional.pairwise_distance(out1, out2)

        # Oblicz stratę
        loss = torch.mean(
            label * torch.pow(euclidean_distance, 2) +  # Dla tej samej klasy
            (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)  # Dla różnych klas
        )
        return loss