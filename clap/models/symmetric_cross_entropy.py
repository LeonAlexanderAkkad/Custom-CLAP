import torch
from torch import nn
import torch.nn.functional as F


class SymmetricCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, similarity: torch.Tensor) -> torch.Tensor:
        # Compute Cross-entropy loss along the text axis
        text_loss = F.cross_entropy(similarity.T, torch.arange(similarity.shape[0])).to(similarity.device)

        # Compute Cross-entropy loss along the audio axis
        audio_loss = F.cross_entropy(similarity, torch.arange(similarity.shape[0])).to(similarity.device)

        # Compute symmetric Cross-entropy loss
        return 0.5 * (text_loss + audio_loss)
