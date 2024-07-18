import torch
from torch import nn


class ContrastiveLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, similarity: torch.Tensor):
        # TODO: Implement the contrastive loss!
        raise NotImplementedError
