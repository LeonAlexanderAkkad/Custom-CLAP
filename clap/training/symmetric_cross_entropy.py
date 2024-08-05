import torch
from torch import nn
import torch.nn.functional as F


class SymmetricCrossEntropyLoss(nn.Module):
    """Symmetric Cross-Entropy Loss for paired data (e.g., text and audio).

    This loss function computes the average of the cross-entropy losses calculated
    along two axes of the similarity matrix: one for text and the other for audio.
    It ensures symmetry in the loss computation, promoting consistency in paired data
    representations.

    Methods
    -------
    forward(similarity: torch.Tensor) -> torch.Tensor
        Computes the symmetric cross-entropy loss given a similarity matrix.

    Example
    -------
    >>> criterion = SymmetricCrossEntropyLoss()
    >>> similarity = torch.randn(32, 32)  # Example similarity matrix
    >>> loss = criterion(similarity)
    >>> print(loss)
    """
    def __init__(self):
        """Initializes the SymmetricCrossEntropyLoss module."""
        super().__init__()

    def forward(self, similarity: torch.Tensor) -> torch.Tensor:
        """Computes the symmetric cross-entropy loss given a similarity matrix.

        Parameters
        ----------
        similarity : torch.Tensor
            A 2D tensor of shape (N, N) representing the similarity scores
            between paired data (e.g., text and audio).

        Returns
        -------
        torch.Tensor
            A scalar tensor representing the symmetric cross-entropy loss.
        """
        # Compute Cross-entropy loss along the text axis
        audio_loss = F.cross_entropy(similarity, torch.arange(similarity.shape[0]).to(similarity.device))

        # Compute Cross-entropy loss along the audio axis
        text_loss = F.cross_entropy(similarity.T, torch.arange(similarity.shape[0]).to(similarity.device))

        # Compute symmetric Cross-entropy loss
        return 0.5 * (text_loss + audio_loss)
