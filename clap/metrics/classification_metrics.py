from dataclasses import dataclass, field

import torch

import numpy as np

from sklearn.metrics import accuracy_score


@dataclass
class BatchClassificationMetrics:
    """Simple Dataclass for storing batch classification metrics."""
    batch_losses: list[float] = field(default_factory=list)
    batch_accuracy: list[float] = field(default_factory=list)

    def update(
            self,
            loss: float = None,
            accuracy: float = None
    ):
        if loss is not None:
            self.batch_losses.append(loss)
        if accuracy is not None:
            self.batch_accuracy.append(accuracy)

    def compute_average_metrics(self) -> dict[str, float]:
        def __mean(values):
            return sum(values) / len(values) if values else 0

        return {
            "avg_loss": __mean(self.batch_losses),
            "avg_accuracy": __mean(self.batch_accuracy)
        }

    @staticmethod
    def compute_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Computes the accuracy of the audio classification task."""
        predictions = predictions.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()
        predictions = np.argmax(predictions, axis=1)
        acc = accuracy_score(targets, predictions)

        return acc


@dataclass
class EpochClassificationMetrics:
    """Simple Dataclass for storing epoch classification metrics."""
    epoch_losses: list[float] = field(default_factory=list)
    epoch_accuracy: list[float] = field(default_factory=list)

    def update(self, batch_avg_metrics: dict[str, float]):
        self.epoch_losses.append(batch_avg_metrics["avg_loss"])
        self.epoch_accuracy.append(batch_avg_metrics["avg_accuracy"])

    def compute_average_epoch_metrics(self) -> dict[str, float]:
        def __mean(values):
            return sum(values) / len(values) if values else 0

        return {
            "epoch_avg_loss": __mean(self.epoch_losses),
            "epoch_avg_accuracy": __mean(self.epoch_accuracy)
        }
