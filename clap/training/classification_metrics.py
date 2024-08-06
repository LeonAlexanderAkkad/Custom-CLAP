from dataclasses import dataclass, field


@dataclass
class BatchClassificationMetrics:
    """Simple Dataclass for storing batch classification metrics."""
    batch_losses: list[float] = field(default_factory=list)
    batch_accuracy: list[float] = field(default_factory=list)

    def update(
            self,
            loss: float = 0,
            accuracy: float = 0,
    ):
        self.batch_losses.append(loss)
        self.batch_accuracy.append(accuracy)

    def compute_average_metrics(self) -> dict[str, float]:
        def __mean(values):
            return sum(values) / len(values) if values else 0

        return {
            "avg_loss": __mean(self.batch_losses),
            "avg_accuracy": __mean(self.batch_accuracy)
        }


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
