from dataclasses import dataclass, field


@dataclass
class BatchMetrics:
    batch_losses: list[float] = field(default_factory=list)
    batch_r1_a2t: list[float] = field(default_factory=list)
    batch_r5_a2t: list[float] = field(default_factory=list)
    batch_r10_a2t: list[float] = field(default_factory=list)
    batch_map_a2t: list[float] = field(default_factory=list)
    batch_r1_t2a: list[float] = field(default_factory=list)
    batch_r5_t2a: list[float] = field(default_factory=list)
    batch_r10_t2a: list[float] = field(default_factory=list)
    batch_map_t2a: list[float] = field(default_factory=list)

    def add_metrics(
            self,
            loss: float,
            r1_a2t: float,
            r5_a2t: float,
            r10_a2t: float,
            map_a2t: float,
            r1_t2a: float,
            r5_t2a: float,
            r10_t2a: float,
            map_t2a: float
    ):
        self.batch_losses.append(loss)
        self.batch_r1_a2t.append(r1_a2t)
        self.batch_r5_a2t.append(r5_a2t)
        self.batch_r10_a2t.append(r10_a2t)
        self.batch_map_a2t.append(map_a2t)
        self.batch_r1_t2a.append(r1_t2a)
        self.batch_r5_t2a.append(r5_t2a)
        self.batch_r10_t2a.append(r10_t2a)
        self.batch_map_t2a.append(map_t2a)

    def compute_average_metrics(self) -> dict[str, float]:
        def __mean(values):
            return sum(values) / len(values) if values else 0

        return {
            "avg_loss": __mean(self.batch_losses),
            "avg_r1_a2t": __mean(self.batch_r1_a2t),
            "avg_r5_a2t": __mean(self.batch_r5_a2t),
            "avg_r10_a2t": __mean(self.batch_r10_a2t),
            "avg_map_a2t": __mean(self.batch_map_a2t),
            "avg_r1_t2a": __mean(self.batch_r1_t2a),
            "avg_r5_t2a": __mean(self.batch_r5_t2a),
            "avg_r10_t2a": __mean(self.batch_r10_t2a),
            "avg_map_t2a": __mean(self.batch_map_t2a)
        }


@dataclass
class EpochMetrics:
    epoch_losses: list[float] = field(default_factory=list)
    epoch_r1_a2t: list[float] = field(default_factory=list)
    epoch_r5_a2t: list[float] = field(default_factory=list)
    epoch_r10_a2t: list[float] = field(default_factory=list)
    epoch_map_a2t: list[float] = field(default_factory=list)
    epoch_r1_t2a: list[float] = field(default_factory=list)
    epoch_r5_t2a: list[float] = field(default_factory=list)
    epoch_r10_t2a: list[float] = field(default_factory=list)
    epoch_map_t2a: list[float] = field(default_factory=list)

    def add_epoch_metrics(self, batch_avg_metrics: dict[str, float]):
        self.epoch_losses.append(batch_avg_metrics["loss"])
        self.epoch_r1_a2t.append(batch_avg_metrics["r1_a2t"])
        self.epoch_r5_a2t.append(batch_avg_metrics["r5_a2t"])
        self.epoch_r10_a2t.append(batch_avg_metrics["r10_a2t"])
        self.epoch_map_a2t.append(batch_avg_metrics["map_a2t"])
        self.epoch_r1_t2a.append(batch_avg_metrics["r1_t2a"])
        self.epoch_r5_t2a.append(batch_avg_metrics["r5_t2a"])
        self.epoch_r10_t2a.append(batch_avg_metrics["r10_t2a"])
        self.epoch_map_t2a.append(batch_avg_metrics["map_t2a"])

    def compute_average_epoch_metrics(self) -> dict[str, float]:
        def __mean(values):
            return sum(values) / len(values) if values else 0

        return {
            "epoch_avg_loss": __mean(self.epoch_losses),
            "epoch_avg_r1_a2t": __mean(self.epoch_r1_a2t),
            "epoch_avg_r5_a2t": __mean(self.epoch_r5_a2t),
            "epoch_avg_r10_a2t": __mean(self.epoch_r10_a2t),
            "epoch_avg_map_a2t": __mean(self.epoch_map_a2t),
            "epoch_avg_r1_t2a": __mean(self.epoch_r1_t2a),
            "epoch_avg_r5_t2a": __mean(self.epoch_r5_t2a),
            "epoch_avg_r10_t2a": __mean(self.epoch_r10_t2a),
            "epoch_avg_map_t2a": __mean(self.epoch_map_t2a)
        }
