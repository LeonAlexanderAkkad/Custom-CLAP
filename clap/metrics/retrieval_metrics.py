from dataclasses import dataclass, field

import numpy as np

import torch


@dataclass
class BatchRetrievalMetrics:
    batch_losses: list[float] = field(default_factory=list)
    batch_r1_a2t: list[float] = field(default_factory=list)
    batch_r5_a2t: list[float] = field(default_factory=list)
    batch_r10_a2t: list[float] = field(default_factory=list)
    batch_map10_a2t: list[float] = field(default_factory=list)
    batch_r1_t2a: list[float] = field(default_factory=list)
    batch_r5_t2a: list[float] = field(default_factory=list)
    batch_r10_t2a: list[float] = field(default_factory=list)
    batch_map10_t2a: list[float] = field(default_factory=list)

    def update(
            self,
            loss: float = None,
            r1_a2t: float = None,
            r5_a2t: float = None,
            r10_a2t: float = None,
            map10_a2t: float = None,
            r1_t2a: float = None,
            r5_t2a: float = None,
            r10_t2a: float = None,
            map10_t2a: float = None
    ):
        if loss is not None:
            self.batch_losses.append(loss)
        if r1_a2t is not None:
            self.batch_r1_a2t.append(r1_a2t)
        if r5_a2t is not None:
            self.batch_r5_a2t.append(r5_a2t)
        if r10_a2t is not None:
            self.batch_r10_a2t.append(r10_a2t)
        if map10_a2t is not None:
            self.batch_map10_a2t.append(map10_a2t)
        if r1_t2a is not None:
            self.batch_r1_t2a.append(r1_t2a)
        if r5_t2a is not None:
            self.batch_r5_t2a.append(r5_t2a)
        if r10_t2a is not None:
            self.batch_r10_t2a.append(r10_t2a)
        if map10_t2a is not None:
            self.batch_map10_t2a.append(map10_t2a)

    def compute_average_metrics(self) -> dict[str, float]:
        def __mean(values):
            return sum(values) / len(values) if values else 0

        return {
            "avg_loss": __mean(self.batch_losses),
            "avg_r1_a2t": __mean(self.batch_r1_a2t),
            "avg_r5_a2t": __mean(self.batch_r5_a2t),
            "avg_r10_a2t": __mean(self.batch_r10_a2t),
            "avg_map10_a2t": __mean(self.batch_map10_a2t),
            "avg_r1_t2a": __mean(self.batch_r1_t2a),
            "avg_r5_t2a": __mean(self.batch_r5_t2a),
            "avg_r10_t2a": __mean(self.batch_r10_t2a),
            "avg_map10_t2a": __mean(self.batch_map10_t2a)
        }

    @staticmethod
    def compute_a2t_metrics(similarity: torch.Tensor):
        """Computes A2T retrieval metrics for a similarity matrix containing the similarity between audios and text."""
        num_audios = similarity.shape[0]
        sorted_sim = torch.argsort(similarity, descending=True)

        ranks = np.zeros(num_audios)
        ap10 = np.zeros(num_audios)
        for i in range(num_audios):
            # Initialize the range for the current index (5 consecutive elements)
            current_idx = np.arange(5 * i, 5 * i + 5)

            # Get the ranks of the current indices in the sorted similarities
            ranks_sim = np.where(np.isin(sorted_sim[i], current_idx))[0]

            # Find the minimum rank for the current index
            rank = ranks_sim.min() if ranks_sim.size > 0 else 1e20

            # Get ranks that are less than 10 and adjust their values for sim_map
            sim_ap = ranks_sim[ranks_sim < 10] + 1

            # Compute mAP@10 for the current index
            if len(sim_ap) > 0:
                ap10[i] = np.sum(np.arange(1, len(sim_ap) + 1) / sim_ap) / 5
            else:
                ap10[i] = 0.0

            # Store the rank
            ranks[i] = rank

        # Compute metrics
        r1 = len(np.where(ranks < 1)[0]) / len(ranks)
        r5 = len(np.where(ranks < 5)[0]) / len(ranks)
        r10 = len(np.where(ranks < 10)[0]) / len(ranks)
        map10 = np.sum(ap10) / len(ranks)

        return r1, r5, r10, map10

    @staticmethod
    def compute_t2a_metrics(similarity: torch.Tensor):
        """Computes T2A retrieval metrics for a similarity matrix containing the similarity between audios and text."""
        num_audios = similarity.shape[1]
        sorted_sim = torch.argsort(similarity, descending=True)

        ranks = np.zeros(5 * num_audios)
        for audio_idx in range(num_audios):
            for i in range(5):
                ranks[5 * audio_idx + i] = np.where(sorted_sim[5 * audio_idx + i] == audio_idx)[0][0]

        # Compute metrics
        r1 = len(np.where(ranks < 1)[0]) / len(ranks)
        r5 = len(np.where(ranks < 5)[0]) / len(ranks)
        r10 = len(np.where(ranks < 10)[0]) / len(ranks)
        map10 = np.sum(1 / (ranks[np.where(ranks < 10)[0]] + 1)) / len(ranks)

        return r1, r5, r10, map10


@dataclass
class EpochRetrievalMetrics:
    epoch_losses: list[float] = field(default_factory=list)
    epoch_r1_a2t: list[float] = field(default_factory=list)
    epoch_r5_a2t: list[float] = field(default_factory=list)
    epoch_r10_a2t: list[float] = field(default_factory=list)
    epoch_map10_a2t: list[float] = field(default_factory=list)
    epoch_r1_t2a: list[float] = field(default_factory=list)
    epoch_r5_t2a: list[float] = field(default_factory=list)
    epoch_r10_t2a: list[float] = field(default_factory=list)
    epoch_map10_t2a: list[float] = field(default_factory=list)

    def update(self, batch_avg_metrics: dict[str, float]):
        self.epoch_losses.append(batch_avg_metrics["avg_loss"])
        self.epoch_r1_a2t.append(batch_avg_metrics["avg_r1_a2t"])
        self.epoch_r5_a2t.append(batch_avg_metrics["avg_r5_a2t"])
        self.epoch_r10_a2t.append(batch_avg_metrics["avg_r10_a2t"])
        self.epoch_map10_a2t.append(batch_avg_metrics["avg_map10_a2t"])
        self.epoch_r1_t2a.append(batch_avg_metrics["avg_r1_t2a"])
        self.epoch_r5_t2a.append(batch_avg_metrics["avg_r5_t2a"])
        self.epoch_r10_t2a.append(batch_avg_metrics["avg_r10_t2a"])
        self.epoch_map10_t2a.append(batch_avg_metrics["avg_map10_t2a"])

    def compute_average_epoch_metrics(self) -> dict[str, float]:
        def __mean(values):
            return sum(values) / len(values) if values else 0

        return {
            "epoch_avg_loss": __mean(self.epoch_losses),
            "epoch_avg_r1_a2t": __mean(self.epoch_r1_a2t),
            "epoch_avg_r5_a2t": __mean(self.epoch_r5_a2t),
            "epoch_avg_r10_a2t": __mean(self.epoch_r10_a2t),
            "epoch_avg_map10_a2t": __mean(self.epoch_map10_a2t),
            "epoch_avg_r1_t2a": __mean(self.epoch_r1_t2a),
            "epoch_avg_r5_t2a": __mean(self.epoch_r5_t2a),
            "epoch_avg_r10_t2a": __mean(self.epoch_r10_t2a),
            "epoch_avg_map10_t2a": __mean(self.epoch_map10_t2a)
        }
