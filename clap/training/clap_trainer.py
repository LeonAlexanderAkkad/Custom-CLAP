import random
import time
from pathlib import Path

import numpy as np

from tqdm import tqdm

import wandb as wb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

import os

from .metrics import BatchMetrics, EpochMetrics
from ..utils import get_target_device
from ..model import Clap


# TODO: Start training
# TODO: Implement new evaluation Dataset for downstream tasks
# TODO: Change relative imports in __init__ files
# TODO: Update docstrings and readme as well as provide metric and examples

class ClapTrainer:
    """A trainer class for optimizing and evaluating the Clap model.

    This class handles the training, evaluation, and checkpoint loading for the Clap model.

    Attributes
    ----------
    train_loader : DataLoader
        DataLoader for the training dataset.
    val_loader : DataLoader
        DataLoader for the validation dataset.
    test_loader : DataLoader
        DataLoader for the test dataset.
    model : Clap
        The Clap model to be trained.
    optimizer : Optimizer
        The optimizer used for training.
    loss_fn : nn.Module
        The loss function used during training.
    epochs : int
        The total number of training epochs.
    scheduler : LRScheduler
        The learning rate scheduler used during training.
    train_epoch_metrics : EpochMetrics
        Metrics computed over training epochs.
    val_epoch_metrics : EpochMetrics
        Metrics computed over validation epochs.
    test_epoch_metrics : EpochMetrics
        Metrics computed over test epochs.
    current_epoch : int
        Current epoch number.

    Methods
    -------
    from_ckpt(cls, ckpt, optimizer, scheduler, train_loader, val_loader, test_loader, epochs)
        Creates an instance of ClapTrainer from a checkpoint file.
    compute_recall_at_k(similarity, k)
        Compute Recall@K for a given similarity matrix.
    compute_average_precision(scores, labels)
        Compute Average Precision (AP) for a single query.
    compute_mean_average_precision(similarity, k=None)
        Compute mean Average Precision (mAP) or mean Average Precision at k (mAP@k).
    set_random_seed(seed=None)
        Set the random seed for reproducibility.
    get_lr(optimizer)
        Get the learning rate used for optimizing.
    train_and_eval(out_path, early_stopping=False)
        Optimizes a given model for a number of epochs and saves the best model.
    eval_model(test_set=False)
        Evaluates a given model on valuation or test data.
    train_model()
        Trains a given model on the training data.
    """
    def __init__(
            self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            test_loader: DataLoader,
            model: Clap,
            optimizer: Optimizer,
            loss_fn: nn.Module,
            epochs: int,
            scheduler: LRScheduler
    ):
        """Initializes a ClapTrainer instance for training, validating, and testing a Clap model.

        Parameters
        ----------
        train_loader : DataLoader
            DataLoader instance for loading training data in batches.
        val_loader : DataLoader
            DataLoader instance for loading validation data in batches.
        test_loader : DataLoader
            DataLoader instance for loading test data in batches.
        model : Clap
            The Clap model to be trained and evaluated.
        optimizer : Optimizer
            The optimizer used for updating the model parameters.
        loss_fn : nn.Module
            The loss function used to compute the loss during training and evaluation.
        epochs : int
            The number of epochs to train the model.
        scheduler : LRScheduler
            Learning rate scheduler to adjust the learning rate during training.
        """

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.train_epoch_metrics = EpochMetrics()
        self.val_epoch_metrics = EpochMetrics()
        self.test_epoch_metrics = EpochMetrics()
        self.current_epoch = 0
        self.epochs = epochs
        self._lr = self.get_lr(self.optimizer)
        self._device = get_target_device()
        self._global_train_step = 0
        self._global_val_step = 0
        self._global_test_step = 0

    def train_and_eval(
            self,
            out_path: str,
            early_stopping: bool = False
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
        """Optimizes a given model for a number of epochs and saves the best model.

        The function computes both the defined loss and metrics like Recall@K and mAP for A-T and T-A respectively.
        Depending on the best loss on the validation data, the best model is then saved to the specified file.
        Moreover, wandb is utilized in order to monitor the training process.
        Finally, a scheduling of the learning rate is utilized as well.

        Parameters
        ----------
        out_path: str
            Path to the file where the results and best model should be stored.
        early_stopping: bool = False
            Bool used to specify if early stopping should be applied.

        Returns
        -------
        tuple[dict[str, float], dict[str, float], dict[str, float]]
            Dictionaries for the train, validation and test average metrics computed during training and/or evaluation.
        """

        best_loss = None
        # Tell wandb to watch the model.
        wb.watch(self.model, criterion=self.loss_fn, log="all", log_freq=10)

        print("\nStarting to train Model")
        for epoch in range(self.epochs):

            # Get epoch metrics
            train_metrics = self.train_model()
            val_metrics = self.eval_model()

            # Update epoch metrics
            self.train_epoch_metrics.update(train_metrics)
            self.val_epoch_metrics.update(val_metrics)

            # Update scheduler
            self.scheduler.step()

            wb.log(
                {
                    "train/loss": train_metrics["avg_loss"],
                    "train/a2t/recall@1": train_metrics["avg_r1_a2t"],
                    "train/a2t/recall@5": train_metrics["avg_r5_a2t"],
                    "train/a2t/recall@10": train_metrics["avg_r10_a2t"],
                    "train/a2t/mAP": train_metrics["avg_map_a2t"],
                    "train/t2a/recall@1": train_metrics["avg_r1_t2a"],
                    "train/t2a/recall@5": train_metrics["avg_r5_t2a"],
                    "train/t2a/recall@10": train_metrics["avg_r10_t2a"],
                    "train/t2a/mAP": train_metrics["avg_map_t2a"],
                    "val/loss": val_metrics["avg_loss"],
                    "val/a2t/recall@1": val_metrics["avg_r1_a2t"],
                    "val/a2t/recall@5": val_metrics["avg_r5_a2t"],
                    "val/a2t/recall@10": val_metrics["avg_r10_a2t"],
                    "val/a2t/mAP": val_metrics["avg_map_a2t"],
                    "val/t2a/recall@1": val_metrics["avg_r1_t2a"],
                    "val/t2a/recall@5": val_metrics["avg_r5_t2a"],
                    "val/t2a/recall@10": val_metrics["avg_r10_t2a"],
                    "val/t2a/mAP": val_metrics["avg_map_t2a"],
                    "epoch": self.current_epoch
                }
            )

            print(
                f"\nEpoch: {str(self.current_epoch + 1).zfill(len(str(self.epochs)))} (lr={self._lr}) || "
                f"Training loss: {train_metrics['avg_loss']:.4f} || "
                f"Validation loss: {val_metrics['avg_loss']:.4f} || "
                f"Training A-T R@1: {train_metrics['avg_r1_a2t']:.4f} || "
                f"Validation A-T R@1: {val_metrics['avg_r1_a2t']:.4f} || "
                f"Training A-T mAp: {train_metrics['avg_map_a2t']:.4f} || "
                f"Validation A-T mAP: {val_metrics['avg_map_a2t']:.4f} || "
                f"Training T-A R@1: {train_metrics['avg_r1_t2a']:.4f} || "
                f"Validation T-A R@1: {val_metrics['avg_r1_t2a']:.4f} || "
                f"Training T-A mAP: {train_metrics['avg_map_t2a']:.4f} || "
                f"Validation T-A mAP: {val_metrics['avg_map_t2a']:.4f}"
            )

            # Check for early stopping.
            if early_stopping:
                if np.argmin(self.val_epoch_metrics.epoch_losses) <= epoch - 5:
                    print(f"\nEarly stopping on epoch {epoch}!")
                    # Get test metrics and update the epoch metrics
                    test_metrics = self.eval_model(test_set=True)
                    self.test_epoch_metrics.update(test_metrics)

                    print(
                        f"\nFinal loss: {test_metrics['avg_loss']} || "
                        f"Final test T-A R@1: {test_metrics['avg_r1_t2a']:.4f} || "
                        f"Final test A-T R@1: {test_metrics['avg_r1_a2t']:.4f} || "
                        f"Final test T-A mAP: {test_metrics['avg_map_t2a']:.4f} || "
                        f"Final test A-T mAP: {test_metrics['avg_map_a2t']:.4f}"
                    )
                    print("\nDone!")

                    return (self.train_epoch_metrics.compute_average_epoch_metrics(),
                            self.test_epoch_metrics.compute_average_epoch_metrics(),
                            self.val_epoch_metrics.compute_average_epoch_metrics())

            # Save the best model
            if not best_loss or val_metrics["avg_loss"] < best_loss:
                best_loss = val_metrics["avg_loss"]
                torch.save({
                    "epoch": self.current_epoch,
                    "model": self.model.state_dict(),
                    "loss_fn": self.loss_fn,
                    "train_metrics": self.train_epoch_metrics,
                    "val_metrics": self.val_epoch_metrics
                }, out_path)
                print(f"\nModel saved to {out_path}")

            print("\n" + 100 * "=")

        # Get test metrics and update the epoch metrics
        test_metrics = self.eval_model(test_set=True)
        self.test_epoch_metrics.update(test_metrics)

        # Necessary to work with model in jupyter notebook after training is done.
        wb.unwatch(self.model)

        print(
            f"\nFinal loss: {test_metrics['avg_loss']} || "
            f"Final test T-A R@1: {test_metrics['avg_r1_t2a']:.4f} || "
            f"Final test A-T R@1: {test_metrics['avg_r1_a2t']:.4f} || "
            f"Final test T-A mAP: {test_metrics['avg_map_t2a']:.4f} || "
            f"Final test A-T mAP: {test_metrics['avg_map_a2t']:.4f}"
        )
        print("\nDone!")

        return (self.train_epoch_metrics.compute_average_epoch_metrics(),
                self.test_epoch_metrics.compute_average_epoch_metrics(),
                self.val_epoch_metrics.compute_average_epoch_metrics())

    def eval_model(self, test_set: bool = False) -> dict[str, float]:
        """Evaluates a given model on test data.

        Parameters
        ----------
        test_set: bool = False
            Bool used to decide whether to log the model predictions as test or validation performance.

        Returns
        -------
        dict[str, float]
            A dictionary containing all metrics computed during evaluation.
        """

        # Turn on evaluation mode for the model.
        self.model.eval()

        batch_metrics = BatchMetrics()

        # Compute the loss with torch.no_grad() as gradients aren't used.
        with torch.no_grad():
            for _, caption, audio in tqdm(self.test_loader, desc="Evaluating model on val/test set"):

                # Compute similarity matrix and loss
                text_em, audio_em, _ = self.model(caption, audio)
                similarity = self.model.compute_similarity(text_em, audio_em)
                loss = self.loss_fn(similarity)

                # Audio-to-Text retrieval
                r1_a2t = self.compute_recall_at_k(similarity, k=1)
                r5_a2t = self.compute_recall_at_k(similarity, k=5)
                r10_a2t = self.compute_recall_at_k(similarity, k=10)
                map_a2t = self.compute_mean_average_precision(similarity)

                # Text-to-Audio retrieval
                r1_t2a = self.compute_recall_at_k(similarity.T, k=1)
                r5_t2a = self.compute_recall_at_k(similarity.T, k=5)
                r10_t2a = self.compute_recall_at_k(similarity.T, k=10)
                map_t2a = self.compute_mean_average_precision(similarity.T)

                # Log metrics
                batch_metrics.update(loss=loss.item(), r1_a2t=r1_a2t, r5_a2t=r5_a2t, r10_a2t=r10_a2t, map_a2t=map_a2t,
                                     r1_t2a=r1_t2a, r5_t2a=r5_t2a, r10_t2a=r10_t2a, map_t2a=map_t2a)

                # Log batch loss and accuracy as well as predictions.
                if test_set:
                    wb.log(
                        {
                            "test/batch loss": loss.item(),
                            "test/a2t/batch recall@1": r1_a2t,
                            "test/a2t/batch recall@5": r5_a2t,
                            "test/a2t/batch recall@10": r10_a2t,
                            "test/a2t/batch mAP": map_a2t,
                            "test/t2a/batch recall@1": r1_t2a,
                            "test/t2a/batch recall@5": r5_t2a,
                            "test/t2a/batch recall@10": r10_t2a,
                            "test/t2a/batch mAP": map_t2a,
                            "test/step": self._global_test_step
                        }
                    )
                    self._global_test_step += 1
                else:
                    wb.log(
                        {
                            "val/batch loss": loss.item(),
                            "val/a2t/batch recall@1": r1_a2t,
                            "val/a2t/batch recall@5": r5_a2t,
                            "val/a2t/batch recall@10": r10_a2t,
                            "val/a2t/batch mAP": map_a2t,
                            "val/t2a/batch recall@1": r1_t2a,
                            "val/t2a/batch recall@5": r5_t2a,
                            "val/t2a/batch recall@10": r10_t2a,
                            "val/t2a/batch mAP": map_t2a,
                            "val/step": self._global_val_step
                        }
                    )
                    self._global_val_step += 1

        return batch_metrics.compute_average_metrics()

    def train_model(self) -> dict[str, float]:
        """Trains a given model on the training data.

        Returns
        -------
        dict[str, float]
            A dictionary containing all metrics computed during training.
        """

        # Put the model into train mode and enable gradients computation.
        self.model.train()
        torch.enable_grad()

        batch_metrics = BatchMetrics()

        lr = ClapTrainer.get_lr(self.optimizer)

        for _, caption, audio in tqdm(self.train_loader, desc=f"Training epoch {self.current_epoch + 1} ({lr=})"):
            # Compute similarity matrix and loss
            text_em, audio_em, _ = self.model(caption, audio)
            similarity = self.model.compute_similarity(text_em, audio_em)
            loss = self.loss_fn(similarity)

            # Audio-to-Text retrieval
            r1_a2t = self.compute_recall_at_k(similarity, k=1)
            r5_a2t = self.compute_recall_at_k(similarity, k=5)
            r10_a2t = self.compute_recall_at_k(similarity, k=10)
            map_a2t = self.compute_mean_average_precision(similarity)

            # Text-to-Audio retrieval
            r1_t2a = self.compute_recall_at_k(similarity.T, k=1)
            r5_t2a = self.compute_recall_at_k(similarity.T, k=5)
            r10_t2a = self.compute_recall_at_k(similarity.T, k=10)
            map_t2a = self.compute_mean_average_precision(similarity.T)

            # Log metrics
            batch_metrics.update(loss=loss.item(), r1_a2t=r1_a2t, r5_a2t=r5_a2t, r10_a2t=r10_a2t, map_a2t=map_a2t,
                                 r1_t2a=r1_t2a, r5_t2a=r5_t2a, r10_t2a=r10_t2a, map_t2a=map_t2a)

            # Compute the gradients.
            loss.backward()
            # Perform the update.
            self.optimizer.step()
            # Reset the accumulated gradients.
            self.optimizer.zero_grad()
            # Log batch loss and accuracy.
            wb.log(
                {
                    "train/batch loss": loss.item(),
                    "train/a2t/batch recall@1": r1_a2t,
                    "train/a2t/batch recall@5": r5_a2t,
                    "train/a2t/batch recall@10": r10_a2t,
                    "train/a2t/batch mAP": map_a2t,
                    "train/t2a/batch recall@1": r1_t2a,
                    "train/t2a/batch recall@5": r5_t2a,
                    "train/t2a/batch recall@10": r10_t2a,
                    "train/t2a/batch mAP": map_t2a,
                    "train/step": self._global_train_step
                }
            )

            self._global_train_step += 1

        self.current_epoch += 1

        return batch_metrics.compute_average_metrics()

    @classmethod
    def from_ckpt(
            cls,
            ckpt: Path | str,
            optimizer: Optimizer,
            scheduler: LRScheduler,
            train_loader: DataLoader,
            val_loader: DataLoader,
            test_loader: DataLoader,
            epochs: int
    ) -> "ClapTrainer":
        """Create an instance of ClapTrainer from a checkpoint file (e.g., a ckpt file).

        This method loads a Clap model and its associated training state from a checkpoint file.
        It initializes the ClapTrainer with the loaded model, optimizer, scheduler, data loaders,
        and other training parameters.

        Parameters
        ----------
        ckpt : Path or str
            The path to the checkpoint file.
        optimizer : Optimizer
            The optimizer to use for training.
        scheduler : LRScheduler
            The learning rate scheduler to use for training.
        train_loader : DataLoader
            The data loader for the training dataset.
        val_loader : DataLoader
            The data loader for the validation dataset.
        test_loader : DataLoader
            The data loader for the test dataset.
        epochs : int
            The total number of training epochs.

        Returns
        -------
        ClapTrainer
            An instance of the ClapTrainer class initialized with the checkpoint.

        Examples
        --------
        **Example 1: Initializing with a Path object**

        >>> ckpt_path = Path("path/to/checkpoint.ckpt")
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        >>> scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        >>> train_loader = torch.utils.data.DataLoader()
        >>> val_loader = torch.utils.data.DataLoader()
        >>> test_loader = torch.utils.data.DataLoader()
        >>> epochs = 100
        >>> trainer = ClapTrainer.from_ckpt(
        ...     ckpt=ckpt_path,
        ...     optimizer=optimizer,
        ...     scheduler=scheduler,
        ...     train_loader=train_loader,
        ...     val_loader=val_loader,
        ...     test_loader=test_loader,
        ...     epochs=epochs
        ... )

        **Example 3: Handling errors**

        Ensure the checkpoint file path is correct and the file is properly formatted:

        >>> try:
        ...     trainer = ClapTrainer.from_ckpt(
        ...         ckpt="invalid/path/to/checkpoint.ckpt",
        ...         optimizer=optimizer,
        ...         scheduler=scheduler,
        ...         train_loader=train_loader,
        ...         val_loader=val_loader,
        ...         test_loader=test_loader,
        ...         epochs=epochs
        ...     )
        ... except FileNotFoundError:
        ...     print("Checkpoint file not found.")
        ... except KeyError as e:
        ...     print(f"Checkpoint file is missing required key: {e}")

        """
        # Load model with specified encoder and version
        audio_encoder = ckpt.split("_")[1]
        text_encoder = ckpt.split("_")[2]
        version = ckpt.split("_")[3][:2]
        model = Clap.from_ckpt(audio_encoder=audio_encoder, text_encoder=text_encoder, version=version)

        # Load metrics and other hyperparameters
        ckpt = torch.load(ckpt)
        epoch = ckpt["epoch"]
        train_metrics = ckpt["train_epoch_metrics"]
        val_metrics = ckpt["val_epoch_metrics"]
        loss_fn = ckpt["loss_fn"]

        # Initialize trainer
        trainer = cls(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            epochs=epochs
        )

        # Set metrics and current epoch
        trainer.train_epoch_metrics = train_metrics
        trainer.val_epoch_metrics = val_metrics
        trainer.current_epoch = epoch

        return trainer

    @staticmethod
    def compute_recall_at_k(similarity: torch.Tensor, k: int) -> float:
        """Compute Recall@K for a given similarity matrix.

        Parameters
        ----------
        similarity : torch.Tensor,
            Tensor of shape (N, N) where N is the number of samples.
        k : int
            The value of K for Recall@K.

        Returns
        -------
        float
            The Recall@K for the A-T or T-A retrieval respectively.
        """

        N = similarity.shape[0]
        correct = 0

        for i in range(N):
            # Get the indices of the top K similarities for each text
            top_k_indices = torch.topk(similarity[i], k).indices
            # Check if the correct audio is within the top K
            if i in top_k_indices:
                correct += 1

        recall_at_k = correct / N
        return recall_at_k

    @staticmethod
    def compute_average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
        """
        Compute Average Precision (AP) for a single query.
        """
        sorted_indices = np.argsort(-scores)
        sorted_labels = labels[sorted_indices]

        # Compute precision for all k
        cumulative_sum = np.cumsum(sorted_labels, 0)
        precision_at_k = cumulative_sum / (np.arange(len(sorted_labels)) + 1)

        relevant_indices = np.where(sorted_labels == 1)[0]
        # Prevent 0 division
        if len(relevant_indices) == 0:
            return 0

        # Compute average precision
        ap = np.sum(precision_at_k[relevant_indices]) / len(relevant_indices)

        return ap

    @staticmethod
    def compute_mean_average_precision(similarity: torch.Tensor, k: int = None):
        """
        Compute mean Average Precision (mAP) or mean Average Precision at k (mAP@k).
        """
        num_queries = similarity.shape[0]
        similarity = similarity.cpu().detach().numpy()

        # Create ground truth identity matrix
        ground_truth = np.eye(num_queries)

        aps = []
        for i in range(num_queries):
            scores = similarity[i]
            labels = ground_truth[i]

            # Compute top_k map if specified
            if k is not None:
                sorted_indices = np.argsort(-scores)
                top_k_indices = sorted_indices[:k]
                scores = scores[top_k_indices]
                labels = labels[top_k_indices]

            # Compute average precision
            ap = ClapTrainer.compute_average_precision(scores, labels)
            aps.append(ap)

        # Compute mean average precision
        map_score = np.mean(aps)

        return map_score

    @staticmethod
    def set_random_seed(seed: int | None = 42) -> int:
        """Decide if behaviour is random or set by a seed."""
        if seed is None:
            seed = time.time_ns() % (2 ** 32)

        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # When running on the CuDNN backend, two further options must be set
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Set a fixed value for the hash seed
        os.environ["PYTHONHASHSEED"] = str(seed)
        print(f"Random seed set as {seed}")

        return seed

    @staticmethod
    def get_lr(optimizer):
        """Get the learning rate used for optimizing."""
        return np.array([param_group['lr'] for param_group in optimizer.param_groups])
