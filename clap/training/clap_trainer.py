from pathlib import Path

import os

import numpy as np

from tqdm import tqdm

import wandb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from .retrieval_metrics import BatchRetrievalMetrics, EpochRetrievalMetrics
from ..utils import get_target_device, load_clap_ckpt
from ..model import Clap


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
    train_epoch_metrics : EpochRetrievalMetrics
        Metrics computed over training epochs.
    val_epoch_metrics : EpochRetrievalMetrics
        Metrics computed over validation epochs.
    test_epoch_metrics : EpochRetrievalMetrics
        Metrics computed over test epochs.
    current_epoch : int
        Current epoch number.
    distill_from : list[Clap] (default: None)
            A list of Clap models to distill from when calculating the loss.
    distill_weight : float (default: 0.0)
        The weight of the distillation loss during training.

    Methods
    -------
    from_ckpt(cls, audio_encoder, text_encoder, cfg_version, ckpt_version, optimizer, scheduler, train_loader, val_loader, test_loader, epochs)
        Creates an instance of ClapTrainer from a checkpoint file.
    compute_recall_at_k(similarity, k)
        Compute Recall@K for a given similarity matrix.
    compute_average_precision(scores, labels)
        Compute Average Precision (AP) for a single query.
    compute_mean_average_precision(similarity, k=None)
        Compute mean Average Precision (mAP) or mean Average Precision at k (mAP@k).
    compute_loss(similarity, text_embeddings, audio_embeddings)
        Compute Loss for a similarity matrix and embeddings.
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
            scheduler: LRScheduler,
            enable_wandb_logging: bool = False,
            distill_from: list[Clap] = None,
            distill_weight: float = 0.0
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
        enable_wandb_logging : bool (default: False)
            Whether to enable wandb logging.
        distill_from : list[Clap] (default: None)
            A list of Clap models to distill from when calculating the loss.
        distill_weight : float (default: 0.0)
            The weight of the distillation loss during training.
        """

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.enable_wandb_logging = enable_wandb_logging
        self.train_epoch_metrics = EpochRetrievalMetrics()
        self.val_epoch_metrics = EpochRetrievalMetrics()
        self.test_epoch_metrics = EpochRetrievalMetrics()
        self.current_epoch = 0
        self.epochs = epochs
        self.distill_from = distill_from
        self.distill_weight = distill_weight
        self._device = get_target_device()
        if self.enable_wandb_logging:
            self._global_train_step = 0
            self._global_val_step = 0
            self._global_test_step = 0

    def train_and_eval(
            self,
            audio_encoder: str,
            text_encoder: str,
            version: str | int,
            early_stopping: bool = False
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
        """Optimizes a given model for a number of epochs and saves the best model.

        The function computes both the defined loss and metrics like Recall@K and mAP@K for A-T and T-A respectively.
        Depending on the best loss on the validation data, the best model is then saved to the specified file.
        Moreover, wandb is utilized in order to monitor the training process.
        Finally, a scheduling of the learning rate is utilized as well.

        Parameters
        ----------
        audio_encoder : str
            The audio encoder used for encoding the audio. This will be used for creating a checkpoint file.
        text_encoder : str
            The text encoder used for encoding the text. This will be used for creating a checkpoint file.
        version : str | int
            The version used for creating a checkpoint file.
        early_stopping: bool = False
            Bool used to specify if early stopping should be applied.

        Returns
        -------
        tuple[dict[str, float], dict[str, float], dict[str, float]]
            Dictionaries for the train, validation and test average metrics computed during training and/or evaluation.
        """
        ckpt_path = Path(
            __file__).parent.parent / "checkpoints" / f"clap_{audio_encoder}_{text_encoder}_v{version}.ckpt"
        os.makedirs(ckpt_path.parent, exist_ok=True)

        if self.enable_wandb_logging:
            # Tell wandb to watch the model.
            wandb.watch(self.model, criterion=self.loss_fn, log="all", log_freq=10)

        print("\nStarting to train Model")
        for _ in range(self.current_epoch, self.epochs):

            # Get epoch metrics
            train_metrics = self.train_model()
            val_metrics = self.eval_model()

            # Update epoch metrics
            self.train_epoch_metrics.update(train_metrics)
            self.val_epoch_metrics.update(val_metrics)

            if self.enable_wandb_logging:
                wandb.log(
                    {
                        "train/loss": train_metrics["avg_loss"],
                        "train/a2t/recall@1": train_metrics["avg_r1_a2t"],
                        "train/a2t/recall@5": train_metrics["avg_r5_a2t"],
                        "train/a2t/recall@10": train_metrics["avg_r10_a2t"],
                        "train/a2t/mAP@10": train_metrics["avg_map10_a2t"],
                        "train/a2t/mAP": train_metrics["avg_map_a2t"],
                        "train/t2a/recall@1": train_metrics["avg_r1_t2a"],
                        "train/t2a/recall@5": train_metrics["avg_r5_t2a"],
                        "train/t2a/recall@10": train_metrics["avg_r10_t2a"],
                        "train/t2a/mAP@10": train_metrics["avg_map10_t2a"],
                        "train/t2a/mAP": train_metrics["avg_map_t2a"],
                        "val/loss": val_metrics["avg_loss"],
                        "val/a2t/recall@1": val_metrics["avg_r1_a2t"],
                        "val/a2t/recall@5": val_metrics["avg_r5_a2t"],
                        "val/a2t/recall@10": val_metrics["avg_r10_a2t"],
                        "val/a2t/mAP@10": val_metrics["avg_map10_a2t"],
                        "val/a2t/mAP": val_metrics["avg_map_a2t"],
                        "val/t2a/recall@1": val_metrics["avg_r1_t2a"],
                        "val/t2a/recall@5": val_metrics["avg_r5_t2a"],
                        "val/t2a/recall@10": val_metrics["avg_r10_t2a"],
                        "val/t2a/mAP@10": val_metrics["avg_map10_t2a"],
                        "val/t2a/mAP": val_metrics["avg_map_t2a"],
                        "epoch": self.current_epoch
                    }
                )

            print(
                f"\nEpoch: {str(self.current_epoch).zfill(len(str(self.epochs)))} || "
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
                if np.argmin(self.val_epoch_metrics.epoch_losses) <= self.current_epoch - 5:
                    print(f"\nEarly stopping on epoch {self.current_epoch}!")
                    # Get test metrics and update the epoch metrics
                    test_metrics = self.eval_model(test_set=True)
                    self.test_epoch_metrics.update(test_metrics)

                    if self.enable_wandb_logging:
                        # Necessary to work with model in jupyter notebook after training is done.
                        wandb.unwatch(self.model)

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

            # Update current epoch counter
            self.current_epoch += 1

            # Save the best model
            if np.argmin(self.val_epoch_metrics.epoch_losses) == self.current_epoch-1:
                torch.save({
                    "epoch": self.current_epoch,
                    "model": self.model.state_dict(),
                    "loss_fn": self.loss_fn,
                    "train_metrics": self.train_epoch_metrics,
                    "val_metrics": self.val_epoch_metrics
                }, ckpt_path)
                print(f"\nModel saved to {ckpt_path}")

            print("\n" + 100 * "=")

        # Get test metrics and update the epoch metrics
        test_metrics = self.eval_model(test_set=True)
        self.test_epoch_metrics.update(test_metrics)

        if self.enable_wandb_logging:
            # Necessary to work with model in jupyter notebook after training is done.
            wandb.unwatch(self.model)

        print(
            f"\nFinal loss: {test_metrics['avg_loss']} || "
            f"Final test T-A R@1: {test_metrics['avg_r1_t2a']:.4f} || "
            f"Final test A-T R@1: {test_metrics['avg_r1_a2t']:.4f} || "
            f"Final test T-A mAP: {test_metrics['avg_map_t2a']:.4f} || "
            f"Final test A-T mAP: {test_metrics['avg_map_a2t']:.4f}"
        )
        print("\nDone!")

        return (self.train_epoch_metrics.compute_average_epoch_metrics(),
                self.val_epoch_metrics.compute_average_epoch_metrics(),
                self.test_epoch_metrics.compute_average_epoch_metrics())

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

        batch_metrics = BatchRetrievalMetrics()

        # Compute the loss with torch.no_grad() as gradients aren't used.
        with torch.no_grad():
            for _, caption, audio in tqdm(self.test_loader, desc="Evaluating model on val/test set"):

                # Compute similarity matrix and loss
                text_em, audio_em, _ = self.model(caption, audio)
                similarity = self.model.compute_similarity(text_em, audio_em)

                # Compute loss
                loss = self.compute_loss(similarity, text_em, audio_em)

                # Audio-to-Text retrieval
                r1_a2t, r5_a2t, r10_a2t, map10_a2t, map_a2t = self.compute_retrieval_metrics(similarity)

                # Text-to-Audio retrieval
                r1_t2a, r5_t2a, r10_t2a, map10_t2a, map_t2a = self.compute_retrieval_metrics(similarity.T)

                # Update metrics
                batch_metrics.update(
                    loss=loss.item(),
                    r1_a2t=r1_a2t,
                    r5_a2t=r5_a2t,
                    r10_a2t=r10_a2t,
                    map10_a2t=map10_a2t,
                    map_a2t=map_a2t,
                    r1_t2a=r1_t2a,
                    r5_t2a=r5_t2a,
                    r10_t2a=r10_t2a,
                    map10_t2a=map10_t2a,
                    map_t2a=map_t2a
                )

                if self.enable_wandb_logging:
                    # Log metrics.
                    if test_set:
                        wandb.log(
                            {
                                "test/batch loss": loss.item(),
                                "test/a2t/batch recall@1": r1_a2t,
                                "test/a2t/batch recall@5": r5_a2t,
                                "test/a2t/batch recall@10": r10_a2t,
                                "test/a2t/batch mAP@10": map10_a2t,
                                "test/a2t/batch mAP": map_a2t,
                                "test/t2a/batch recall@1": r1_t2a,
                                "test/t2a/batch recall@5": r5_t2a,
                                "test/t2a/batch recall@10": r10_t2a,
                                "test/t2a/batch mAP@10": map10_t2a,
                                "test/t2a/batch mAP": map_t2a,
                                "test/step": self._global_test_step
                            }
                        )
                        self._global_test_step += 1
                    else:
                        wandb.log(
                            {
                                "val/batch loss": loss.item(),
                                "val/a2t/batch recall@1": r1_a2t,
                                "val/a2t/batch recall@5": r5_a2t,
                                "val/a2t/batch recall@10": r10_a2t,
                                "val/a2t/batch mAP@10": map10_a2t,
                                "val/a2t/batch mAP": map_a2t,
                                "val/t2a/batch recall@1": r1_t2a,
                                "val/t2a/batch recall@5": r5_t2a,
                                "val/t2a/batch recall@10": r10_t2a,
                                "val/t2a/batch mAP@10": map10_t2a,
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

        batch_metrics = BatchRetrievalMetrics()

        for _, caption, audio in tqdm(self.train_loader, desc=f"Training epoch {self.current_epoch}"):
            # Zero the gradients
            self.optimizer.zero_grad()

            # Compute similarity matrix and loss
            text_em, audio_em, _ = self.model(caption, audio)
            similarity = self.model.compute_similarity(text_em, audio_em)

            # Compute loss
            loss = self.compute_loss(similarity, text_em, audio_em)

            # Audio-to-Text retrieval
            r1_a2t, r5_a2t, r10_a2t, map10_a2t, map_a2t = self.compute_retrieval_metrics(similarity)

            # Text-to-Audio retrieval
            r1_t2a, r5_t2a, r10_t2a, map10_t2a, map_t2a = self.compute_retrieval_metrics(similarity.T)

            # Update metrics
            batch_metrics.update(
                loss=loss.item(),
                r1_a2t=r1_a2t,
                r5_a2t=r5_a2t,
                r10_a2t=r10_a2t,
                map10_a2t=map10_a2t,
                map_a2t=map_a2t,
                r1_t2a=r1_t2a,
                r5_t2a=r5_t2a,
                r10_t2a=r10_t2a,
                map10_t2a=map10_t2a,
                map_t2a=map_t2a
            )

            # Compute the gradients and perform the optimizer and scheduler step.
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            if self.enable_wandb_logging:
                # Log metrics.
                wandb.log(
                    {
                        "train/batch loss": loss.item(),
                        "train/a2t/batch recall@1": r1_a2t,
                        "train/a2t/batch recall@5": r5_a2t,
                        "train/a2t/batch recall@10": r10_a2t,
                        "train/a2t/batch mAP@10": map10_a2t,
                        "train/a2t/batch mAP": map_a2t,
                        "train/t2a/batch recall@1": r1_t2a,
                        "train/t2a/batch recall@5": r5_t2a,
                        "train/t2a/batch recall@10": r10_t2a,
                        "train/t2a/batch mAP@10": map10_t2a,
                        "train/t2a/batch mAP": map_t2a,
                        "train/step": self._global_train_step
                    }
                )

                self._global_train_step += 1

        return batch_metrics.compute_average_metrics()

    def compute_loss(
            self,
            similarity: torch.Tensor,
            text_embeddings: torch.Tensor,
            audio_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Computes the loss for a given similarity metrix and audio and text embeddings."""
        # Get contrastive loss
        if (1 - self.distill_weight) > 0:
            audio_target = torch.arange(similarity.shape[0]).to(similarity.device)
            text_target = torch.arange(similarity.shape[0]).to(similarity.device)
            contrastive_loss = self.loss_fn(similarity, audio_target, text_target)
        else:
            contrastive_loss = 0

        # Get distilled loss
        if self.distill_from is not None and self.distill_weight > 0:
            target = torch.mean(
                torch.stack(
                    [distill_model.compute_similarity(text_embeddings, audio_embeddings) for distill_model in
                     self.distill_from]),
                dim=0
            ).to(self._device)

            # Compute audio and text target probabilities
            audio_target = target.softmax(dim=1)
            text_target = target.T.softmax(dim=1)

            distilled_loss = self.loss_fn(similarity, audio_target, text_target)
        else:
            distilled_loss = 0

        loss = (1 - self.distill_weight) * contrastive_loss + self.distill_weight * distilled_loss

        return loss

    @classmethod
    def from_ckpt(
            cls,
            audio_encoder: str,
            text_encoder: str,
            cfg_version: str | int,
            ckpt_version: str | int,
            optimizer: Optimizer,
            scheduler: LRScheduler,
            train_loader: DataLoader,
            val_loader: DataLoader,
            test_loader: DataLoader,
            epochs: int,
            enable_wandb_logging: bool = False,
            distill_from: list[Clap] = None,
            distill_weight: float = 0.0
    ) -> "ClapTrainer":
        """Create an instance of ClapTrainer from a checkpoint file (e.g., a ckpt file).

        This method loads a Clap model and its associated training state from a checkpoint file.
        It initializes the ClapTrainer with the loaded model, optimizer, scheduler, data loaders,
        and other training parameters.

        Parameters
        ----------
        audio_encoder : str
            The name of the audio encoder to use.
        text_encoder : str
            The name of the text encoder to use.
        cfg_version : str | int
            The version of the config file to load.
        ckpt_version : str or int
            The version of the checkpoint to load.
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
        enable_wandb_logging : bool (default: False)
            Whether to enable wandb logging.
        distill_from : list[Clap] (default: None)
            A list of Clap models to distill from when calculating the loss.
        distill_weight : float (default: 0.0)
            The weight of the distillation loss during training.

        Returns
        -------
        ClapTrainer
            An instance of the ClapTrainer class initialized with the checkpoint.

        Raises
        ------
        FileNotFoundError
            If the checkpoint file is not found.
        KeyError
            If the checkpoint file is missing required keys.

        Examples
        --------
        **Example 1: Initializing with encoders and version**

        >>> audio_encoder = "cnn14"
        >>> text_encoder = "distilroberta-base"
        >>> cfg_version = 1
        >>> ckpt_version = 1
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        >>> scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        >>> train_loader = torch.utils.data.DataLoader(...)
        >>> val_loader = torch.utils.data.DataLoader(...)
        >>> test_loader = torch.utils.data.DataLoader(...)
        >>> epochs = 100
        >>> trainer = ClapTrainer.from_ckpt(
        ...     audio_encoder=audio_encoder,
        ...     text_encoder=text_encoder,
        ...     cfg_version=cfg_version,
        ...     ckpt_version=ckpt_version,
        ...     optimizer=optimizer,
        ...     scheduler=scheduler,
        ...     train_loader=train_loader,
        ...     val_loader=val_loader,
        ...     test_loader=test_loader,
        ...     epochs=epochs,
        ...     enable_wandb_logging=False
        ... )

        **Example 2: Handling errors**

        Ensure the checkpoint file path is correct and the file is properly formatted:

        >>> try:
        ...     trainer = ClapTrainer.from_ckpt(
        ...         audio_encoder="invalid_audio_encoder",
        ...         text_encoder="invalid_text_encoder",
        ...         cfg_version=999,
        ...         ckpt_version=999,
        ...         optimizer=optimizer,
        ...         scheduler=scheduler,
        ...         train_loader=train_loader,
        ...         val_loader=val_loader,
        ...         test_loader=test_loader,
        ...         epochs=epochs,
        ...         enable_wandb_logging=False
        ...     )
        ... except FileNotFoundError:
        ...     print("Checkpoint file not found.")
        ... except KeyError as e:
        ...     print(f"Checkpoint file is missing required key: {e}")

        Notes
        -----
        - The method `load_clap_ckpt` is responsible for finding and loading the checkpoint file.
        - The method `Clap.from_ckpt` is used to load the Clap model from the checkpoint.
        """
        # Load model with specified encoder and version
        model = Clap.from_ckpt(
            audio_encoder=audio_encoder,
            text_encoder=text_encoder,
            cfg_version=cfg_version,
            ckpt_version=ckpt_version
        ).to(get_target_device())

        # Load metrics and other hyperparameters
        ckpt = load_clap_ckpt(audio_encoder=audio_encoder, text_encoder=text_encoder, version=ckpt_version)
        epoch = ckpt["epoch"]
        train_metrics = ckpt["train_metrics"]
        val_metrics = ckpt["val_metrics"]
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
            epochs=epochs,
            enable_wandb_logging=enable_wandb_logging,
            distill_from=distill_from,
            distill_weight=distill_weight
        )

        # Set metrics and current epoch
        trainer.train_epoch_metrics = train_metrics
        trainer.val_epoch_metrics = val_metrics
        trainer.current_epoch = epoch

        return trainer

    # TODO: Fix metrics for clotho!
    # TODO: Do not evaluate the metrics during training!
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
            # Get the indices of the top K similarities for each query
            top_k_indices = torch.topk(similarity[i], k).indices
            # Check if the correct pair is within the top K
            if i in top_k_indices:
                correct += 1

        recall_at_k = correct / N
        return recall_at_k

    @staticmethod
    def compute_average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
        """ Compute Average Precision (AP) for a single query."""
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
    def compute_retrieval_metrics(similarity: torch.Tensor):
        """Compute retrieval metrics for a given similarity matrix."""
        r1 = ClapTrainer.compute_recall_at_k(similarity, k=1)
        r5 = ClapTrainer.compute_recall_at_k(similarity, k=5)
        r10 = ClapTrainer.compute_recall_at_k(similarity, k=10)
        map10 = ClapTrainer.compute_mean_average_precision(similarity, k=10)
        map = ClapTrainer.compute_mean_average_precision(similarity)

        return r1, r5, r10, map, map10
