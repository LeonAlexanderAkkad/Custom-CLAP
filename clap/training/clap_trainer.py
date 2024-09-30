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

from ..metrics import BatchRetrievalMetrics, EpochRetrievalMetrics
from ..evaluate import eval_retrieval
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
            ckpt_path: str | Path,
            early_stopping: bool = False
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
        """Optimizes a given model for a number of epochs and saves the best model.

        The function computes both the defined loss and metrics like Recall@K and mAP@K for A-T and T-A respectively.
        Depending on the loss on the validation data, the best model is then saved to the specified file.
        Moreover, wandb is utilized in order to monitor the training process.
        Finally, a scheduling of the learning rate is utilized as well.

        Parameters
        ----------
        ckpt_path : str | Path
            Path to the checkpoint where the model will be saved.
        early_stopping : bool = False
            Bool used to specify if early stopping should be applied.

        Returns
        -------
        tuple[dict[str, float], dict[str, float], dict[str, float]]
            Dictionaries for the train, validation and test average metrics computed during training and/or evaluation.
        """
        os.makedirs(Path(ckpt_path).parent, exist_ok=True)

        if self.enable_wandb_logging:
            # Tell wandb to watch the model.
            wandb.watch(self.model, criterion=self.loss_fn, log="all", log_freq=10)

        print("\nStarting to train Model")
        for _ in range(self.current_epoch, self.epochs):

            # Get epoch metrics
            train_metrics = self.train_model()
            val_metrics = self.eval_model(self.val_loader)

            # Update epoch metrics
            self.train_epoch_metrics.update(train_metrics)
            self.val_epoch_metrics.update(val_metrics)

            if self.enable_wandb_logging:
                wandb.log(
                    {
                        "train/loss": train_metrics["avg_loss"],
                        "val/loss": val_metrics["avg_loss"],
                        "val/a2t/recall@1": val_metrics["avg_r1_a2t"],
                        "val/a2t/recall@5": val_metrics["avg_r5_a2t"],
                        "val/a2t/recall@10": val_metrics["avg_r10_a2t"],
                        "val/a2t/mAP@10": val_metrics["avg_map10_a2t"],
                        "val/t2a/recall@1": val_metrics["avg_r1_t2a"],
                        "val/t2a/recall@5": val_metrics["avg_r5_t2a"],
                        "val/t2a/recall@10": val_metrics["avg_r10_t2a"],
                        "val/t2a/mAP@10": val_metrics["avg_map10_t2a"],
                        "epoch": self.current_epoch
                    }
                )

            print(
                f"\nEpoch: {str(self.current_epoch).zfill(len(str(self.epochs)))} || "
                f"Training loss: {train_metrics['avg_loss']:.4f} || "
                f"Validation loss: {val_metrics['avg_loss']:.4f} || "
                f"Validation A-T R@1: {val_metrics['avg_r1_a2t']:.4f} || "
                f"Validation T-A R@1: {val_metrics['avg_r1_t2a']:.4f} || "
                f"Validation A-T mAP@10: {val_metrics['avg_map10_a2t']:.4f} || "
                f"Validation T-A mAP@10: {val_metrics['avg_map10_t2a']:.4f}"
            )

            # Check for early stopping.
            if early_stopping:
                if np.argmin(self.val_epoch_metrics.epoch_losses) <= self.current_epoch - 5:
                    print(f"\nEarly stopping on epoch {self.current_epoch}!")
                    # Get test metrics and update the epoch metrics
                    test_metrics = self.eval_model(self.test_loader, test_set=True)
                    self.test_epoch_metrics.update(test_metrics)

                    if self.enable_wandb_logging:
                        # Necessary to work with model in jupyter notebook after training is done.
                        wandb.unwatch(self.model)

                    print(
                        f"\nFinal loss: {test_metrics['avg_loss']} || "
                        f"Final test T-A R@1: {test_metrics['avg_r1_t2a']:.4f} || "
                        f"Final test A-T R@1: {test_metrics['avg_r1_a2t']:.4f} || "
                        f"Final test T-A mAP@10: {test_metrics['avg_map10_t2a']:.4f} || "
                        f"Final test A-T mAP@10: {test_metrics['avg_map10_a2t']:.4f}"
                    )
                    print("\nDone!")

                    return (self.train_epoch_metrics.compute_average_epoch_metrics(),
                            self.test_epoch_metrics.compute_average_epoch_metrics(),
                            self.val_epoch_metrics.compute_average_epoch_metrics())

            # Update current epoch counter
            self.current_epoch += 1

            # Save the best model
            if np.argmax(self.val_epoch_metrics.epoch_map10_t2a) == self.current_epoch - 1:
                torch.save({
                    "epoch": self.current_epoch,
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                    "loss_fn": self.loss_fn,
                    "train_metrics": self.train_epoch_metrics,
                    "val_metrics": self.val_epoch_metrics,
                    "val_step": self._global_val_step,
                    "train_step": self._global_train_step
                }, ckpt_path)
                print(f"\nBest model saved to {ckpt_path}")

            print("\n" + 100 * "=")

        # Get test metrics and update the epoch metrics
        test_metrics = self.eval_model(self.test_loader, test_set=True)
        self.test_epoch_metrics.update(test_metrics)

        # Save last model
        current_ckpt_path = ckpt_path.split(".")[0] + f"_epoch{self.current_epoch}.ckpt"
        torch.save({
            "epoch": self.current_epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "loss_fn": self.loss_fn,
            "train_metrics": self.train_epoch_metrics,
            "val_metrics": self.val_epoch_metrics,
            "val_step": self._global_val_step,
            "train_step": self._global_train_step
        }, current_ckpt_path)
        print(f"\nCurrent model saved to {current_ckpt_path}")

        if self.enable_wandb_logging:
            # Necessary to work with model in jupyter notebook after training is done.
            wandb.unwatch(self.model)

        print(
            f"\nFinal loss: {test_metrics['avg_loss']} || "
            f"Final test T-A R@1: {test_metrics['avg_r1_t2a']:.4f} || "
            f"Final test A-T R@1: {test_metrics['avg_r1_a2t']:.4f} || "
            f"Final test T-A mAP@10: {test_metrics['avg_map10_t2a']:.4f} || "
            f"Final test A-T mAP@10: {test_metrics['avg_map10_a2t']:.4f}"
        )
        print("\nDone!")

        return (self.train_epoch_metrics.compute_average_epoch_metrics(),
                self.val_epoch_metrics.compute_average_epoch_metrics(),
                self.test_epoch_metrics.compute_average_epoch_metrics())

    def eval_model(self, eval_loader: DataLoader, test_set: bool = False) -> dict[str, float]:
        """Evaluates a given model on test data.

        Parameters
        ----------
        eval_loader : DataLoader
            The data loader used to evaluate the model.
        test_set : bool = False
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
            for _, caption, audio in tqdm(eval_loader, desc="Evaluating model on val/test set"):

                # Compute similarity matrix and loss
                text_em, audio_em, _ = self.model(caption, audio)
                similarity = self.model.compute_similarity(text_em, audio_em)

                # Compute loss
                loss = self.compute_eval_loss(similarity)

                # Update metrics
                batch_metrics.update(
                    loss=loss.item()
                )

                if self.enable_wandb_logging:
                    # Log metrics.
                    if test_set:
                        wandb.log(
                            {
                                "test/batch loss": loss.item(),
                                "test/step": self._global_test_step
                            }
                        )
                        self._global_test_step += 1
                    else:
                        wandb.log(
                            {
                                "val/batch loss": loss.item(),
                                "val/step": self._global_val_step
                            }
                        )
                        self._global_val_step += 1

            retrieval_metrics = eval_retrieval(self.model, eval_loader)

            batch_metrics.update(
                r1_a2t=retrieval_metrics["avg_r1_a2t"],
                r5_a2t=retrieval_metrics["avg_r5_a2t"],
                r10_a2t=retrieval_metrics["avg_r10_a2t"],
                map10_a2t=retrieval_metrics["avg_map10_a2t"],
                r1_t2a=retrieval_metrics["avg_r1_t2a"],
                r5_t2a=retrieval_metrics["avg_r5_t2a"],
                r10_t2a=retrieval_metrics["avg_r10_t2a"],
                map10_t2a=retrieval_metrics["avg_map10_t2a"],
            )

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
            loss = self.compute_train_loss(similarity, caption, audio)

            # Update metrics
            batch_metrics.update(
                loss=loss.item()
            )

            # Compute the gradients and perform the optimizer and scheduler step.
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # Clamp the logits just like in the paper for stability
            with torch.no_grad():
                self.model.logit_scale.clamp_(min=0, max=np.log(100))

            if self.enable_wandb_logging:
                # Log metrics.
                wandb.log(
                    {
                        "train/batch loss": loss.item(),
                        "train/learning rater": self.optimizer.param_groups[0]["lr"],
                        "train/step": self._global_train_step
                    }
                )

                self._global_train_step += 1

        return batch_metrics.compute_average_metrics()

    def compute_train_loss(
            self,
            similarity: torch.Tensor,
            caption: torch.Tensor,
            audio: torch.Tensor
    ) -> torch.Tensor:
        """Computes the training loss for a given similarity metrix and caption and audio."""
        # Get contrastive loss
        if (1 - self.distill_weight) > 0:
            audio_target = torch.arange(similarity.shape[0]).to(similarity.device)
            text_target = torch.arange(similarity.shape[0]).to(similarity.device)
            contrastive_loss = self.loss_fn(similarity, audio_target, text_target)
        else:
            contrastive_loss = 0

        if self.distill_from is not None and self.distill_weight > 0:
            similarities = []
            for distill_model in self.distill_from:
                text_em, audio_em, _ = distill_model(caption, audio)
                similarities.append(self.model.compute_similarity(text_em, audio_em))

            target = torch.mean(torch.stack(similarities, dim=0), dim=0).to(self._device)

            # Compute audio and text target probabilities
            audio_target = target.softmax(dim=1)
            text_target = target.T.softmax(dim=1)

            distilled_loss = self.loss_fn(similarity, audio_target, text_target)
        else:
            distilled_loss = 0

        loss = (1 - self.distill_weight) * contrastive_loss + self.distill_weight * distilled_loss

        return loss

    def compute_eval_loss(
            self,
            similarity: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the evaluation loss for a given similarity metrix."""
        audio_target = torch.arange(similarity.shape[0]).to(similarity.device)
        text_target = torch.arange(similarity.shape[0]).to(similarity.device)
        contrastive_loss = self.loss_fn(similarity, audio_target, text_target)

        return contrastive_loss

    @classmethod
    def from_ckpt(
            cls,
            config_path: str | Path,
            ckpt_path: str | Path,
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
        config_path : str | Path
            Path to the config file to use for the model.
        ckpt_path : str | Path
            Path to the checkpoint file to use for the model.
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

        >>> config_path = r"path/to/config.yml"
        >>> ckpt_path = r"path/to/ckpt.ckpt"
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        >>> scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        >>> train_loader = torch.utils.data.DataLoader(...)
        >>> val_loader = torch.utils.data.DataLoader(...)
        >>> test_loader = torch.utils.data.DataLoader(...)
        >>> epochs = 100
        >>> trainer = ClapTrainer.from_ckpt(
        ...     config_path = r"path/to/config.yml",
        ...     ckpt_path = r"path/to/ckpt.ckpt",
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
        ...         config_path = r"invalid/path/to/config.yml",
        ...         ckpt_path=r"invalid/path/to/ckpt.ckpt",
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
            config_path=config_path,
            ckpt_path=ckpt_path
        ).to(get_target_device())

        # Load metrics and other hyperparameters
        ckpt = load_clap_ckpt(ckpt_path)
        epoch = ckpt["epoch"]
        train_metrics = ckpt["train_metrics"]
        val_metrics = ckpt["val_metrics"]
        train_step = ckpt["train_step"]
        val_step = ckpt["val_step"]
        loss_fn = ckpt["loss_fn"]
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])

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
        trainer._global_train_step = train_step
        trainer._global_val_step = val_step

        return trainer
