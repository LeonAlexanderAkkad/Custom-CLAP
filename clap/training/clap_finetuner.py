import os

from pathlib import Path

import numpy as np

from tqdm import tqdm

import wandb as wb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from sklearn.metrics import accuracy_score

from .classification_metrics import BatchClassificationMetrics, EpochClassificationMetrics
from ..utils import get_target_device, load_clf_ckpt
from ..model import ClapAudioClassifier


class ClapFinetuner:
    """A finetuner class for training an audio classifier / fine-tuning a CLAP model.

    This class handles the training, evaluation, and checkpoint loading for the Classifier model.

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

    Methods
    -------
    from_ckpt(cls, audio_encoder, text_encoder, clap_cfg_version, clf_ckpt_version, optimizer, scheduler, train_loader, val_loader, test_loader, epochs)
        Creates an instance of ClapFinetuner from a checkpoint file.
    compute_accuracy(predictions, targets)
    finetune_and_eval(out_path, early_stopping=False)
        Optimizes a given model for a number of epochs and saves the best model.
    eval_model(test_set=False)
        Evaluates a given model on valuation or test data.
    finetune_model()
        Fine-tunes a given model on the training data.
    """
    def __init__(
            self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            test_loader: DataLoader,
            model: ClapAudioClassifier,
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
        model : ClapAudioClassifier
            The ClapAudioClassifier model to be trained and evaluated.
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
        self.train_epoch_metrics = EpochClassificationMetrics()
        self.val_epoch_metrics = EpochClassificationMetrics()
        self.test_epoch_metrics = EpochClassificationMetrics()
        self.current_epoch = 0
        self.epochs = epochs
        self._device = get_target_device()
        self._global_train_step = 0
        self._global_val_step = 0
        self._global_test_step = 0

    def finetune_and_eval(
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
        best_loss = None
        ckpt_path = Path(__file__).parent.parent / "checkpoints" / f"clf_{audio_encoder}_{text_encoder}_v{version}.ckpt"
        os.makedirs(ckpt_path.parent, exist_ok=True)
        # Tell wandb to watch the model.
        wb.watch(self.model, criterion=self.loss_fn, log="all", log_freq=10)

        print("\nStarting to finetune Classifier")
        for _ in range(self.current_epoch, self.epochs):

            # Get epoch metrics
            train_metrics = self.finetune_model()
            val_metrics = self.eval_model()

            # Update epoch metrics
            self.train_epoch_metrics.update(train_metrics)
            self.val_epoch_metrics.update(val_metrics)

            wb.log(
                {
                    "train/loss": train_metrics["avg_loss"],
                    "train/accuracy": train_metrics["avg_accuracy"],
                    "val/loss": val_metrics["avg_loss"],
                    "val/accuracy": val_metrics["avg_accuracy"],
                    "epoch": self.current_epoch
                }
            )

            print(
                f"\nEpoch: {str(self.current_epoch).zfill(len(str(self.epochs)))} || "
                f"Training loss: {train_metrics['avg_loss']:.4f} || "
                f"Validation loss: {val_metrics['avg_loss']:.4f} || "
                f"Training accuracy: {train_metrics['avg_accuracy']:.4f} || "
                f"Validation accuracy: {val_metrics['avg_accuracy']:.4f}"
            )

            # Check for early stopping.
            if early_stopping:
                if np.argmin(self.val_epoch_metrics.epoch_losses) <= self.current_epoch - 5:
                    print(f"\nEarly stopping on epoch {self.current_epoch}!")
                    # Get test metrics and update the epoch metrics
                    test_metrics = self.eval_model(test_set=True)
                    self.test_epoch_metrics.update(test_metrics)

                    print(
                        f"\nFinal loss: {test_metrics['avg_loss']} || "
                        f"Final test accuracy: {test_metrics['avg_accuracy']:.4f}"
                    )
                    print("\nDone!")

                    return (self.train_epoch_metrics.compute_average_epoch_metrics(),
                            self.test_epoch_metrics.compute_average_epoch_metrics(),
                            self.val_epoch_metrics.compute_average_epoch_metrics())

            # Update current epoch counter
            self.current_epoch += 1

            # Save the best model
            if not best_loss or val_metrics["avg_loss"] < best_loss:
                best_loss = val_metrics["avg_loss"]
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

        # Necessary to work with model in jupyter notebook after training is done.
        wb.unwatch(self.model)

        print(
            f"\nFinal loss: {test_metrics['avg_loss']} || "
            f"Final test accuracy: {test_metrics['avg_accuracy']:.4f}"
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

        batch_metrics = BatchClassificationMetrics()

        # Compute the loss with torch.no_grad() as gradients aren't used.
        with torch.no_grad():
            for _, target, audio in tqdm(self.test_loader, desc="Evaluating model on val/test set"):

                # Compute predictions and loss
                prediction = self.model(audio)
                loss = self.loss_fn(prediction, target)

                # Compute accuracy
                acc = self.compute_accuracy(prediction, target)

                # Update metrics
                batch_metrics.update(
                    loss=loss.item(),
                    accuracy=acc
                )

                # Log metrics
                batch_metrics.update(
                    loss=loss.item(),
                    accuracy=acc
                )

                # Log batch loss and accuracy as well as predictions.
                if test_set:
                    wb.log(
                        {
                            "test/batch loss": loss.item(),
                            "test/batch accuracy": acc,
                            "test/step": self._global_test_step
                        }
                    )
                    self._global_test_step += 1
                else:
                    wb.log(
                        {
                            "val/batch loss": loss.item(),
                            "val/batch accuracy": acc,
                            "val/step": self._global_val_step
                        }
                    )
                    self._global_val_step += 1

        return batch_metrics.compute_average_metrics()

    def finetune_model(self) -> dict[str, float]:
        """Trains a given model on the training data.

        Returns
        -------
        dict[str, float]
            A dictionary containing all metrics computed during training.
        """
        # Put the model into train mode and enable gradients computation.
        self.model.train()
        torch.enable_grad()

        batch_metrics = BatchClassificationMetrics()

        for _, target, audio in tqdm(self.train_loader, desc=f"Training epoch {self.current_epoch}"):
            # Zero the gradients
            self.optimizer.zero_grad()

            # Compute similarity matrix and loss
            prediction = self.model(audio)
            loss = self.loss_fn(prediction, target)

            # Compute accuracy
            acc = self.compute_accuracy(prediction, target)

            # Log metrics
            batch_metrics.update(
                loss=loss.item(),
                accuracy=acc
            )

            # Compute the gradients and perform the optimizer and scheduler step.
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # Log batch loss and accuracy.
            wb.log(
                {
                    "train/batch loss": loss.item(),
                    "train/batch accuracy": acc,
                    "train/step": self._global_train_step
                }
            )

            self._global_train_step += 1

        return batch_metrics.compute_average_metrics()

    @classmethod
    def from_ckpt(
            cls,
            audio_encoder: str,
            text_encoder: str,
            clap_cfg_version: str | int,
            clf_ckpt_version: str | int,
            optimizer: Optimizer,
            scheduler: LRScheduler,
            train_loader: DataLoader,
            val_loader: DataLoader,
            test_loader: DataLoader,
            epochs: int
    ) -> "ClapFinetuner":
        """Create an instance of ClapFinetuner from a checkpoint file (e.g., a ckpt file).

        This method loads a ClapAudioClassifier model and its associated training state from a checkpoint file.
        It initializes the ClapFinetuner with the loaded model, optimizer, scheduler, data loaders,
        and other training parameters.

        Parameters
        ----------
        audio_encoder : str
            The name of the audio encoder to use.
        text_encoder : str
            The name of the text encoder to use.
        clap_cfg_version : str | int
            The version of the clap config file to load.
        clf_ckpt_version : str or int
            The version of the classifier checkpoint to load.
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
        ClapFinetuner
            An instance of the ClapFinetuner class initialized with the checkpoint.

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
        >>> trainer = ClapFinetuner.from_ckpt(
        ...     audio_encoder=audio_encoder,
        ...     text_encoder=text_encoder,
        ...     clap_cfg_version=cfg_version,
        ...     clf_ckpt_version=ckpt_version,
        ...     optimizer=optimizer,
        ...     scheduler=scheduler,
        ...     train_loader=train_loader,
        ...     val_loader=val_loader,
        ...     test_loader=test_loader,
        ...     epochs=epochs
        ... )

        **Example 2: Handling errors**

        Ensure the checkpoint file path is correct and the file is properly formatted:

        >>> try:
        ...     trainer = ClapFinetuner.from_ckpt(
        ...         audio_encoder="invalid_audio_encoder",
        ...         text_encoder="invalid_text_encoder",
        ...         cfg_version=999,
        ...         ckpt_version=999,
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

        Notes
        -----
        - The method `load_ckpt` is responsible for finding and loading the checkpoint file.
        - The method `Clap.from_ckpt` is used to load the Clap model from the checkpoint.
        """
        # Load model with specified encoder and version
        model = ClapAudioClassifier.from_ckpt(
            audio_encoder=audio_encoder,
            text_encoder=text_encoder,
            clap_cfg_version=clap_cfg_version,
            clf_ckpt_version=clf_ckpt_version
        ).to(get_target_device())

        # Load metrics and other hyperparameters
        ckpt = load_clf_ckpt(audio_encoder=audio_encoder, text_encoder=text_encoder, version=clf_ckpt_version)
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
            epochs=epochs
        )

        # Set metrics and current epoch
        trainer.train_epoch_metrics = train_metrics
        trainer.val_epoch_metrics = val_metrics
        trainer.current_epoch = epoch

        return trainer

    @staticmethod
    def compute_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        predictions = predictions.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()
        predictions = np.argmax(predictions, axis=1)
        acc = accuracy_score(targets, predictions)

        return acc
