import random
import time

import numpy as np

from tqdm import tqdm

import wandb as wb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer

from typing import Optional, Tuple

import os

from .utils import get_target_device
from .model import Clap


class ClapTrainer:
    def __init__(
            self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            test_loader: DataLoader,
            model: Clap,
            optimizer: Optimizer,
            loss_fun: nn.Module,
            epochs: int
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.model = model
        self.optimizer = optimizer
        self.loss_fun = loss_fun
        self._lr = ClapTrainer.get_lr(self.optimizer)
        self._epochs = epochs
        self._device = get_target_device()
        self._global_train_step = 0
        self._global_val_step = 0
        self._global_test_step = 0
        self._current_epoch = 0

    @property
    def epochs(self):
        return self._epochs

    @epochs.setter
    def epochs(self, new_epoch):
        if isinstance(new_epoch, int) and new_epoch > 0:
            self._epochs = new_epoch
            return

        print("Please enter a valid epoch")

    def train_and_eval(
            self,
            out_path: str,
            adapt_lr_factor: Optional[float] = None,
            early_stopping: bool = False
    ) -> Tuple[float, float, float, float, float, float, float]:
        """Optimizes a given model for a number of epochs and saves the best model.

        The function computes both the defined loss and the accuracy between the output of the model and the given target.
        Depending on the best loss on the validation data, the best model is then saved to the specified file.
        Moreover, wandb is utilized in order to monitor the training process.
        Finally, a scheduling of the learning rate is implemented as well.

        Parameters
        ----------
        out_path: str
            Path to the file where the results and best model should be stored.
        adapt_lr_factor: float = None
            Factor used to adapt the learning rate if the model starts to over-fit on the training data.
        early_stopping: bool = False
            Bool used to specify if early stopping should be applied.

        Returns
        -------
        Tuple[float, float, float, float, float, float, float]
            A tuple containing the final test loss and Recall@K.
        """

        best_loss = 0
        # Tell wandb to watch the model.
        wb.watch(self.model, criterion=self.loss_fun, log="all", log_freq=10)
        train_losses = []
        train_accuracies = []
        validation_losses = []
        validation_accuracies = []
        print("\nStarting to train Model")
        for epoch in range(self.epochs):

            train_loss, train_r1_t2a, train_r5_t2a, train_r10_t2a, train_r1_a2t, train_r5_a2t, train_r10_a2t = self.train_model()
            val_loss, val_r1_t2a, val_r5_t2a, val_r10_t2a, val_r1_a2t, val_r5_a2t, val_r10_a2t = self.eval_model()

            # train_losses.append(train_loss)
            # train_accuracies.append(train_acc)
            # validation_losses.append(val_loss)
            # validation_accuracies.append(val_acc)

            wb.log(
                {
                    "train/loss": train_loss,
                    "train/t2a/recall@1": train_r1_t2a,
                    "train/t2a/recall@5": train_r5_t2a,
                    "train/t2a/recall@10": train_r10_t2a,
                    "train/a2t/recall@1": train_r1_a2t,
                    "train/a2t/recall@5": train_r5_a2t,
                    "train/a2t/recall@10": train_r10_a2t,
                    "val/loss": val_loss,
                    "val/t2a/recall@1": val_r1_t2a,
                    "val/t2a/recall@5": val_r5_t2a,
                    "val/t2a/recall@10": val_r10_t2a,
                    "val/a2t/recall@1": val_r1_a2t,
                    "val/a2t/recall@5": val_r5_a2t,
                    "val/a2t/recall@10": val_r10_a2t,
                    "epoch": self._current_epoch
                }
            )

            print(f"\nEpoch: {str(self._current_epoch + 1).zfill(len(str(self.epochs)))} (lr={self._lr}) || "
                  f"Validation loss: {val_loss:.4f} || "
                  f"Validation T-A R@1: {val_r1_t2a:.4f} || "
                  f"Validation A-T R@1: {val_r1_a2t:.4f} || "
                  f"Training loss: {train_loss:.4f} || "
                  f"Training T-A R@1: {train_r1_t2a:.4f} || "
                  f"Training A-T R@1: {train_r1_a2t:.4f}")

            # Check for early stopping.
            if early_stopping:
                if np.argmin(validation_losses) <= epoch - 5:
                    print(f"\nEarly stopping on epoch {epoch}!")
                    test_loss, test_r1_t2a, test_r5_t2a, test_r10_t2a, test_r1_a2t, test_r5_a2t, test_r10_a2t = self.eval_model(save_predictions=True)
                    print(f"\nFinal loss: {test_loss} || Final test T-A R@1: {test_r1_t2a:.4f} || Final test A-T R@1: {test_r1_a2t:.4f}")
                    print("\nDone!")

                    return test_loss, test_r1_t2a, test_r5_t2a, test_r10_t2a, test_r1_a2t, test_r5_a2t, test_r10_a2t

            # Either save the best model or adapt the learning rate if necessary.
            if not best_loss or val_loss < best_loss:
                best_loss = val_loss
                torch.save({"epoch": self._current_epoch,
                            "model": self.model.state_dict(),
                            "optimizer": self.optimizer.state_dict(),
                            "best_val_loss": best_loss}, out_path)
                print(f"\nModel saved to {out_path}")
            else:
                if adapt_lr_factor is not None:
                    self._lr = self._lr / adapt_lr_factor
                    for param_group, lr in zip(self.optimizer.param_groups, self._lr):
                        param_group["lr"] = lr
                    print(f"\nNew learning rate: {self._lr}")

            print("\n" + 100 * "=")

        test_loss, test_r1_t2a, test_r5_t2a, test_r10_t2a, test_r1_a2t, test_r5_a2t, test_r10_a2t = self.eval_model(save_predictions=True)

        # Necessary to work with model in jupyter notebook after training is done.
        wb.unwatch(self.model)

        print(f"\nFinal loss: {test_loss} || Final test T-A R@1: {test_r1_t2a:.4f} || Final test A-T R@1: {test_r1_a2t:.4f}")
        print("\nDone!")

        return test_loss, test_r1_t2a, test_r5_t2a, test_r10_t2a, test_r1_a2t, test_r5_a2t, test_r10_a2t

    def eval_model(self, save_predictions: bool = False) -> tuple[float, float, float, float, float, float, float]:
        """Evaluates a given model on test data.

        Parameters
        ----------
        save_predictions: bool = False
            Bool used to decide whether to log the model predictions or not.

        Returns
        -------
        float, float, float, float, float, float, float
            The specified average loss and average Recall@K for Audio to Text and vice versa.
        """

        # Turn on evaluation mode for the model.
        self.model.eval()

        total_loss = []

        total_r1_t2a = []
        total_r5_t2a = []
        total_r10_t2a = []

        total_r1_a2t = []
        total_r5_a2t = []
        total_r10_a2t = []
        # test_table = self.create_table() if save_predictions else None

        # Compute the loss with torch.no_grad() as gradients aren't used.
        with torch.no_grad():
            for _, caption, audio in tqdm(self.test_loader, desc="Evaluating model on val/test set"):

                # Compute similarity matrix and loss
                text_em, audio_em, _ = self.model(caption, audio)
                similarity = self.model.compute_similarity(text_em, audio_em)
                loss = self.loss_fun(similarity)

                # Text-to-Audio retrieval
                r1_t2a = self.compute_recall_at_k(similarity, k=1)
                r5_t2a = self.compute_recall_at_k(similarity, k=5)
                r10_t2a = self.compute_recall_at_k(similarity, k=10)
                total_r1_t2a.append(r1_t2a)
                total_r5_t2a.append(r5_t2a)
                total_r10_t2a.append(r10_t2a)

                # Audio-to-Text retrieval
                r1_a2t = self.compute_recall_at_k(similarity.T, k=1)
                r5_a2t = self.compute_recall_at_k(similarity.T, k=5)
                r10_a2t = self.compute_recall_at_k(similarity.T, k=10)
                total_r1_a2t.append(r1_a2t)
                total_r5_a2t.append(r5_a2t)
                total_r10_a2t.append(r10_a2t)

                # Log batch loss and accuracy as well as predictions.
                if save_predictions:
                    # self.log_pred_target(test_table, idx, output, target)
                    wb.log(
                        {
                            "test/batch loss": loss.item(),
                            "test/t2a/batch recall@1": r1_t2a,
                            "test/t2a/batch recall@5": r5_t2a,
                            "test/t2a/batch recall@10": r10_t2a,
                            "test/a2t/batch recall@1": r1_a2t,
                            "test/a2t/batch recall@5": r5_a2t,
                            "test/a2t/batch recall@10": r10_a2t,
                            "test/step": self._global_test_step
                        }
                    )
                    self._global_test_step += 1
                else:
                    wb.log(
                        {
                            "val/batch loss": loss.item(),
                            "val/t2a/batch recall@1": r1_t2a,
                            "val/t2a/batch recall@5": r5_t2a,
                            "val/t2a/batch recall@10": r10_t2a,
                            "val/a2t/batch recall@1": r1_a2t,
                            "val/a2t/batch recall@5": r5_a2t,
                            "val/a2t/batch recall@10": r10_a2t,
                            "val/step": self._global_val_step
                        }
                    )
                    self._global_val_step += 1

                # Compute total loss.
                total_loss.append(loss.item())

            # log final table
            # if save_predictions:
                # wb.log({"test/predictions": test_table})

        return (np.mean(np.array(total_loss)).item(),
                np.mean(np.array(total_r1_t2a)).item(),
                np.mean(np.array(total_r5_t2a)).item(),
                np.mean(np.array(total_r10_t2a)).item(),
                np.mean(np.array(total_r1_a2t)).item(),
                np.mean(np.array(total_r5_a2t)).item(),
                np.mean(np.array(total_r10_a2t)).item())

    def train_model(self) -> tuple[float, float, float, float, float, float, float]:
        """Trains a given model on the training data.

        Returns
        -------
        float, float, float, float, float, float, float
            The specified average loss and average Recall@K for Audio to Text and vice versa.
        """

        # Put the model into train mode and enable gradients computation.
        self.model.train()
        torch.enable_grad()

        total_loss = []

        total_r1_t2a = []
        total_r5_t2a = []
        total_r10_t2a = []

        total_r1_a2t = []
        total_r5_a2t = []
        total_r10_a2t = []

        lr = ClapTrainer.get_lr(self.optimizer)

        for _, caption, audio in tqdm(self.train_loader, desc=f"Training epoch {self._current_epoch + 1} ({lr=})"):

            # Compute similarity matrix and loss
            text_em, audio_em, _ = self.model(caption, audio)
            similarity = self.model.compute_similarity(text_em, audio_em)
            loss = self.loss_fun(similarity)

            # Text-to-Audio retrieval
            r1_t2a = self.compute_recall_at_k(similarity, k=1)
            r5_t2a = self.compute_recall_at_k(similarity, k=5)
            r10_t2a = self.compute_recall_at_k(similarity, k=10)
            total_r1_t2a.append(r1_t2a)
            total_r5_t2a.append(r5_t2a)
            total_r10_t2a.append(r10_t2a)

            # Audio-to-Text retrieval
            r1_a2t = self.compute_recall_at_k(similarity.T, k=1)
            r5_a2t = self.compute_recall_at_k(similarity.T, k=5)
            r10_a2t = self.compute_recall_at_k(similarity.T, k=10)
            total_r1_a2t.append(r1_a2t)
            total_r5_a2t.append(r5_a2t)
            total_r10_a2t.append(r10_a2t)

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
                    "train/t2a/batch recall@1": r1_t2a,
                    "train/t2a/batch recall@5": r5_t2a,
                    "train/t2a/batch recall@10": r10_t2a,
                    "train/a2t/batch recall@1": r1_a2t,
                    "train/a2t/batch recall@5": r5_a2t,
                    "train/a2t/batch recall@10": r10_a2t,
                    "train/step": self._global_train_step
                }
            )

            self._global_train_step += 1
            # Compute the total loss.
            total_loss.append(loss.item())

        self._current_epoch += 1

        return (np.mean(np.array(total_loss)).item(),
                np.mean(np.array(total_r1_t2a)).item(),
                np.mean(np.array(total_r5_t2a)).item(),
                np.mean(np.array(total_r10_t2a)).item(),
                np.mean(np.array(total_r1_a2t)).item(),
                np.mean(np.array(total_r5_a2t)).item(),
                np.mean(np.array(total_r10_a2t)).item())

    # TODO: implement abstract methods

    # @abstractmethod
    # def log_pred_target(self, test_table: wb.Table, idx: torch.Tensor, output: torch.Tensor, target: torch.Tensor):
    #     """Log the predictions and the respective target to the test table."""
    #     pass
    #
    # @abstractmethod
    # def create_table(self) -> wb.Table:
    #     """Creates a table to log predictions"""
    #     pass

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

        # For all samples in the batch
        for i in range(N):
            # Get the indices of the top K similarities for each text
            top_k_indices = torch.topk(similarity[i], k).indices
            # Check if the correct audio is within the top K
            if i in top_k_indices:
                correct += 1

        recall_at_k = correct / N
        return recall_at_k

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
