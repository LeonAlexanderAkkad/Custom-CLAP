import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm

from sklearn.metrics import accuracy_score

from ..model import Clap


def eval_zero_shot_classification(model: Clap, eval_loader: DataLoader, class_embeddings: torch.Tensor) -> float:
    """Evaluates the zero-shot classification accuracy of a given model.

    Parameters
    ----------
    model : Clap
        The model used to compute audio embeddings and similarities.
    eval_loader : DataLoader
        A DataLoader providing batches of data, where each batch contains audio samples and their corresponding target classes.
    class_embeddings : torch.Tensor
        A tensor representing the embeddings of the classes to compare against.

    Returns
    -------
    float
        The zero-shot classification accuracy of the model.
    """
    # Set the model to eval mode and freeze the encoders
    model.eval()

    # Compute predictions and targets
    y_predictions, y_targets = [], []
    for _, target, audio in tqdm(eval_loader, total=len(eval_loader), desc="Evaluating Zero-Shot Classification"):
        audio_embeddings = model.get_audio_embeddings(audio)
        similarity = model.compute_similarity(class_embeddings, audio_embeddings)
        y_pred = F.softmax(similarity.detach().cpu(), dim=1).numpy()
        y_predictions.append(y_pred)
        y_targets.append(target)

    # Compute Zero-Shot accuracy
    y_predictions = np.concatenate(y_predictions, axis=0)
    y_predictions = np.argmax(y_predictions, axis=1)
    y_targets = np.concatenate(y_targets, axis=0)
    acc = accuracy_score(y_targets, y_predictions)

    return acc
