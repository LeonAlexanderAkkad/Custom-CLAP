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
    # Set the model to eval mode
    model.eval()

    # Compute predictions and targets
    predictions, targets = [], []
    for _, target, audio in tqdm(eval_loader, total=len(eval_loader), desc="Evaluating Zero-Shot Classification"):
        audio_embeddings = model.get_audio_embeddings(audio)
        similarity = model.compute_similarity(class_embeddings, audio_embeddings)
        pred = F.softmax(similarity.detach().cpu(), dim=1).numpy()
        predictions.append(pred)
        targets.append(target.detach().cpu().numpy())

    # Compute Zero-Shot accuracy
    predictions = np.concatenate(predictions, axis=0)
    predictions = np.argmax(predictions, axis=1)
    targets = np.concatenate(targets, axis=0)
    acc = accuracy_score(targets, predictions)

    return acc
