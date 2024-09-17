import torch
from torch.utils.data import DataLoader

from tqdm import tqdm

from ..model import ClapAudioClassifier
from ..metrics import BatchClassificationMetrics


def eval_fine_tuned_classification(model: ClapAudioClassifier, eval_loader: DataLoader) -> float:
    """Evaluates the classification accuracy of a given fine-tuned classifier.

    Parameters
    ----------
    model : ClapAudioClassifier
        The model used to compute the class predictions.
    eval_loader : DataLoader
        A DataLoader providing batches of data, where each batch contains audio samples and their corresponding target classes.

    Returns
    -------
    float
        The avg classification accuracy of the classifier.
    """
    # Set the model to eval mode
    model.eval()

    batch_metrics = BatchClassificationMetrics()

    # Compute predictions
    predictions, targets = [], []
    with torch.no_grad():
        for _, target, audio in tqdm(eval_loader, total=len(eval_loader), desc="Evaluating Fine-Tuned Classification"):
            prediction = model(audio)
            predictions.append(prediction)
            targets.append(target)

    # Compute accuracy
    acc = BatchClassificationMetrics.compute_accuracy(prediction, target)

    # Update metrics
    batch_metrics.update(accuracy=acc)

    return batch_metrics.compute_average_metrics()["avg_accuracy"]
