import torch
from torch.utils.data import DataLoader

from tqdm import tqdm

from ..model import Clap
from ..training import BatchRetrievalMetrics, ClapTrainer


def eval_retrieval(model: Clap, eval_loader: DataLoader) -> dict[str, float]:
    """Evaluates the retrieval performance of the given model on a given test dataset.

    Parameters
    ----------
    model : Clap
        The pre-trained model used to compute text and audio embeddings and their similarities.
    eval_loader : DataLoader
        A DataLoader providing batches of test data, where each batch contains captions and audio samples.

    Returns
    -------
    dict[str, float]
        A dictionary containing the averaged retrieval metrics
    """
    # Set the model to eval mode
    model.eval()

    batch_metrics = BatchRetrievalMetrics()

    # Compute embeddings for every caption
    text_embeddings = []
    for _, text, _ in tqdm(eval_loader, total=len(eval_loader), desc="Computing text embeddings"):
        text_embeddings.extend(model.get_text_embeddings(text))

    text_embeddings = torch.stack(text_embeddings, dim=0)

    # Compute the complete similarity matrix
    similarity = torch.empty(size=(len(text_embeddings), len(text_embeddings)))
    for i, (_, _, audio) in tqdm(enumerate(eval_loader), total=len(eval_loader), desc="Computing similarity scores"):
        # Compute similarity matrix
        audio_embedding = model.get_audio_embeddings(audio)
        sim = model.compute_similarity(text_embeddings, audio_embedding)
        similarity[i*eval_loader.batch_size:(i+1)*eval_loader.batch_size] = sim

    # Audio-to-Text retrieval
    r1_a2t, r5_a2t, r10_a2t, map10_a2t, map_a2t = ClapTrainer.compute_retrieval_metrics(similarity)

    # Text-to-Audio retrieval
    r1_t2a, r5_t2a, r10_t2a, map10_t2a, map_t2a = ClapTrainer.compute_retrieval_metrics(similarity.T)

    # Log metrics
    batch_metrics.update(
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

    return batch_metrics.compute_average_metrics()
