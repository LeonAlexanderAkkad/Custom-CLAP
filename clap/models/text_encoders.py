import torch.nn as nn
from transformers import AutoModel


TEXT_ENCODERS = {"roberta"}


def load_text_encoder(name: str) -> type[nn.Module]:
    """Loads respective pretrained text encoder model from Huggingface."""
    if name not in TEXT_ENCODERS:
        raise NotImplementedError(f"Text encoder '{name}' not implemented.\nAvailable encoders: {list(TEXT_ENCODERS)}")

    return AutoModel.from_pretrained(name)
