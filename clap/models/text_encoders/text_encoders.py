from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


TEXT_ENCODERS = {"roberta"}


def load_text_encoder(name: str) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Loads respective pretrained text encoder model from Huggingface."""
    if name not in TEXT_ENCODERS:
        raise NotImplementedError(f"Text encoder '{name}' not implemented.\nAvailable encoders: {list(TEXT_ENCODERS)}")

    return AutoModel.from_pretrained(name), AutoTokenizer.from_pretrained(name)
