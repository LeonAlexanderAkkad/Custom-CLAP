from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

# TODO: Change to factory method
def load_text_encoder(name: str) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Loads respective pretrained text encoder model from Huggingface."""
    text_encoders = {"roberta"}
    if name not in text_encoders:
        raise NotImplementedError(f"Text encoder '{name}' not implemented.\nAvailable encoders: {list(text_encoders)}")

    return AutoModel.from_pretrained(name), AutoTokenizer.from_pretrained(name)
