import torch.nn as nn

from transformers import AutoModel, AutoTokenizer

from audio_encoders import Cnn14, HTSAT


class Clap:
    """A factory for creating a clap model with pretrained text and audio encoders."""

    text_encoders = {'roberta': (AutoModel.from_pretrained, AutoTokenizer.from_pretrained)}
    audio_encoders = {'cnn14': Cnn14, 'htsat': HTSAT}

    def __init__(self):
        # TODO: Finish Factory!
        pass
