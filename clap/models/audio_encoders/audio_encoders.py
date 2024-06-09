import torch.nn as nn

from cnn14 import Cnn14


AUDIO_ENCODERS = {"Cnn14"}


def load_audio_encoder(name: str = "Cnn14") -> type[nn.Module]:
    """Loads respective audio encoder model"""
    if name not in AUDIO_ENCODERS:
        raise NotImplementedError(f"Audio encoder '{name}' not implemented.\nAvailable encoders: {list(AUDIO_ENCODERS)}")

    if name == "Cnn14":
        return Cnn14

# TODO: Implement Feature Fusion
# TODO: Implement HTS-AT
