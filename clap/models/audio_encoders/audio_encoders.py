import torch.nn as nn

from cnn14 import Cnn14


AUDIO_ENCODERS = {"Cnn14"}


def load_audio_encoder(config: dict) -> nn.Module:
    """Loads respective audio encoder model"""
    name = config["name"]
    if name not in AUDIO_ENCODERS:
        raise NotImplementedError(f"Audio encoder '{name}' not implemented.\nAvailable encoders: {list(AUDIO_ENCODERS)}")

    if name == "Cnn14":
        return Cnn14(sampling_rate=config["sampling_rate"], window_size=config["window_size"],
                     hop_size=config["hop_size"], mel_bins=config["mel_bins"], f_min=config["f_min"],
                     f_max=config["f_max"], classes_num=config["classes_num"], emb_out=config["audio_out"])
