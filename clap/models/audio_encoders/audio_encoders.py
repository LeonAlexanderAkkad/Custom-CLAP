import torch
import torch.nn as nn

from .htsat import HTSAT
from .cnn14 import Cnn14

# TODO: Change to factory method
def load_audio_encoder(config: dict) -> nn.Module:
    """Loads respective audio encoder model"""
    name = config["name"]
    pretrained_path = config["pretrained_path"]

    if name == "Cnn14":
        model = Cnn14(
            sampling_rate=config["sampling_rate"],
            window_size=config["window_size"],
            hop_size=config["hop_size"],
            mel_bins=config["mel_bins"],
            f_min=config["f_min"],
            f_max=config["f_max"],
            classes_num=config["classes_num"],
            emb_out=config["out_size"]
        )
        if pretrained_path:
            ckpt = torch.load(pretrained_path)
            model.load_state_dict(ckpt["model"])
    elif name == "HTSAT":
        model = HTSAT(
            # TODO: HTSAT implementation
        )

        if pretrained_path:
            ckpt = torch.load(pretrained_path)
            model.load_state_dict(ckpt["model"])
    else:
        raise NotImplementedError(f"Audio encoder '{name}' not implemented.")

    return model
