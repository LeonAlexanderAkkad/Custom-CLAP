from pathlib import Path

import torch
from torch import nn

from .clap import Clap as ClapModel

import yaml


TEXT_ENCODERS = {"roberta"}
AUDIO_ENCODERS = {"Cnn14", "HTSAT"}

# TODO: Change factory class to factory method

class ClapFactory:
    """A factory for creating a clap model with pretrained text and audio encoders."""

    def load_clap(self, config: Path | str, ckpt: Path | str | None = None) -> ClapModel:
        config = self.load_config(config)
        if config["text_encoder"]["name"] not in TEXT_ENCODERS:
            raise NotImplementedError(
                f"Text encoder {config['text_encoder']['name']} not implemented.\n"
                f"Available encoders: {list(TEXT_ENCODERS)}"
            )
        if config["audio_encoder"]["name"] not in AUDIO_ENCODERS:
            raise NotImplementedError(
                f"Audio encoder {config['audio_encoder']['name']} not implemented.\n"
                f"Available encoders: {list(AUDIO_ENCODERS)}"
            )

        model = ClapModel(config)

        if ckpt is not None:
            self.load_ckpt(model, ckpt)

        return model

    @staticmethod
    def load_ckpt(model: nn.Module, ckpt: Path | str):
        ckpt = torch.load(ckpt)
        model.load_state_dict(ckpt["model_state_dict"])

    @staticmethod
    def load_config(config: Path | str) -> dict:
        with open(config, "r") as f:
            config = yaml.safe_load(f)

        return config
