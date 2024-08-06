import os

import torch
from torch import nn
import torch.nn.functional as F

from ..projection import Projection

from .cnn14 import Cnn14
from .htsat import HTSAT_Swin_Transformer


AUDIO_ENCODERS = {"CNN14": Cnn14, "HTSAT": HTSAT_Swin_Transformer}


class AudioEncoder(nn.Module):
    """Defines and loads the audio encoder for CLAP."""

    def __init__(self, audio_cfg: dict, proj_cfg: dict):
        super().__init__()

        self.audio_cfg = audio_cfg
        self.proj_cfg = proj_cfg
        self.name = self.audio_cfg["name"].upper()

        self.base = self.load_audio_encoder()

        self.projection = Projection(
            n_input_features=self.audio_cfg["out_size"],
            n_hidden_features=self.proj_cfg["hidden_size"],
            n_output_features=self.proj_cfg["out_size"],
            activation_function=nn.GELU(),
            dropout=0.5
        )

    def forward(self, audio: dict[str, torch.Tensor]):
        # Returns a dictionary with the probabilities of each class being present in the audio and the embedding.
        output = self.base(audio)
        # Projects the embedding into the same space as the text embedding.
        projected = self.projection(output["embedding"])

        return F.normalize(projected, dim=-1)

    def load_audio_encoder(self) -> nn.Module:
        """Loads respective audio encoder model"""
        if not self.__is_valid():
            raise NotImplementedError(
                f"Text encoder '{self.name}' not implemented.\nAvailable encoders: {list(AUDIO_ENCODERS.keys())}"
            )

        pretrained_path = self.audio_cfg["pretrained_abs_path"]

        encoder = AUDIO_ENCODERS[self.name](config=self.audio_cfg)

        if pretrained_path and os.path.exists(pretrained_path):
            ckpt = torch.load(pretrained_path)

            if ckpt.get("model") is not None:
                state_dict = ckpt["model"]
            elif ckpt.get("state_dict") is not None:
                state_dict = ckpt["state_dict"]
                for key in list(state_dict.keys()):
                    v = state_dict.pop(key)
                    state_dict[key[10:]] = v
            else:
                state_dict = ckpt

            encoder.load_state_dict(state_dict, strict=False)

        return encoder

    def __is_valid(self) -> bool:
        """Checks if the audio encoder is valid."""
        for encoder in AUDIO_ENCODERS.keys():
            if encoder in self.name:
                self.name = encoder
                return True

        return False
