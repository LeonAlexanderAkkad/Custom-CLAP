import torch
from torch import nn

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

    def forward(self, audio: torch.Tensor):
        # Returns a dictionary with the probabilities of each class being present in the audio and the embedding.
        output = self.base(audio)
        # Projects the embedding into the same space as the text embedding.
        projected = self.projection(output["embedding"])

        return projected

    def load_audio_encoder(self) -> nn.Module:
        """Loads respective audio encoder model"""
        if not self.__is_valid():
            raise NotImplementedError(
                f"Text encoder '{self.name}' not implemented.\nAvailable encoders: {list(AUDIO_ENCODERS.keys())}"
            )

        pretrained_path = self.audio_cfg["pretrained_path"]

        encoder = AUDIO_ENCODERS[self.name](config=self.audio_cfg)

        if pretrained_path:
            ckpt = torch.load(pretrained_path)
            # pop_keys = list(ckpt["model"].keys())
            # for key in pop_keys:
            #     if key.startswith("spectrogram_extractor.") or key.startswith("logmel_extractor."):
            #         ckpt["model"].pop(key)

            if ckpt.get("model") is not None:
                state_dict = ckpt["model"]
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
