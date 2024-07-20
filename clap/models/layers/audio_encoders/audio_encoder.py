import torch
from torch import nn

from ..projection import Projection

from cnn14 import Cnn14
from htsat import HTSAT_Swin_Transformer


AUDIO_ENCODERS = {"Cnn14", "HTSAT"}


class AudioEncoder(nn.Module):
    """Defines and loads the audio encoder for CLAP."""

    def __init__(self, config_audio: dict, config_proj: dict):
        super().__init__()

        self.config_audio = config_audio
        self.config_proj = config_proj
        self.name = self.config_audio["name"]

        self.audio_encoder = self.load_audio_encoder()

        self.projection = Projection(
            n_input_features=self.config_audio["out_size"],
            n_hidden_features=self.config_proj["hidden_size"],
            n_output_features=self.config_proj["out_size"],
            activation_function=nn.GELU(),
            dropout=0.5
        )

    def forward(self, audio: torch.Tensor):
        # Returns a dictionary with the probabilities of each class being present in the audio and the embedding.
        output = self.audio_encoder(audio)
        # Projects the embedding into the same space as the text embedding.
        projected = self.projection(output["embedding"])

        return projected

    def load_audio_encoder(self) -> nn.Module:
        """Loads respective audio encoder model"""
        pretrained_path = self.config_audio["pretrained_path"]

        if self.name == "Cnn14":
            model = Cnn14(
                sampling_rate=self.config_audio["sampling_rate"],
                window_size=self.config_audio["window_size"],
                hop_size=self.config_audio["hop_size"],
                mel_bins=self.config_audio["mel_bins"],
                f_min=self.config_audio["f_min"],
                f_max=self.config_audio["f_max"],
                classes_num=self.config_audio["classes_num"],
                emb_out=self.config_audio["out_size"]
            )
            if pretrained_path:
                ckpt = torch.load(pretrained_path)
                model.load_state_dict(ckpt["model"])
        elif self.name == "HTSAT":
            model = HTSAT_Swin_Transformer(config=self.config_audio)
            if pretrained_path:
                ckpt = torch.load(pretrained_path)
                model.load_state_dict(ckpt["model"])
        else:
            raise NotImplementedError(f"Audio encoder '{self.name}' not implemented.\n"
                                      f"Available encoders: {list(AUDIO_ENCODERS)}.")

        return model
