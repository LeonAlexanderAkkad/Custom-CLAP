import torch
from torch import nn

from ..projection import Projection

from .cnn14 import Cnn14
from .htsat import HTSAT_Swin_Transformer


AUDIO_ENCODERS = {"Cnn14", "HTSAT"}


class AudioEncoder(nn.Module):
    """Defines and loads the audio encoder for CLAP."""

    def __init__(self, audio_cfg: dict, proj_cfg: dict):
        super().__init__()

        self.audio_cfg = audio_cfg
        self.proj_cfg = proj_cfg
        self.name = self.audio_cfg["name"]

        self.audio_encoder = self.load_audio_encoder()

        self.projection = Projection(
            n_input_features=self.audio_cfg["out_size"],
            n_hidden_features=self.proj_cfg["hidden_size"],
            n_output_features=self.proj_cfg["out_size"],
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
        pretrained_path = self.audio_cfg["pretrained_path"]

        match self.name.upper():
            case "CNN14":
                model = Cnn14(
                    # sampling_rate=self.audio_cfg["sampling_rate"],
                    # window_size=self.audio_cfg["window_size"],
                    # hop_size=self.audio_cfg["hop_size"],
                    # mel_bins=self.audio_cfg["mel_bins"],
                    # f_min=self.audio_cfg["f_min"],
                    # f_max=self.audio_cfg["f_max"],
                    # classes_num=self.audio_cfg["classes_num"],
                    # emb_out=self.audio_cfg["out_size"],
                    config=self.audio_cfg
                )
            case "HTSAT":
                model = HTSAT_Swin_Transformer(config=self.config_audio)
            case _:
                raise NotImplementedError(f"Audio encoder '{self.name}' not implemented.\n"
                                          f"Available encoders: {list(AUDIO_ENCODERS)}.")

        if pretrained_path:
            ckpt = torch.load(pretrained_path)
            # pop_keys = list(ckpt["model"].keys())
            # for key in pop_keys:
            #     if key.startswith("spectrogram_extractor.") or key.startswith("logmel_extractor."):
            #         ckpt["model"].pop(key)

            model.load_state_dict(ckpt["model"], strict=False)

        return model
