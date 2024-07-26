import os

from pathlib import Path

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from .layers import TextEncoder, AudioEncoder
from ..utils import load_config


class Clap(nn.Module):
    """Custom CLAP implementation based on https://doi.org/10.1109/ICASSP49357.2023.10095889.

    Attributes
    ----------
    text_encoder : nn.Module
        The pretrained text encoder. Available encoders: RoBERTa, GPT-2.
    audio_encoder : nn.Module
        The pretrained audio encoder. Available encoders: Cnn14, HTSAT.
    logit_scale : nn.Parameter
        Trainable parameter for scaling the logits and used as temperature when calculating the similarity matrix.
    """
    def __init__(self, config: dict | Path | str):
        """
        Initializes the CLAP model.

        Parameters
        ----------
        config : dict | Path | str
            The config file containing all the hyperparameters.

        """
        super().__init__()

        config = load_config(config)

        self.text_encoder = TextEncoder(config["text"], config["projection"])

        self.audio_encoder = AudioEncoder(config["audio"], config["projection"])

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, text: torch.Tensor, audio: torch.Tensor):
        text_embedding = self.text_encoder(text)
        audio_embedding = self.audio_encoder(audio)

        return F.normalize(text_embedding, dim=-1), F.normalize(audio_embedding, dim=-1), self.logit_scale.exp()

    def compute_similarity(self, text_embedding: torch.Tensor, audio_embedding: torch.Tensor) -> torch.Tensor:
        """Computes the similarity matrix between a normalized text embedding and audio embedding."""
        similarity = self.logit_scale.exp() * text_embedding @ audio_embedding.T

        return similarity.T

    @classmethod
    def from_ckpt(cls, ckpt: Path | str) -> "Clap":
        """
        Create an instance of Clap from a checkpoint file (e.g. a ckpt file)

        Parameters
        ----------
        ckpt: Path or str
            The path to the checkpoint file.
            This can be either a `Path` object or a string representing the file path.

        Returns
        -------
        Clap
            An instance of the Clap class initialized with the checkpoint

        Examples
        --------
        >>> clap_instance = Clap.from_ckpt("clap/checkpoints/clap_cnn14_roberta.ckpt")
        >>> isinstance(clap_instance, Clap)
        True
        """
        version = os.path.basename(ckpt).split(".")[0]

        config_path = Path(__file__).parent / "configs" / f"{version}.yml"

        clap = cls(config_path)
        cls.__load_ckpt(clap, ckpt)

        return clap

    @staticmethod
    def __load_ckpt(model: nn.Module, ckpt: Path | str):
        ckpt = torch.load(ckpt)
        model.load_state_dict(ckpt["model_state_dict"])
