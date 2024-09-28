from pathlib import Path

import numpy as np

import torch
from torch import nn

from .layers import TextEncoder, AudioEncoder
from ..utils import load_clap_config, load_clap_ckpt


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
    def __init__(self, config: dict):
        """Initializes the CLAP model.

        This constructor sets up the CLAP model by initializing the text encoder,
        audio encoder, and logit scale using the provided configuration.

        Parameters
        ----------
        config : dict
            The configuration dictionary containing all the hyperparameters.

        Example
        --------

        >>> config_dict = {
        ...     "text": {"param1": "value1", "param2": "value2"},
        ...     "audio": {"param1": "value1", "param2": "value2"},
        ...     "projection": {"param1": "value1", "param2": "value2"}
        ... }
        >>> clap_model = Clap(config_dict)
        """
        super().__init__()

        self.text_encoder = TextEncoder(config["text"], config["projection"])

        self.audio_encoder = AudioEncoder(config["audio"], config["projection"])

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, text: list[str], audio: dict[str, torch.Tensor]):
        text_embeddings = self.text_encoder(text)
        audio_embeddings = self.audio_encoder(audio)

        return text_embeddings, audio_embeddings, self.logit_scale.exp()

    def freeze_text_encoder(self):
        """Freezes the text encoder."""
        for param in self.text_encoder.parameters():
            param.requires_grad = False

    def unfreeze_text_encoder(self):
        """Unfreezes the text encoder."""
        for param in self.text_encoder.parameters():
            param.requires_grad = True

    def freeze_audio_encoder(self):
        """Freezes the audio encoder."""
        for param in self.audio_encoder.parameters():
            param.requires_grad = False

    def unfreeze_audio_encoder(self):
        """Unfreezes the audio encoder."""
        for param in self.audio_encoder.parameters():
            param.requires_grad = True

    def freeze_model(self):
        """Freezes both text and audio encoders as well as the logit scale."""
        self.freeze_text_encoder()
        self.freeze_audio_encoder()
        self.logit_scale.requires_grad = False

    def unfreeze_model(self):
        """Unfreezes both text and audio encoders as well as the logit scale."""
        self.unfreeze_text_encoder()
        self.unfreeze_audio_encoder()
        self.logit_scale.requires_grad = True

    def compute_similarity(self, text_embedding: torch.Tensor, audio_embedding: torch.Tensor) -> torch.Tensor:
        """Computes the similarity matrix between a normalized text embedding and audio embedding."""
        similarity = self.logit_scale.exp() * text_embedding @ audio_embedding.T
        similarity.clamp_(max=100)

        return similarity.T

    def get_text_embeddings(self, text: list[str]):
        """Returns the normalized text embedding of the provided text."""
        with torch.no_grad():
            return self.text_encoder(text)

    def get_audio_embeddings(self, audio: dict[str, torch.Tensor]):
        """Returns the normalized audio embedding of the provided audio."""
        with torch.no_grad():
            return self.audio_encoder(audio)

    @classmethod
    def from_ckpt(cls, config_path: str | Path, ckpt_path: str | Path) -> "Clap":
        """Create an instance of Clap from a checkpoint file (e.g. a ckpt file).

        It initializes a `Clap` instance using the found config and checkpoint files.

        Parameters
        ----------
        config_path : str | Path
            The path to the checkpoint file.
        ckpt_path : str | Path
            The path to the checkpoint file.

        Returns
        -------
        Clap
            An instance of the Clap class initialized with the checkpoint.

        Raises
        ------
        FileNotFoundError
            If no matching config file or checkpoint file is found.

        Examples
        --------
        Create a Clap instance from a specific checkpoint version:

        >>> clap_instance = Clap.from_ckpt("path/to/config.yml", "path/to/ckpt.ckpt")

        If these files are found, it will initialize the Clap instance using them.

        If no matching files are found, a `FileNotFoundError` will be raised.

        Notes
        -----
        - The method `load_clap_ckpt` is responsible for finding and loading the checkpoint file.
        - The method `load_clap_config` is used to find and load the configuration file.
        """
        clap = cls.from_config_file(config_path)

        ckpt = load_clap_ckpt(ckpt_path)
        clap.load_state_dict(ckpt["model"])

        return clap

    @classmethod
    def from_config_file(cls, config_path: str | Path) -> "Clap":
        """Create an instance of Clap from audio and text encoders as well as a config version.

        This method initializes a Clap instance based on the provided
        `audio_encoder` and `text_encoder` by loading the corresponding configuration file.

        Parameters
        ----------
        config_path : str | Path
            The path to the config file.

        Returns
        -------
        Clap
            An instance of the Clap class initialized with the encoders.

        Raises
        ------
        FileNotFoundError
            If no matching config file is found.

        Examples
        --------
        Create a Clap instance a given config file:

        >>> clap_instance = Clap.from_config_file("path/to/config.yml")

        Notes
        -----
        - The method `load_clap_config` is responsible for finding and loading the configuration file.
        """

        config = load_clap_config(config_path)
        return cls(config)
