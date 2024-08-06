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

    def freeze_encoders(self):
        """Freezes both text and audio encoders."""
        self.freeze_text_encoder()
        self.freeze_audio_encoder()

    def unfreeze_encoders(self):
        """Unfreezes both text and audio encoders."""
        self.unfreeze_text_encoder()
        self.unfreeze_audio_encoder()

    def compute_similarity(self, text_embedding: torch.Tensor, audio_embedding: torch.Tensor) -> torch.Tensor:
        """Computes the similarity matrix between a normalized text embedding and audio embedding."""
        similarity = self.logit_scale.exp() * text_embedding @ audio_embedding.T

        return similarity.T

    def get_text_embeddings(self, text: list[str]):
        """Returns the normalized text embedding of the provided text."""
        with torch.no_grad():
            return self.text_encoder(text)

    def get_audio_embeddings(self, audio: list[str]):
        """Returns the normalized audio embedding of the provided audio."""
        with torch.no_grad():
            return self.audio_encoder(audio)

    @classmethod
    def from_ckpt(cls, audio_encoder: str, text_encoder: str, ckpt_version: str | int, cfg_version: str | int) -> "Clap":
        """Create an instance of Clap from a checkpoint file (e.g. a ckpt file).

        This method searches for the appropriate configuration and checkpoint files
        based on the provided `audio_encoder`, `text_encoder`, and `version` parameters.
        It initializes a `Clap` instance using the found files.

        Parameters
        ----------
        audio_encoder : str
            The name of the audio encoder to use.
        text_encoder : str
            The name of the text encoder to use.
        ckpt_version : str or int
            The version of the checkpoint to load.
        cfg_version: str or int
            The version of the config to use.

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

        >>> clap_instance = Clap.from_ckpt("cnn14", "distilroberta-base", 1, 1)

        This will search for a configuration file named `clap_cnn14_distilroberta-base_v1.yml`
        and a checkpoint file named `clap_cnn14_distilroberta-base_v1.ckpt`.
        If the files are found, it will initialize the Clap instance using these files.

        If no matching files are found, a `FileNotFoundError` will be raised:

        >>> Clap.from_ckpt("nonexistent_audio_encoder", "nonexistent_text_encoder", 999, 999)
        Traceback (most recent call last):
            ...
        FileNotFoundError: Config file not found for nonexistent_audio_encoder and nonexistent_text_encoder and v999.
                           Available configs: [clap_cnn14_distilroberta-base_v1.yml, ...]

        Notes
        -----
        - The method `load_clap_ckpt` is responsible for finding and loading the checkpoint file.
        - The method `load_clap_config` is used to find and load the configuration file.
        """
        clap = cls.from_encoders(audio_encoder, text_encoder, cfg_version)

        ckpt = load_clap_ckpt(audio_encoder, text_encoder, ckpt_version)
        clap.load_state_dict(ckpt["model"])

        return clap

    @classmethod
    def from_encoders(cls, audio_encoder: str, text_encoder: str, version: str | int) -> "Clap":
        """Create an instance of Clap from audio and text encoders as well as a config version.

        This method initializes a Clap instance based on the provided
        `audio_encoder` and `text_encoder` by loading the corresponding configuration file.

        Parameters
        ----------
        audio_encoder : str
            The name of the audio encoder to use.
        text_encoder : str
            The name of the text encoder to use.
        version: str or int
            The version of the config to load.

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
        Create a Clap instance from specific audio and text encoders:

        >>> clap_instance = Clap.from_encoders("cnn14", "distilroberta-base", 1)

        This will search for a configuration file named `clap_cnn14_distilroberta-base_v1.yml`
        and initialize the Clap instance using the configuration from this file.

        Notes
        -----
        - The method `load_clap_config` is responsible for finding and loading the configuration file.
        """

        config = load_clap_config(audio_encoder, text_encoder, version)
        return cls(config)
