import os

from glob import glob

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
        """Initializes the CLAP model.

        This constructor sets up the CLAP model by initializing the text encoder,
        audio encoder, and logit scale using the provided configuration.

        Parameters
        ----------
        config : dict, Path, or str
            The configuration file containing all the hyperparameters. It can be
            a dictionary, a Path object, or a string representing the path to the
            configuration file.

        Examples
        --------
        **Example 1: Initializing with a dictionary**

        >>> config_dict = {
        ...     "text": {"param1": "value1", "param2": "value2"},
        ...     "audio": {"param1": "value1", "param2": "value2"},
        ...     "projection": {"param1": "value1", "param2": "value2"}
        ... }
        >>> clap_model = Clap(config_dict)

        **Example 2: Initializing with a Path object**

        >>> config_path = Path("path/to/config.yml")
        >>> clap_model = Clap(config_path)

        **Example 4: Loading from a YAML configuration file**

        If the configuration is stored in a YAML file, the `load_config` function
        will handle loading it into a dictionary internally:

        >>> clap_model = Clap("path/to/config.yml")

        **Example 5: Handling errors**

        Ensure the configuration file path is correct and the file is properly
        formatted:

        >>> try:
        ...     clap_model = Clap("invalid/path/to/config.yml")
        ... except FileNotFoundError:
        ...     print("Config file not found.")
        ... except ValueError as e:
        ...     print(f"Config file is invalid: {e}")
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
    def from_ckpt(cls, audio_encoder: str, text_encoder: str, version: str | int) -> "Clap":
        """Create an instance of Clap from a checkpoint file (e.g. a ckpt file).

        This method searches for the appropriate configuration and checkpoint files
        based on the provided `audio_encoder`, `text_encoder`, and `version` parameters.
        It initializes a `Clap` instance using these files.

        Parameters
        ----------
        audio_encoder : str
            The name of the audio encoder to use.
        text_encoder : str
            The name of the text encoder to use.
        version : str or int
            The version of the checkpoint to load.

        Returns
        -------
        Clap
            An instance of the Clap class initialized with the checkpoint.

        Raises
        ------
        ValueError
            If no matching config file or checkpoint file is found.

        Examples
        --------
        >>> clap_instance = Clap.from_ckpt("cnn14", "distilroberta-base", 1)

        This will not only search for a configuration file that includes `cnn14` and `distilroberta-base`
        in its name and ends with `v1.yml`, but also a checkpoint file that
        includes `cnn14` and `distilroberta-base` in its name and ends with `v1.ckpt`.

        If no matching files are found, a `ValueError` will be raised:
        >>> Clap.from_ckpt("nonexistent_audio_encoder", "nonexistent_text_encoder", 999)
        Traceback (most recent call last):
            ...
        ValueError: No config file found. Available configs: ['clap/configs/clap_cnn14_distilroberta-base.yml', ...]
        """
        # Get config path
        config = None
        config_paths = glob(os.path.join("clap", "configs", "*.yml"))
        for config_path in config_paths:
            if audio_encoder in config_path and text_encoder in config_path and "v" + str(version) in config_path:
                config = config_path

        if config is None:
            raise ValueError(f"No config file found. Available configs: {config_paths}")

        # Get checkpoint path
        ckpt = None
        ckpt_paths = glob(os.path.join("clap", "checkpoints", "*.ckpt"))
        for ckpt_path in ckpt_paths:
            if audio_encoder in ckpt_path and text_encoder in ckpt_path and "v" + str(version) in ckpt_path:
                ckpt = ckpt_path

        if ckpt is None:
            raise ValueError(f"No config file found. Available configs: {ckpt_paths}")

        clap = cls(config)
        cls.__load_ckpt(clap, ckpt)

        return clap

    @staticmethod
    def __load_ckpt(model: nn.Module, ckpt: Path | str | dict):
        if not isinstance(ckpt, dict):
            ckpt = torch.load(ckpt)

        model.load_state_dict(ckpt["model"])
