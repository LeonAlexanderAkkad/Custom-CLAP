from pathlib import Path

import yaml

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from .layers import TextEncoder, AudioEncoder


class Clap(nn.Module):
    """Custom CLAP implementation based on https://doi.org/10.1109/ICASSP49357.2023.10095889.

    Attributes
    ----------
    text_encoder : nn.Module
        The pretrained text encoder.
    audio_encoder : nn.Module
        The pretrained audio encoder.
    logit_scale : nn.Parameter
        Trainable parameter for scaling the logits and used as temperature when calculating the similarity matrix.
    """
    def __init__(self, config: dict):
        """
        Initializes the CLAP model.

        Parameters
        ----------
        config : dict
            Dictionary containing the parameters of the CLAP model:

            Text Parameters
            ---------------
            text_enc_name : str
                Name of the pretrained text encoder to load.
            text_proj_in : int
                Number of input channels for the projection layer in the text encoder.

            Audio Parameters
            ----------------
            audio_enc_name : str
                Name of the pretrained audio encoder to load.
            sample_rate : int
                Sample rate used for the LogmelFilterBank to determine the frequency range of the input audio.
            window_size : int
                Size of the window used for the Short-Time Fourier Transform (STFT), used to compute the spectrogram.
            hop_size : int
                Stride of the window used for the Short-Time Fourier Transform.
            mel_bins : int
                Number of bins in the mel spectrogram.
            f_min : int
                Lower bound for the frequency of the Mel filter bank.
            f_max : int
                Upper bound for the frequency of the Mel filter bank.
            classes_num : int
                Number of classes in the audio dataset.
            audio_proj_in : int
                Number of input channels for the projection layer in the audio encoder.

            Shared Parameters
            -----------------
            proj_hidden : int
                Number of hidden channels for the projection layer.
            proj_out : int
                Number of output channels for the projection layer.
        """
        super().__init__()

        self.text_encoder = TextEncoder(config["text_encoder"], config["projection"])

        self.audio_encoder = AudioEncoder(config["audio_encoder"], config["projection"])

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
    def from_ckpt(cls, config: Path | str, ckpt: Path | str) -> "Clap":
        """
        Create an instance of Clap from a checkpoint file (e.g. a ckpt file)
        and a given configuration file (e.g. a yml file).

        Parameters
        ----------
        config : Path or str
            The path to the configuration file.
            This can be either a `Path` object or a string representing the file path.

        ckpt: Path or str
            The path to the checkpoint file.
            This can be either a `Path` object or a string representing the file path.

        Returns
        -------
        Clap
            An instance of the Clap class initialized with the checkpoint
            and configuration loaded from the specified file.

        Examples
        --------
        >>> clap_instance = Clap.from_ckpt("config.yml", "checkpoint.ckpt")
        >>> isinstance(clap_instance, Clap)
        True
        """
        clap = cls.from_file(config)
        cls.load_ckpt(clap, ckpt)

        return clap

    @classmethod
    def from_file(cls, config: Path | str) -> "Clap":
        """
        Create an instance of Clap from a configuration file (e.g. a yml config file).

        Parameters
        ----------
        config : Path or str
            The path to the configuration file.
            This can be either a `Path` object or a string representing the file path.

        Returns
        -------
        Clap
            An instance of the Clap class initialized with the configuration loaded from the specified file.

        Examples
        --------
        >>> clap_instance = Clap.from_file("config.yml")
        >>> isinstance(clap_instance, Clap)
        True

        """
        config = cls.load_config(config)
        clap = Clap(config)

        return clap

    @staticmethod
    def load_ckpt(model: nn.Module, ckpt: Path | str):
        ckpt = torch.load(ckpt)
        model.load_state_dict(ckpt["model_state_dict"])

    @staticmethod
    def load_config(config: Path | str) -> dict:
        with open(config, "r") as f:
            config = yaml.safe_load(f)

        return config
