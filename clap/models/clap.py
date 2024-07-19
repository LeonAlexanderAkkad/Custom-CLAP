import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from .text_encoders import load_text_encoder
from .audio_encoders import load_audio_encoder


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
    def from_ckpt(cls, ...):
        pass


class TextEncoder(nn.Module):
    """
    Defines and loads the text encoder for CLAP.
    """

    def __init__(self, config_text: dict, config_proj: dict):
        super().__init__()

        self.config_text = config_text
        self.config_proj = config_proj
        self.name = self.config_text["name"]
        self.text_encoder, self.tokenizer = load_text_encoder(self.name)

        self.projection = Projection(
            n_input_features=self.config_text["out_size"],
            n_hidden_features=self.config_proj["hidden_size"],
            n_output_features=self.config_proj["out_size"],
            activation_function=nn.GELU(),
            dropout=0.5
        )

    def forward(self, text: torch.Tensor):
        if "bert" in self.name:
            # Tokenize text into a dictionary with the shape:
            # {'input_ids': torch.Tensor, 'attention_mask': torch.Tensor, 'token_type_ids': torch.Tensor}
            tokenized_text = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.config_text["max_length"],
                return_tensors="pt"
            )

            # Get the last hidden state
            output = self.text_encoder(**tokenized_text)[0]
            # Extract CLS token
            output = output[:, 0, :]
        else:
            raise NotImplementedError(f"No forward method implemented for {self.name}")

        # Projects the embedding into the same space as the audio embedding.
        projected = self.projection(output)

        return projected


class AudioEncoder(nn.Module):
    """Defines and loads the audio encoder for CLAP."""

    def __init__(self, config_audio: dict, config_proj: dict):
        super().__init__()

        self.config_audio = config_audio
        self.config_proj = config_proj
        self.audio_encoder = load_audio_encoder(self.config_audio)

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


class Projection(nn.Module):
    """The final projection layer for both the text and audio encoder of CLAP."""

    def __init__(
            self,
            n_input_features: int,
            n_hidden_features: int,
            n_output_features: int,
            activation_function: nn.Module = nn.GELU(),
            dropout: float = 0.5
    ):
        super().__init__()

        self.fc1 = nn.Linear(n_input_features, n_hidden_features)
        self.fc2 = nn.Linear(n_hidden_features, n_output_features)

        self.act_fn = activation_function
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(n_output_features)

    def forward(self, x: torch.Tensor):
        out1 = self.fc1(x)
        out2 = self.dropout(self.fc2(self.act_fn(out1)))
        out = self.layer_norm(out1 + out2)

        return out
