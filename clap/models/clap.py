import numpy as np

import torch
from torch import nn

from audio_encoders import load_audio_encoder


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
    def __init__(
            self,
            # TextEncoder params
            text_enc_name: str,
            text_proj_in: int,
            # AudioEncoder params
            audio_enc_name: str,
            sample_rate: int,
            window_size: int,
            hop_size: int,
            mel_bins: int,
            f_min: int,
            f_max: int,
            classes_num: int,
            audio_proj_in: int,
            # Shared params
            proj_hidden: int,
            proj_out: int
    ):
        """
        Initializes the CLAP model.

        Parameters
        ----------

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

        self.text_encoder = TextEncoder(
            name=text_enc_name,
            proj_in=text_proj_in,
            proj_hidden=proj_hidden,
            proj_out=proj_out
        )

        self.audio_encoder = AudioEncoder(
            name=audio_enc_name,
            sample_rate=sample_rate,
            window_size=window_size,
            hop_size=hop_size,
            mel_bins=mel_bins,
            f_min=f_min,
            f_max=f_max,
            classes_num=classes_num,
            proj_in=audio_proj_in,
            proj_hidden=proj_hidden,
            proj_out=proj_out
        )

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, text: torch.Tensor, audio: torch.Tensor):
        text_embeddings = self.text_encoder(text)
        audio_embeddings = self.audio_encoder(audio)

        return text_embeddings, audio_embeddings, self.logit_scale.exp()


class TextEncoder(nn.Module):
    """
    Defines and loads the text encoder for CLAP.
    """
    def __init__(
            self,
            name: str,
            proj_in: int,
            proj_hidden: int,
            proj_out: int
    ):
        super().__init__()

        self.name = name
        self.text_encoder = load_audio_encoder(name)

        self.projection = Projection(
            n_input_features=proj_in,
            n_hidden_features=proj_hidden,
            n_output_features=proj_out,
            activation_function=nn.GELU(),
            dropout=0.5
        )

    def forward(self, text: dict[str, torch.Tensor]):
        if "bert" in self.name:
            # Get the last hidden state
            output = self.text_encoder(**text)[0]
            # Extract CLS token
            output = output[:, 0, :]
        else:
            raise NotImplementedError(f"No forward method implemented for {self.name}")

        # Projects the embedding into the same space as the audio embedding.
        projected = self.projection(output)

        return projected


class AudioEncoder(nn.Module):
    """Defines and loads the audio encoder for CLAP."""
    def __init__(
            self,
            name: str,
            sample_rate: int,
            window_size: int,
            hop_size: int,
            mel_bins: int,
            f_min: int,
            f_max: int,
            classes_num: int,
            proj_in: int,
            proj_hidden: int,
            proj_out: int
    ):
        super().__init__()

        audio_encoder = load_audio_encoder(name)

        self.audio_encoder = audio_encoder(
            sample_rate=sample_rate,
            window_size=window_size,
            hop_size=hop_size,
            mel_bins=mel_bins,
            fmin=f_min,
            fmax=f_max,
            classes_num=classes_num
        )

        self.projection = Projection(
            n_input_features=proj_in,
            n_hidden_features=proj_hidden,
            n_output_features=proj_out,
            activation_function=nn.GELU(),
            dropout=0.5
        )

    def forward(self, audio: torch.Tensor):
        # Returns a dictionary with the probabilities of each class being present in the audio and the embedding.
        output = self.audio_encoder(audio)
        # Projects the embedding into the same space as the text embedding.
        projected = self.projection(output["embedding"])
        probabilities = output["clip_wise_output"]

        return projected, probabilities


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
