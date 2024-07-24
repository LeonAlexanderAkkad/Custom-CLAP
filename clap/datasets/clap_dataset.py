from pathlib import Path

import random

from typing import Literal, Union

import numpy as np

import torch
import torchvision
import torchaudio
import torchaudio.transforms as T

from .audiocaps import AudioCaps
from .clotho import Clotho
from .audio_dataset import AudioDataset
from ..utils import load_config, get_target_device


AVAILABLE_DATASETS = {"Audiocaps", "Clotho"}


class ClapDataset(AudioDataset):
    def __init__(
            self,
            config: dict | Path | str,
            kind: Literal["train", "val", "test"] = "train",
            download: bool = False,
            datasets: list[Union[Literal["Audiocaps"], Literal["Clotho"]]] = ("Audiocaps", "Clotho")
    ):
        self.datasets = datasets
        super().__init__(kind, download)

        self.audio_cfg = load_config(config)["audio"]

        self.sampling_rate = self.audio_cfg["sampling_rate"]
        self.use_fusion = self.audio_cfg["use_fusion"]
        self.device = get_target_device()

        self.mel_extractor = T.MelSpectrogram(
            sample_rate=self.sampling_rate,
            n_fft=self.audio_cfg["window_size"],
            win_length=self.audio_cfg["window_size"],
            hop_length=self.audio_cfg["hop_size"],
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm=None,
            n_mels=self.audio_cfg["mel_bins"],
            f_min=self.audio_cfg["f_min"],
            f_max=self.audio_cfg["f_max"]
        ).to(self.device)

        self.log_mel_extractor = T.AmplitudeToDB(top_db=None).to(self.device)

    def __getitem__(self, index: int):
        """Returns file given an index."""
        audio_path = self.data[index]
        audio = self.load_sample(audio_path)
        caption = self.captions[index]

        return audio_path, caption, audio

    def get_data(self):
        data = []
        captions = []
        for dataset in self.datasets:
            match dataset.lower():
                case "audiocaps":
                    audiocaps = AudioCaps(self.kind, self.download)
                    data.extend(audiocaps.data)
                    captions.extend(audiocaps.captions)
                case "clotho":
                    clotho = Clotho(self.kind, self.download)
                    data.extend(clotho.data)
                    captions.extend(clotho.captions)
                case _:
                    raise ValueError(f"Unknown dataset {dataset}. Available datasets: {list(AVAILABLE_DATASETS)}")

        return data, captions

    def load_sample(self, audio_path: str) -> dict[str, torch.Tensor]:
        """Method for loading the given sample such that it can be used for training."""
        # Get the audio and reshape it into mono.
        audio, sample_rate = self.load_audio(audio_path)
        audio = audio.reshape(-1)
        audio_len = len(audio)

        max_len = self.audio_cfg["duration"] * sample_rate
        audio_sample = dict()

        # Audio is too long and needs to be cropped or fused.
        if audio_len > max_len:
            audio_mel, is_longer = self.truncate_audio(audio, max_len)
        # Audio is too short and most be extended.
        elif audio_len < max_len:
            audio_mel, is_longer = self.pad_audio(audio, max_len)
        # Audio is neither too short nor too long
        else:
            audio_mel = self.get_mel_spectrogram(audio)
            audio_mel = torch.stack([audio_mel, audio_mel, audio_mel, audio_mel], dim=0)
            is_longer = False

        audio_sample["is_longer"] = torch.tensor(is_longer).to(self.device)
        audio_sample["audio"] = audio_mel

        return audio_sample

    def truncate_audio(self, audio: torch.Tensor, max_len: int) -> tuple[torch.Tensor, bool]:
        if self.use_fusion:
            # Feature Fusion
            audio_mel = self.get_mel_spectrogram(audio)
            # Split the spectrogram into three parts
            chunk_frames = max_len // self.audio_cfg['hop_size'] + 1  # The +1 is related to the spectrogram computation
            total_frames = audio_mel.shape[0]
            if chunk_frames == total_frames:
                # there is a corner case where the audio length is
                # larger than max_len but smaller than max_len+hop_size
                # In this case, we just use the whole audio
                audio_mel = torch.stack([audio_mel, audio_mel, audio_mel, audio_mel], dim=0)
                is_longer = False
            else:
                ranges = np.array_split(list(range(0, total_frames - chunk_frames + 1)), 3)

                # If the audio is too short, use the first chunk
                if len(ranges[1]) == 0:
                    ranges[1] = np.array([0])
                if len(ranges[2]) == 0:
                    ranges[2] = np.array([0])

                # Randomly choose the index for each part
                idx_front = np.random.choice(ranges[0])
                idx_middle = np.random.choice(ranges[1])
                idx_back = np.random.choice(ranges[2])

                # Select the mel for each chung
                audio_mel_chunk_front = audio_mel[idx_front:idx_front + chunk_frames, :]
                audio_mel_chunk_middle = audio_mel[idx_middle:idx_middle + chunk_frames, :]
                audio_mel_chunk_back = audio_mel[idx_back:idx_back + chunk_frames, :]

                # Shrink the mel
                audio_mel_shrink = torchvision.transforms.Resize(
                    size=[chunk_frames, self.audio_cfg['mel_bins']]
                )(audio_mel.unsqueeze(0)).squeeze(0)

                # Finally, stack the mel chunks
                audio_mel = torch.stack(
                    [audio_mel_shrink, audio_mel_chunk_front, audio_mel_chunk_middle, audio_mel_chunk_back], dim=0
                )
                is_longer = True
        else:
            start_index = random.randrange(len(audio) - max_len)
            audio = audio[start_index:start_index + max_len]
            audio_mel = self.get_mel_spectrogram(audio)
            is_longer = False

        return audio_mel, is_longer

    def pad_audio(self, audio: torch.Tensor, max_len: int) -> tuple[torch.Tensor, bool]:
        # Simply repeat the audio.
        repeat_factor = int(np.ceil(max_len / len(audio)))
        # Remove excess part of audio_time_series.
        audio = audio.repeat(repeat_factor)[0:max_len]
        audio_mel = self.get_mel_spectrogram(audio)

        if self.use_fusion:
            audio_mel = torch.stack([audio_mel, audio_mel, audio_mel, audio_mel], dim=0)
        is_longer = False

        return audio_mel, is_longer

    def load_audio(self, audio_path: str, resample: bool = True):
        """Loads the given audio and resamples it if wanted"""
        audio, sampling_rate = torchaudio.load(audio_path)

        if resample and sampling_rate != self.sampling_rate:
            resampler = T.Resample(sampling_rate, self.sampling_rate)
            audio = resampler(audio)
            sampling_rate = self.sampling_rate

        return audio.to(self.device), sampling_rate

    def get_mel_spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        """Returns the log-mel spectrogram of the given audio."""
        audio_spec = self.mel_extractor(audio)
        audio_mel = self.log_mel_extractor(audio_spec)

        return audio_mel.T
