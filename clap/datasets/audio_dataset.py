from abc import ABC, abstractmethod

import random

from typing import Literal

import numpy as np

import torch
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T


class AudioDataset(Dataset, ABC):
    """Simple abstract audio dataset used for training."""

    def __init__(
        self,
        audiodata_dir: str,
        metadata_dir: str,
        kind: Literal["train", "val", "test"] = "train",
        download: bool = False,
        sampling_rate: int = 44100,
        duration: int = 10
    ):
        """Initialize the audio dataset and download it if needed."""
        self.sampling_rate = sampling_rate
        self.duration = duration
        self.kind = kind
        self.download = download
        self.data, self.captions = self.get_data(audiodata_dir, metadata_dir)

    def __getitem__(self, index: int):
        """Returns file given an index."""
        sample = self.load_sample(self.data[index])
        target = self.captions[index]

        return index, sample, target

    def __len__(self):
        """Returns the number of files in the dataset."""
        return len(self.data)

    def load_sample(self, audio_path: str) -> torch.Tensor:
        """Method for loading the given sample such that it can be used for training."""
        # Get the audio and reshape it into mono.
        audio, sample_rate = self.load_audio(audio_path)
        audio = audio.reshape(-1)

        max_len = self.duration * sample_rate

        # Audio is too short and most be extended.
        if len(audio) < max_len:
            # Simply repeat the audio.
            repeat_factor = int(np.ceil(max_len / len(audio)))
            audio = audio.repeat(repeat_factor)
            # Remove excess part of audio_time_series.
            audio = audio[0:max_len]
        # Audio is too long and needs to be cropped or fused.
        elif len(audio) > max_len:
            # TODO: Implement fusion.
            start_index = random.randrange(len(audio) - max_len)
            audio = audio[start_index:start_index + max_len]

        return torch.FloatTensor(audio)

    def load_audio(self, audio_path: str, resample: bool = True):
        """Loads the given audio and resamples it if wanted"""
        audio, sampling_rate = torchaudio.load(audio_path)

        if resample and sampling_rate != self.sampling_rate:
            resampler = T.Resample(sampling_rate, self.sampling_rate)
            audio = resampler(audio)
            sampling_rate = self.sampling_rate

        return audio, sampling_rate

    @abstractmethod
    def get_data(self, audio_data_dir: str, metadata_dir: str):
        raise NotImplementedError
