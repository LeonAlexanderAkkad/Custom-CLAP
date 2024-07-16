from abc import ABC, abstractmethod

import random

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

import torchaudio
import torchaudio.transforms as T


class AudioDataset(Dataset, ABC):
    """Simple abstract dataset used for training."""

    def __init__(self, audio_data_dir: str, metadata_path: str, sample_rate: int = 44100, duration: int = 10):
        """Sort and store all files found in given directory."""
        self.sample_rate = sample_rate
        self.duration = duration
        self.data, self.captions = self.get_data(audio_data_dir, metadata_path)

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
        mono_audio = audio.reshape(-1)

        max_len = self.duration * sample_rate

        # Audio is too short and most be extended.
        if len(mono_audio) < max_len:
            # Simply repeat the audio.
            repeat_factor = int(np.ceil(max_len / len(mono_audio)))
            mono_audio = mono_audio.repeat(repeat_factor)
            # Remove excess part of audio_time_series.
            mono_audio = mono_audio[0:max_len]
        # Audio is too long and needs to be cropped or fused.
        elif len(mono_audio) > max_len:
            # TODO: Implement fusion.
            start_index = random.randrange(len(mono_audio) - max_len)
            mono_audio = mono_audio[start_index:start_index + max_len]

        return torch.FloatTensor(mono_audio)

    def load_audio(self, audio_path: str, resample: bool = True):
        """Loads the given audio and resamples it if wanted"""
        audio, sample_rate = torchaudio.load(audio_path)

        if resample and sample_rate != self.sample_rate:
            resampler = T.Resample(sample_rate, self.sample_rate)
            audio = resampler(audio)
            sample_rate = self.sample_rate

        return audio, sample_rate

    @abstractmethod
    def get_data(self, audio_data_dir: str, metadata_path: str):
        raise NotImplementedError
