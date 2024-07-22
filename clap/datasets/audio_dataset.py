import os

from abc import ABC, abstractmethod

from pathlib import Path

from typing import Literal

import numpy as np

import torch
import torchvision
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T

from ..utils import load_config, get_target_device


class AudioDataset(Dataset, ABC):
    """Simple abstract audio dataset used for training."""

    def __init__(
            self,
            audiodata_dir: str,
            metadata_dir: str,
            audio_cfg: dict | Path | str,
            kind: Literal["train", "val", "test"] = "train",
            download: bool = False,
    ):
        """Initialize the audio dataset and download it if needed."""
        self.audio_cfg = load_config(audio_cfg)["audio"]
        self.sampling_rate = self.audio_cfg["sampling_rate"]
        self.device = get_target_device()

        self.mel_extractor = T.MelSpectrogram(
            sample_rate=self.sampling_rate,
            n_fft=self.audio_cfg["window_size"],
            win_length=self.audio_cfg["window_size"],
            hop_length=self.audio_cfg["hop_size"],
            center=True,
            pad_mode="reflect",
            power=2,
            norm=None,
            onesided=True,
            n_mels=self.audio_cfg["mel_bins"],
            f_min=self.audio_cfg["f_min"],
            f_max=self.audio_cfg["f_max"]
        ).to(self.device)

        self.log_mel_extractor = T.AmplitudeToDB(top_db=None).to(self.device)

        self.kind = kind
        self.download = download
        self.data, self.captions = self.get_data(os.path.abspath(audiodata_dir), os.path.abspath(metadata_dir))

    def __getitem__(self, index: int):
        """Returns file given an index."""
        audio_path = self.data[index]
        audio = self.load_sample(audio_path)
        caption = self.captions[index]

        return audio_path, caption, audio

    def __len__(self):
        """Returns the number of files in the dataset."""
        return len(self.data)

    def load_sample(self, audio_path: str) -> dict[str, torch.Tensor]:
        """Method for loading the given sample such that it can be used for training."""
        # Get the audio and reshape it into mono.
        audio, sample_rate = self.load_audio(audio_path)
        audio = audio.reshape(-1)

        max_len = self.audio_cfg["duration"] * sample_rate
        audio_sample = dict()

        # Audio is too long and needs to be cropped or fused.
        if len(audio) > max_len:
            # TODO: Implement fusion.
            # Feature Fusion
            audio_mel = self.get_mel_spectrogram(audio)
            # Split the spectrogram into three parts
            chunk_frames = max_len // self.audio_cfg[
                'hop_size'] + 1  # the +1 related to how the spectrogram is computed
            total_frames = audio_mel.shape[0]
            if chunk_frames == total_frames:
                # there is a corner case where the audio length is
                # larger than max_len but smaller than max_len+hop_size
                # In this case, we just use the whole audio
                audio_mel_fusion = torch.stack([audio_mel, audio_mel, audio_mel, audio_mel], dim=0)
                is_longer = False
            else:
                ranges = np.array_split(list(range(0, total_frames - chunk_frames + 1)), 3)

                # If the audio is too short, use the first chunk
                if len(ranges[1]) == 0:
                    ranges[1] = [0]
                if len(ranges[2]) == 0:
                    ranges[2] = [0]

                # Randomly choose the index for each part
                idx_front = np.random.choice(ranges[0])
                idx_middle = np.random.choice(ranges[1])
                idx_back = np.random.choice(ranges[2])

                # Select the mel for each chung
                audio_mel_chunk_front = audio_mel[idx_front:idx_front + chunk_frames, :]
                audio_mel_chunk_middle = audio_mel[idx_middle:idx_middle + chunk_frames, :]
                audio_mel_chunk_back = audio_mel[idx_back:idx_back + chunk_frames, :]

                # Shrink the mel
                audio_mel_shrink = \
                torchvision.transforms.Resize(size=[chunk_frames, self.audio_cfg['mel_bins']])(audio_mel[None])[0]

                # Finally, stack the mel chunks
                audio_mel_fusion = torch.stack(
                    [audio_mel_shrink, audio_mel_chunk_front, audio_mel_chunk_middle, audio_mel_chunk_back], dim=0)
                is_longer = True
        else:
            # Audio is too short and most be extended.
            if len(audio) < max_len:
                # Simply repeat the audio.
                repeat_factor = int(np.ceil(max_len / len(audio)))
                # Remove excess part of audio_time_series.
                audio = audio.repeat(repeat_factor)[0:max_len]

            # Audio has exact length
            audio_mel = self.get_mel_spectrogram(audio)
            audio_mel_fusion = torch.stack([audio_mel, audio_mel, audio_mel, audio_mel], dim=0)
            is_longer = False

        audio_sample["is_longer"] = torch.tensor(is_longer).to(self.device)
        audio_sample["fusion"] = audio_mel_fusion

        # start_index = random.randrange(len(audio) - max_len)
        # audio = audio[start_index:start_index + max_len]

        return audio_sample

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

    @abstractmethod
    def get_data(self, audio_data_dir: str, metadata_dir: str):
        raise NotImplementedError
