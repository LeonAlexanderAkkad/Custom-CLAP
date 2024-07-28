import random

from pathlib import Path

import numpy as np

import torch
import torchvision
import torchaudio
import torchaudio.transforms as T


class AudioProcessor:
    def __init__(self, audio_cfg: dict, device: torch.device = torch.device('cpu')):
        self.audio_cfg = audio_cfg
        self.device = device
        self.sampling_rate = self.audio_cfg["sampling_rate"]

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

    def process_audio(self, audio_path: Path | str, use_fusion) -> tuple[torch.Tensor, torch.Tensor]:
        """Method for loading the given sample such that it can be used for training."""
        # Get the audio and reshape it into mono.
        audio, sample_rate = self.__load_audio(audio_path, self.sampling_rate)
        audio = audio.reshape(-1)
        audio_len = len(audio)

        max_len = self.audio_cfg["duration"] * sample_rate

        # Audio is too long and needs to be cropped or fused.
        if audio_len > max_len:
            audio_mel, is_longer = self.__truncate_audio(audio, max_len, use_fusion)
        # Audio is too short and most be extended.
        elif audio_len < max_len:
            audio_mel, is_longer = self.__pad_audio(audio, max_len, use_fusion)
        # Audio is neither too short nor too long
        else:
            audio_mel = self.__get_mel_spectrogram(audio)
            if use_fusion:
                # Needed for batching
                padding = torch.empty_like(audio_mel, device=audio_mel.device)
                audio_mel = torch.stack([audio_mel, padding, padding, padding], dim=0)
            else:
                # Needed for unified processing in the model
                audio_mel = audio_mel.unsqueeze(0)
            is_longer = False

        return audio_mel.to(self.device), torch.tensor(is_longer).to(self.device)

    def __truncate_audio(self, audio: torch.Tensor, max_len: int, use_fusion: bool) -> tuple[torch.Tensor, bool]:
        if use_fusion:
            # Feature Fusion
            audio_mel = self.__get_mel_spectrogram(audio)
            # Split the spectrogram into three parts
            chunk_frames = max_len // self.audio_cfg['hop_size'] + 1  # The +1 is related to the spectrogram computation
            total_frames = audio_mel.shape[0]
            if chunk_frames == total_frames:
                # there is a corner case where the audio length is
                # larger than max_len but smaller than max_len+hop_size
                # In this case, we just use the whole audio
                padding = torch.empty_like(audio_mel, device=audio_mel.device)
                audio_mel = torch.stack([audio_mel, padding, padding, padding], dim=0)
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

                # Select the mel for each chunk
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
            audio_mel = self.__get_mel_spectrogram(audio).unsqueeze(0)
            is_longer = False

        return audio_mel, is_longer

    def __pad_audio(self, audio: torch.Tensor, max_len: int, use_fusion) -> tuple[torch.Tensor, bool]:
        # Simply repeat the audio.
        repeat_factor = int(np.ceil(max_len / len(audio)))
        # Remove excess part of audio_time_series.
        audio = audio.repeat(repeat_factor)[0:max_len]
        audio_mel = self.__get_mel_spectrogram(audio)

        if use_fusion:
            # Needed for batching
            padding = torch.empty_like(audio_mel, device=audio_mel.device)
            audio_mel = torch.stack([audio_mel, padding, padding, padding], dim=0)
        else:
            # Needed for unified processing in the model
            audio_mel = audio_mel.unsqueeze(0)

        is_longer = False

        return audio_mel, is_longer

    def __get_mel_spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        """Returns the log-mel spectrogram of the given audio."""
        audio_spec = self.mel_extractor(audio)
        audio_mel = self.log_mel_extractor(audio_spec)

        return audio_mel.T

    @staticmethod
    def __load_audio(audio_path: Path | str, target_sampling_rate, resample: bool = True):
        """Loads the given audio and resamples it if wanted"""
        audio, sampling_rate = torchaudio.load(audio_path)

        if resample and sampling_rate != target_sampling_rate:
            resampler = T.Resample(sampling_rate, target_sampling_rate)
            audio = resampler(audio)
            sampling_rate = target_sampling_rate

        return audio, sampling_rate
