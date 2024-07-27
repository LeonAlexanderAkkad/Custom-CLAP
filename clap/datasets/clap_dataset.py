import os.path
from pathlib import Path

import random

from typing import Literal, Union

import numpy as np

import torch
import torchvision
import torchaudio
import torchaudio.transforms as T

import h5py
from tqdm import tqdm

from .audio_processor import AudioProcessor
from .audiocaps import AudioCaps
from .clotho import Clotho
from .audio_dataset import AudioDataset
from ..utils import load_config, get_target_device


AVAILABLE_DATASETS = {"AudioCaps": AudioCaps, "Clotho": Clotho}


class ClapDataset(AudioDataset):
    def __init__(
            self,
            datasets: list[Union[Literal["AudioCaps"], Literal["Clotho"]]] = ("AudioCaps", "Clotho"),
            kind: Literal["train", "val", "test"] = "train",
            preprocess_audio: bool = False,
            config: dict | Path | str | None = None,
            use_fusion: bool = True,
            download: bool = False
    ):
        self.datasets = datasets
        self.preprocess_audio = preprocess_audio
        self.config = config
        self.device = get_target_device()
        self.hdf5_file = None
        self.use_fusion = use_fusion
        super().__init__(kind, download)

        self.dataset_lengths = [len(dataset["path"]) for dataset in self.data]
        self.cumulative_lengths = self.compute_cumulative_lengths()

    def __getitem__(self, index: int):
        """Returns file given an index."""
        # Determine which dataset the index belongs to
        dataset_idx = next(i for i, cum_len in enumerate(self.cumulative_lengths) if cum_len > index)

        if dataset_idx == 0:
            local_idx = index
        else:
            local_idx = index - self.cumulative_lengths[dataset_idx - 1]

        dataset = self.data[dataset_idx]
        audio = dataset["audio"][local_idx]
        is_longer = dataset["is_longer"][local_idx]
        caption = self.captions[index]

        return audio, is_longer, caption

    def compute_cumulative_lengths(self):
        cumulative_lengths = []
        total_length = 0
        for length in self.dataset_lengths:
            total_length += length
            cumulative_lengths.append(total_length)
        return cumulative_lengths

    def get_data(self):
        data = []
        captions = []

        if self.preprocess_audio:
            self.__create_hdf5()

        self.hdf5_file = h5py.File(os.path.join("clap", "datasets", "audio_data.h5"), "r")

        audio_key = "audio_fused" if self.use_fusion else "audio"
        is_longer_key = "is_longer_fused" if self.use_fusion else "is_longer"

        for dataset in self.datasets:
            dataset = dataset.lower()
            data.append({
                "path": self.hdf5_file[dataset]["audio_paths"],
                "audio": self.hdf5_file[dataset][audio_key],
                "is_longer": self.hdf5_file[dataset][is_longer_key]
            })

            # Decode the captions from bytes to UTF-8 strings
            captions.extend([caption.decode("utf-8") for caption in self.hdf5_file[dataset]["captions"]])

        return data, captions

    def __del__(self):
        self.hdf5_file.close()

    # def load_sample(self, audio_path: str, use_fusion) -> dict[str, torch.Tensor]:
    #     """Method for loading the given sample such that it can be used for training."""
    #     # Get the audio and reshape it into mono.
    #     audio, sample_rate = self.load_audio(audio_path, self.sampling_rate)
    #     audio = audio.reshape(-1)
    #     audio_len = len(audio)
    #
    #     max_len = self.audio_cfg["duration"] * sample_rate
    #     audio_sample = dict()
    #
    #     # Audio is too long and needs to be cropped or fused.
    #     if audio_len > max_len:
    #         audio_mel, is_longer = self.truncate_audio(audio, max_len, use_fusion)
    #     # Audio is too short and most be extended.
    #     elif audio_len < max_len:
    #         audio_mel, is_longer = self.pad_audio(audio, max_len, use_fusion)
    #     # Audio is neither too short nor too long
    #     else:
    #         audio_mel = self.get_mel_spectrogram(audio)
    #         if use_fusion:
    #             # Needed for batching
    #             padding = torch.empty_like(audio_mel, device=audio_mel.device)
    #             audio_mel = torch.stack([audio_mel, padding, padding, padding], dim=0)
    #         else:
    #             # Needed for unified processing in the model
    #             audio_mel = audio_mel.unsqueeze(0)
    #         is_longer = False
    #
    #     audio_sample["is_longer"] = torch.tensor(is_longer).to(self.device)
    #     audio_sample["audio"] = audio_mel
    #
    #     return audio_sample
    #
    # def truncate_audio(self, audio: torch.Tensor, max_len: int, use_fusion: bool) -> tuple[torch.Tensor, bool]:
    #     if use_fusion:
    #         # Feature Fusion
    #         audio_mel = self.get_mel_spectrogram(audio)
    #         # Split the spectrogram into three parts
    #         chunk_frames = max_len // self.audio_cfg['hop_size'] + 1  # The +1 is related to the spectrogram computation
    #         total_frames = audio_mel.shape[0]
    #         if chunk_frames == total_frames:
    #             # there is a corner case where the audio length is
    #             # larger than max_len but smaller than max_len+hop_size
    #             # In this case, we just use the whole audio
    #             padding = torch.empty_like(audio_mel, device=audio_mel.device)
    #             audio_mel = torch.stack([audio_mel, padding, padding, padding], dim=0)
    #             is_longer = False
    #         else:
    #             ranges = np.array_split(list(range(0, total_frames - chunk_frames + 1)), 3)
    #
    #             # If the audio is too short, use the first chunk
    #             if len(ranges[1]) == 0:
    #                 ranges[1] = np.array([0])
    #             if len(ranges[2]) == 0:
    #                 ranges[2] = np.array([0])
    #
    #             # Randomly choose the index for each part
    #             idx_front = np.random.choice(ranges[0])
    #             idx_middle = np.random.choice(ranges[1])
    #             idx_back = np.random.choice(ranges[2])
    #
    #             # Select the mel for each chunk
    #             audio_mel_chunk_front = audio_mel[idx_front:idx_front + chunk_frames, :]
    #             audio_mel_chunk_middle = audio_mel[idx_middle:idx_middle + chunk_frames, :]
    #             audio_mel_chunk_back = audio_mel[idx_back:idx_back + chunk_frames, :]
    #
    #             # Shrink the mel
    #             audio_mel_shrink = torchvision.transforms.Resize(
    #                 size=[chunk_frames, self.audio_cfg['mel_bins']]
    #             )(audio_mel.unsqueeze(0)).squeeze(0)
    #
    #             # Finally, stack the mel chunks
    #             audio_mel = torch.stack(
    #                 [audio_mel_shrink, audio_mel_chunk_front, audio_mel_chunk_middle, audio_mel_chunk_back], dim=0
    #             )
    #             is_longer = True
    #     else:
    #         start_index = random.randrange(len(audio) - max_len)
    #         audio = audio[start_index:start_index + max_len]
    #         audio_mel = self.get_mel_spectrogram(audio).unsqueeze(0)
    #         is_longer = False
    #
    #     return audio_mel, is_longer
    #
    # def pad_audio(self, audio: torch.Tensor, max_len: int, use_fusion) -> tuple[torch.Tensor, bool]:
    #     # Simply repeat the audio.
    #     repeat_factor = int(np.ceil(max_len / len(audio)))
    #     # Remove excess part of audio_time_series.
    #     audio = audio.repeat(repeat_factor)[0:max_len]
    #     audio_mel = self.get_mel_spectrogram(audio)
    #
    #     if use_fusion:
    #         # Needed for batching
    #         padding = torch.empty_like(audio_mel, device=audio_mel.device)
    #         audio_mel = torch.stack([audio_mel, padding, padding, padding], dim=0)
    #     else:
    #         # Needed for unified processing in the model
    #         audio_mel = audio_mel.unsqueeze(0)
    #
    #     is_longer = False
    #
    #     return audio_mel, is_longer
    #
    # def load_audio(self, audio_path: str, target_sampling_rate, resample: bool = True):
    #     """Loads the given audio and resamples it if wanted"""
    #     audio, sampling_rate = torchaudio.load(audio_path)
    #
    #     if resample and sampling_rate != target_sampling_rate:
    #         resampler = T.Resample(sampling_rate, target_sampling_rate)
    #         audio = resampler(audio)
    #         sampling_rate = target_sampling_rate
    #
    #     return audio.to(self.device), sampling_rate
    #
    # def get_mel_spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
    #     """Returns the log-mel spectrogram of the given audio."""
    #     audio_spec = self.mel_extractor(audio)
    #     audio_mel = self.log_mel_extractor(audio_spec)
    #
    #     return audio_mel.T

    def __create_hdf5(self):
        if self.config is None:
            raise ValueError("Config needs to be set when preprocessing audio data.")

        audio_cfg = load_config(self.config)["audio"]

        audio_processor = AudioProcessor(audio_cfg)

        with h5py.File(os.path.join("clap", "datasets", f"audio_data{self.kind}.h5"), "w") as h5file:
            for name, dataset in AVAILABLE_DATASETS.items():
                group = h5file.create_group(name.lower())
                audio_dataset = dataset(self.kind, self.download)

                audio_shape_fused = audio_processor.process_audio(audio_dataset.data[0], use_fusion=True)["audio"].shape
                audio_shape = audio_processor.process_audio(audio_dataset.data[0], use_fusion=False)["audio"].shape

                dataset_audio_paths = group.create_dataset(
                    "audio_paths",
                    shape=len(audio_dataset),
                    dtype=h5py.string_dtype("utf-8")
                )

                dataset_audio_fused = group.create_dataset(
                    "audio_fused",
                    shape=(len(audio_dataset),) + audio_shape_fused,
                    dtype=float
                )

                dataset_is_longer_fused = group.create_dataset(
                    "is_longer_fused",
                    shape=len(audio_dataset),
                    dtype=bool
                )

                dataset_audio = group.create_dataset(
                    "audio",
                    shape=(len(audio_dataset),) + audio_shape,
                    dtype=float
                )

                dataset_is_longer = group.create_dataset(
                    "is_longer",
                    shape=len(audio_dataset),
                    dtype=bool
                )

                dataset_captions = group.create_dataset(
                    "captions",
                    shape=len(audio_dataset),
                    dtype=h5py.string_dtype("utf-8")
                )

                for i, (audio_path, caption) in tqdm(enumerate(zip(audio_dataset.data, audio_dataset.captions)), total=len(audio_dataset), desc=f"Processing {name}"):
                    audio_fused = audio_processor.process_audio(audio_path, use_fusion=True)
                    audio = audio_processor.process_audio(audio_path, use_fusion=False)

                    dataset_audio_paths[i] = audio_path.encode("utf-8")
                    dataset_audio_fused[i, :, :, :] = audio_fused["audio"]
                    dataset_is_longer_fused[i] = audio_fused["is_longer"]
                    dataset_audio[i, :, :, :] = audio["audio"]
                    dataset_is_longer[i] = audio["is_longer"]
                    dataset_captions[i] = caption.encode("utf-8")
