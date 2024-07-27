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
        
        self.cumulative_lengths = self.compute_cumulative_lengths()

    def __getitem__(self, index: int):
        """Returns a sample given an index."""
        # Determine which dataset the index belongs to
        dataset_idx = next(i for i, cum_len in enumerate(self.cumulative_lengths) if cum_len > index)

        if dataset_idx == 0:
            local_idx = index
        else:
            local_idx = index - self.cumulative_lengths[dataset_idx - 1]

        # Get audio sample and determine if it is a longer sample that was fused
        sample = dict()
        dataset = self.data[dataset_idx]
        sample["audio"] = torch.tensor(dataset["audio"][local_idx]).to(self.device)
        sample["is_longer"] = torch.tensor(dataset["is_longer"][local_idx]).to(self.device)

        # Get the audio path
        audio_path = dataset["audio_path"][local_idx]
        
        # Decode the specific caption when retrieving it
        caption = self.captions[dataset_idx][local_idx].decode("utf-8")

        return audio_path, caption, sample

    def __len__(self):
        return self.cumulative_lengths[-1]

    def __del__(self):
        self.hdf5_file.close()

    def compute_cumulative_lengths(self):
        dataset_lengths = [len(dataset["path"]) for dataset in self.data]
        cumulative_lengths = []
        total_length = 0
        for length in dataset_lengths:
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
                "audio_path": self.hdf5_file[dataset]["audio_paths"],
                "audio": self.hdf5_file[dataset][audio_key],
                "is_longer": self.hdf5_file[dataset][is_longer_key]
            })

            # Decode the captions from bytes to UTF-8 strings
            captions.append(self.hdf5_file[dataset]["captions"])

        return data, captions

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
