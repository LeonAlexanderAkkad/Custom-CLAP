import os.path
from pathlib import Path

from typing import Literal, Union

import numpy as np

import torch
from torch.utils.data import Dataset

import h5py
from tqdm import tqdm

from .audio_processor import AudioProcessor
from .audiocaps import AudioCaps
from .clotho import Clotho
from ..utils import load_config, get_target_device


AVAILABLE_DATASETS = {"AUDIOCAPS": AudioCaps, "CLOTHO": Clotho}


class ClapDataset(Dataset):
    def __init__(
            self,
            datasets: tuple[Union[Literal["AudioCaps"], Literal["Clotho"]]] = ("AudioCaps", "Clotho"),
            kind: Literal["train", "val", "test"] = "train",
            use_hdf5: bool = True,
            config: dict | Path | str | None = None,
            use_fusion: bool = True,
            download: bool = False
    ):
        self.datasets = datasets
        self.kind = kind
        self.use_hdf5 = use_hdf5
        self.config = config
        self.use_fusion = use_fusion
        self.download = download
        self.device = get_target_device()

        # Decide which mode to use: Either use HDF5 file or use on demand audio preprocessing
        if self.use_hdf5:
            self.hdf5_path = os.path.join("clap", "datasets", f"audio_data_{self.kind}.h5")
            self.__create_hdf5_if_necessary()
            self.hdf5_file = h5py.File(self.hdf5_path, "r")
        else:
            if self.config is None:
                raise ValueError("Config file must be provided when preprocessing audio on demand")
            audio_cfg = load_config(self.config)["audio"]
            self.audio_processor = AudioProcessor(audio_cfg, self.device)

        self.data, self.captions = self.__get_sample()

    def __getitem__(self, index: int) -> tuple[str, str, dict[str, torch.Tensor]]:
        """Returns a sample given an index."""
        sample = dict()
        if self.use_hdf5:
            # Compute cumulative lengths for determining to which dataset the index belongs to
            cumulative_lengths = self.__compute_cumulative_lengths()

            # Determine which dataset the index belongs to
            dataset_idx = next(i for i, cum_len in enumerate(cumulative_lengths) if cum_len > index)

            # Get local index for the respective dataset
            if dataset_idx == 0:
                local_idx = index
            else:
                local_idx = index - cumulative_lengths[dataset_idx - 1]

            # Get audio sample and determine if it is a longer sample that was fused
            dataset = self.data[dataset_idx]
            audio_processed = torch.tensor(dataset["audio"][local_idx]).to(self.device)
            is_longer = torch.tensor(dataset["is_longer"][local_idx]).to(self.device)
            audio_path = dataset["audio_path"][local_idx].decode("utf-8")
        else:
            # Process the audio on demand
            audio_path = self.data[index]
            audio_processed, is_longer = self.audio_processor.process_audio(audio_path, use_fusion=self.use_fusion)

        caption = self.captions[index]

        sample["audio"] = audio_processed
        sample["is_longer"] = is_longer

        return audio_path, caption, sample

    def __len__(self):
        if self.use_hdf5:
            return len(self.data[0]["audio_path"] + self.data[1]["audio_path"])

        return len(self.data)

    def __del__(self):
        if self.use_hdf5:
            self.hdf5_file.close()

    def __get_sample(self) -> tuple[list[str], list[str]] | tuple[list[dict[str, str | np.ndarray | bool]], list[str]]:
        if not self.__is_valid():
            raise ValueError(f"Datasets are not valid.\nAvailable datasets: {list(AVAILABLE_DATASETS.keys())}")

        data = []
        captions = []

        for dataset_name in self.datasets:
            audio_dataset = AVAILABLE_DATASETS[dataset_name.upper()](self.kind, self.download)
            if self.use_hdf5:
                dataset_name = dataset_name.lower()
                audio_key = "audio_fused" if self.use_fusion else "audio"
                is_longer_key = "is_longer_fused" if self.use_fusion else "is_longer"

                data.append({
                    "audio_path": self.hdf5_file[dataset_name]["audio_paths"],
                    "audio": self.hdf5_file[dataset_name][audio_key],
                    "is_longer": self.hdf5_file[dataset_name][is_longer_key]
                })
            else:
                data.extend(audio_dataset.data)

            captions.extend(audio_dataset.captions)

            # Decode the captions from bytes to UTF-8 strings
            captions.append(self.hdf5_file[dataset_name]["captions"])

        return data, captions

    def __create_hdf5_if_necessary(self):
        if not os.path.exists(self.hdf5_path):
            if self.config is None:
                raise ValueError("Config needs to be set when preparing audio data for hdf5 file.")

            audio_cfg = load_config(self.config)["audio"]

            audio_processor = AudioProcessor(audio_cfg)

            with h5py.File(self.hdf5_path, "w") as h5file:
                for name, dataset in AVAILABLE_DATASETS.items():
                    group = h5file.create_group(name.lower())
                    audio_dataset = dataset(self.kind, self.download)

                    audio_shape_fused = audio_processor.process_audio(audio_dataset.data[0], use_fusion=True)[0].shape
                    audio_shape = audio_processor.process_audio(audio_dataset.data[0], use_fusion=False)[0].shape

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

                    for i, audio_path in tqdm(enumerate(audio_dataset.data), total=len(audio_dataset), desc=f"Processing {name}"):
                        audio_fused, is_longer_fused = audio_processor.process_audio(audio_path, use_fusion=True)
                        audio, is_longer = audio_processor.process_audio(audio_path, use_fusion=False)

                        dataset_audio_paths[i] = audio_path.encode("utf-8")
                        dataset_audio_fused[i, :, :, :] = audio_fused
                        dataset_is_longer_fused[i] = is_longer_fused
                        dataset_audio[i, :, :, :] = audio
                        dataset_is_longer[i] = is_longer

    def __compute_cumulative_lengths(self):
        """Computes the cumulative lengths of the audio dataset."""
        dataset_lengths = [len(dataset["audio_path"]) for dataset in self.data]
        cumulative_lengths = []
        total_length = 0
        for length in dataset_lengths:
            total_length += length
            cumulative_lengths.append(total_length)

        return cumulative_lengths

    def __is_valid(self):
        for dataset in self.datasets:
            result = AVAILABLE_DATASETS.get(dataset.upper())
            if result is None:
                return False

        return True
