from pathlib import Path

from typing import Literal, Union

import numpy as np

import torch

from .audio_processor import AudioProcessor
from .audio_dataset import AudioDataset
from .audiocaps import AudioCaps
from .clotho import Clotho
from ..utils import load_config, get_target_device


AVAILABLE_DATASETS = {"AudioCaps": AudioCaps, "Clotho": Clotho}


class ClapDataset(AudioDataset):
    def __init__(
            self,
            config: dict | Path | str,
            datasets: list[Union[Literal["AudioCaps"], Literal["Clotho"]]] = ("AudioCaps", "Clotho"),
            kind: Literal["train", "val", "test"] = "train",
            use_fusion: bool = True,
            download: bool = False
    ):
        self.config = config
        self.datasets = datasets
        self.kind = kind
        self.use_fusion = use_fusion
        self.download = download
        self.device = get_target_device()
        audio_cfg = load_config(self.config)["audio"]
        self.audio_processor = AudioProcessor(audio_cfg, self.device)

        super().__init__(self.kind, self.download)

    def __getitem__(self, index: int) -> tuple[str, str, dict[str, torch.Tensor]]:
        """Returns a sample given an index."""
        sample = dict()

        # Process the audio on demand
        audio_path = self.data[index]
        audio_processed, is_longer = self.audio_processor.process_audio(audio_path, use_fusion=self.use_fusion)

        caption = self.captions[index]

        sample["audio"] = audio_processed
        sample["is_longer"] = is_longer

        return audio_path, caption, sample

    def get_samples(self) -> tuple[list[str], list[str]] | tuple[list[dict[str, str | np.ndarray | bool]], list[str]]:
        if not self.__is_valid():
            raise ValueError(f"Datasets are not valid.\nAvailable datasets: {list(AVAILABLE_DATASETS.keys())}")

        data = []
        captions = []

        for dataset_name in self.datasets:
            audio_dataset = AVAILABLE_DATASETS[dataset_name](self.kind, self.download)

            data.extend(audio_dataset.data)
            captions.extend(audio_dataset.captions)

        return data, captions

    def __is_valid(self):
        for dataset in self.datasets:
            result = AVAILABLE_DATASETS.get(dataset)
            if result is None:
                return False

        return True
