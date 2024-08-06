from pathlib import Path

import json

from typing import Literal, Union

import torch
from torch.utils.data import Dataset

from .audio_processor import AudioProcessor
from .audiocaps import AudioCaps
from .clotho import Clotho
from .esc50 import ESC50
from ..utils import get_target_device


AVAILABLE_DATASETS = {"AudioCaps": AudioCaps, "Clotho": Clotho, "ESC50": ESC50}


class ClapDataset(Dataset):
    """A dataset for handling audio data from multiple sources for either training, validation or testing.

    Attributes
    ----------
    config : dict or Path or str
        Configuration for the dataset.
    datasets : list[Union[Literal["AudioCaps"], Literal["Clotho"]]]
        List of datasets to be used.
    kinds : list[Union[Literal["train"], Literal["val"], Literal["test"]]]
        Type of dataset splits.
    use_fusion : bool
        Flag indicating whether to use fusion audio pre-processing.
    download : bool
        Flag indicating whether to download datasets.
    device : str
        Device used for processing (CPU or GPU).
    audio_processor : AudioProcessor
        Instance of AudioProcessor for processing audio data.
    """
    def __init__(
            self,
            config: dict,
            datasets: list[Union[Literal["AudioCaps"], Literal["Clotho"], Literal["ESC50"]]] | None = None,
            kinds: list[Union[Literal["train"], Literal["val"], Literal["test"]]] | None = None,
            download: bool = False
    ):
        """ Initializes the ClapDataset.

        Parameters
        ----------
        config : dict or Path or str
            Configuration for loading audio data and processing.
        datasets : list[Union[Literal["AudioCaps"], Literal["Clotho"]]], optional
            List of datasets to be used, by default ["AudioCaps", "Clotho"].
        kinds : Literal["train", "val", "test"], optional
            Type of dataset split, by default "train".
        use_fusion : bool, optional
            Whether to use fusion processing for audio, by default True.
        download : bool, optional
            Whether to download the datasets if not available, by default False.
        """
        self.config = config
        self.datasets = ["AudioCaps", "Clotho"] if datasets is None else datasets
        self.kinds = ["train"] if kinds is None else kinds

        self.download = download
        self.device = get_target_device()
        self.audio_cfg = config["audio"]
        self.use_fusion = self.audio_cfg["use_fusion"]
        self.audio_processor = AudioProcessor(self.audio_cfg, self.device)
        self.audio, self.text = self.get_samples()

    def __getitem__(self, index: int) -> tuple[str, str | int, dict[str, torch.Tensor]]:
        """Retrieves a sample from the dataset given an index.

        Parameters
        ----------
        index : int
            Index of the sample to be retrieved.

        Returns
        -------
        tuple[str, str | int, dict[str, torch.Tensor]]
            A tuple containing the audio path, caption / class, and a dictionary with processed audio and
            a flag indicating if the audio is longer.
        """
        audio_sample = dict()

        # Process the audio on demand
        audio_path = self.audio[index]
        audio_processed, is_longer = self.audio_processor.process_audio(audio_path, use_fusion=self.use_fusion)

        text = self.text[index]

        audio_sample["audio"] = audio_processed
        audio_sample["is_longer"] = is_longer

        return audio_path, text, audio_sample

    def __len__(self):
        return len(self.text)

    def get_samples(self) -> tuple[list[str], list[str]]:
        """Retrieves all samples from the dataset.

        Returns
        -------
        tuple[list[str], list[str]]
            A tuple containing a list of audio paths and a list of captions.

        Raises
        ------
        ValueError
            If datasets are invalid.
        """
        if not self.__is_valid():
            raise ValueError(f"Datasets are not valid.\nAvailable datasets: {list(AVAILABLE_DATASETS.keys())}")

        data = []
        captions = []

        for dataset_name in self.datasets:
            for kind in self.kinds:
                audio_dataset = AVAILABLE_DATASETS[dataset_name](kind, self.download)

                data.extend(audio_dataset.audio)
                captions.extend(audio_dataset.text)

        return data, captions

    def __is_valid(self):
        """Validates the datasets to be used.

        Returns
        -------
        bool
            True if all dataset names are valid, False otherwise.
        """
        for dataset in self.datasets:
            result = AVAILABLE_DATASETS.get(dataset)
            if result is None:
                return False

        return True

    @staticmethod
    def load_esc50_class_mapping() -> tuple[dict[str, int], dict[str, str]]:
        """Loads the ESC50 class mapping (i.e. class to idx and idx to class mappings)."""
        class_to_idx = json.load(open(Path(__file__).parent / "esc50" / "class_to_idx.json", "r"))
        idx_to_class = json.load(open(Path(__file__).parent / "esc50" / "idx_to_class.json", "r"))

        return class_to_idx, idx_to_class
