from pathlib import Path

from typing import Literal, Union

import torch

from .audio_processor import AudioProcessor
from .audio_dataset import AudioDataset
from .audiocaps import AudioCaps
from .clotho import Clotho
from ..utils import load_config, get_target_device


AVAILABLE_DATASETS = {"AudioCaps": AudioCaps, "Clotho": Clotho}


class ClapDataset(AudioDataset):
    """A dataset for handling audio data from multiple sources for either training, validation or testing.

    Attributes
    ----------
    config : dict or Path or str
        Configuration for the dataset.
    datasets : list[Union[Literal["AudioCaps"], Literal["Clotho"]]]
        List of datasets to be used.
    kind : str
        Type of dataset split.
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
            config: dict | Path | str,
            datasets: list[Union[Literal["AudioCaps"], Literal["Clotho"]]] = ("AudioCaps", "Clotho"),
            kind: Literal["train", "val", "test"] = "train",
            use_fusion: bool = True,
            download: bool = False
    ):
        """ Initializes the ClapDataset.

        Parameters
        ----------
        config : dict or Path or str
            Configuration for loading audio data and processing.
        datasets : list[Union[Literal["AudioCaps"], Literal["Clotho"]]], optional
            List of datasets to be used, by default ["AudioCaps", "Clotho"].
        kind : Literal["train", "val", "test"], optional
            Type of dataset split, by default "train".
        use_fusion : bool, optional
            Whether to use fusion processing for audio, by default True.
        download : bool, optional
            Whether to download the datasets if not available, by default False.
        """
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
        """Retrieves a sample from the dataset given an index.

        Parameters
        ----------
        index : int
            Index of the sample to be retrieved.

        Returns
        -------
        tuple[str, str, dict[str, torch.Tensor]]
            A tuple containing the audio path, caption, and a dictionary with processed audio and
            a flag indicating if the audio is longer.
        """
        sample = dict()

        # Process the audio on demand
        audio_path = self.data[index]
        audio_processed, is_longer = self.audio_processor.process_audio(audio_path, use_fusion=self.use_fusion)

        caption = self.captions[index]

        sample["audio"] = audio_processed
        sample["is_longer"] = is_longer

        return audio_path, caption, sample

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
            audio_dataset = AVAILABLE_DATASETS[dataset_name](self.kind, self.download)

            data.extend(audio_dataset.data)
            captions.extend(audio_dataset.captions)

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
