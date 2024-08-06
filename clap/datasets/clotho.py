import os

from pathlib import Path

import shutil

import py7zr

from glob import glob

import pandas as pd

from .audio_dataset import AudioDataset


BASE_URL = "https://zenodo.org/record/4783391/files/"
DATASET_DIR_BASE = Path(__file__).parent / "clotho"


class Clotho(AudioDataset):
    def get_samples(self):
        metadata_path = os.path.join(DATASET_DIR_BASE, f"{self.kind}.csv")
        audiodata_dir = os.path.join(DATASET_DIR_BASE, f"{self.kind}_audio")

        # Download metadata and audios if necessary
        if self.download:
            self.__download_dataset("train", DATASET_DIR_BASE / "train.csv", DATASET_DIR_BASE / "train_audio")
            self.__download_dataset("val", DATASET_DIR_BASE / "val.csv", DATASET_DIR_BASE / "val_audio")
            self.__download_dataset("test", DATASET_DIR_BASE / "test.csv", DATASET_DIR_BASE / "test_audio")

        metadata_df = pd.read_csv(metadata_path)

        audio_paths = sorted(glob(os.path.join(audiodata_dir, "*.wav")))
        captions = sum([metadata_df[metadata_df["file_name"] == os.path.basename(audio_path)]["caption"].tolist() for audio_path in audio_paths], [])

        # Generate new lists so that each audio file really belongs to 5 captions
        audio_paths_expanded = []

        for audio_path in audio_paths:
            audio_paths_expanded.extend([audio_path] * 5)

        return audio_paths_expanded, captions

    def __download_dataset(self, kind: str, metadata_path: str | Path, audiodata_dir: str | Path):
        if not os.path.exists(metadata_path):
            # Download metadata and create directory if necessary
            os.makedirs(DATASET_DIR_BASE, exist_ok=True)
            self.__download_metadata(kind, metadata_path)

            # Split the captions to create new samples
            metadata_df = pd.read_csv(metadata_path)
            metadata_df = self.split_captions(metadata_df)
            metadata_df.to_csv(metadata_path, index=False)

        if not os.path.exists(audiodata_dir):
            # Download audios and create directories if necessary
            os.makedirs(audiodata_dir, exist_ok=True)
            self.__download_audio(kind, audiodata_dir)

    def __download_audio(self, kind: str, audiodata_dir: str | Path):
        match kind:
            case "train":
                filename = "clotho_audio_development.7z"
            case "val":
                filename = "clotho_audio_validation.7z"
            case "test":
                filename = "clotho_audio_evaluation.7z"
            case _:
                raise ValueError(f"Unknown kind {kind}")

        zip_path = os.path.join(audiodata_dir, kind + ".7z")
        self.download_file(BASE_URL + filename, zip_path, f"Downloading {kind} audio data")

        # Extract downloaded zip file
        py7zr.SevenZipFile(zip_path, 'r').extractall(audiodata_dir)

        # Delete zip file
        os.remove(zip_path)

        # Remove redundant directory
        original_dir = filename.split("_")[-1][:-3]
        for audio_file in os.listdir(os.path.join(audiodata_dir, original_dir)):
            audio_file_path = os.path.join(audiodata_dir, original_dir, audio_file)
            shutil.move(audio_file_path, audiodata_dir)

        shutil.rmtree(os.path.join(audiodata_dir, original_dir))

        # Remove redundant directory
        for root, dirs, files in os.walk(audiodata_dir):
            if root == audiodata_dir:
                continue
            for file_name in files:
                source_file = os.path.join(root, file_name)
                destination_file = os.path.join(audiodata_dir, file_name)
                shutil.move(source_file, destination_file)
            # Remove the now empty nested directory
            shutil.rmtree(root)

        print(f"Downloaded {kind} audio data to {audiodata_dir}")

    def __download_metadata(self, kind: str, metadata_path: str | Path):
        match kind:
            case "train":
                filename = "clotho_captions_development.csv"
            case "val":
                filename = "clotho_captions_validation.csv"
            case "test":
                filename = "clotho_captions_evaluation.csv"
            case _:
                raise ValueError(f"Unknown kind {kind}")

        self.download_file(BASE_URL + filename, metadata_path, f"Downloading {kind} audio metadata")
        print(f"Downloaded {kind} metadata to {metadata_path}")

    @staticmethod
    def split_captions(metadata_df: pd.DataFrame) -> pd.DataFrame:
        """Split captions to create 5x the amount of data."""
        metadata_df_melted = metadata_df.melt(
            id_vars=["file_name"],
            value_vars=[
                "caption_1",
                "caption_2",
                "caption_3",
                "caption_4",
                "caption_5"
            ],
            var_name="caption_num",
            value_name="caption"
        )

        return metadata_df_melted
