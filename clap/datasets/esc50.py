import os

from pathlib import Path

import shutil

import json

from zipfile import ZipFile

from glob import glob

import pandas as pd

from typing import Literal

from .audio_dataset import AudioDataset


BASE_URL = "https://github.com/karolpiczak/ESC-50/"
DATASET_DIR_BASE = Path(__file__).parent / "esc50"


class ESC50(AudioDataset):
    def get_samples(self):
        metadata_path = os.path.join(DATASET_DIR_BASE, f"{self.kind}.csv")
        audiodata_dir = os.path.join(DATASET_DIR_BASE, f"{self.kind}_audio")

        # Download metadata and audios if necessary
        if self.download:
            self.__download_dataset()

        metadata_df = pd.read_csv(metadata_path)

        audio_paths = sorted(glob(os.path.join(audiodata_dir, "*.wav")))
        categories = [metadata_df[metadata_df["filename"] == os.path.basename(audio_path)]["target"].item() for audio_path in audio_paths]

        return audio_paths, categories

    def __download_dataset(self):
        # Download metadata, create directory and split the metadata into train, validation and test metadata
        metadata_download_path = os.path.join(DATASET_DIR_BASE, "metadata.csv")
        os.makedirs(DATASET_DIR_BASE, exist_ok=True)
        self.__download_metadata(metadata_download_path)
        metadata_df = pd.read_csv(metadata_download_path)
        metadata_df["category"] = metadata_df["category"].str.replace("_", " ")

        # Save class to idx mapping and idx to class mappings
        metadata_df_targets = metadata_df.drop_duplicates(subset="category", keep="first").sort_values(by="target")
        class_to_idx = dict(zip(metadata_df_targets["category"], metadata_df_targets["target"]))
        idx_to_class = dict(zip(metadata_df_targets["target"], metadata_df_targets["category"]))
        with open(DATASET_DIR_BASE / "class_to_idx.json", 'w') as f:
            json.dump(class_to_idx, f, indent=4)
        with open(DATASET_DIR_BASE / "idx_to_class.json", 'w') as f:
            json.dump(idx_to_class, f, indent=4)

        self.__split_metadata(metadata_df)
        os.remove(metadata_download_path)

        # Download audios and create directories if necessary
        audiodata_download_path = os.path.join(DATASET_DIR_BASE, "audio")
        os.makedirs(audiodata_download_path, exist_ok=True)
        self.__download_audio(audiodata_download_path)
        self.__split_audios()

    def __download_audio(self, audiodata_dir: str):
        zip_path = os.path.join(audiodata_dir, "audio.zip")
        self.download_file(BASE_URL + "archive/master.zip", zip_path, f"Downloading audio data")

        # Extract downloaded zip file
        ZipFile(zip_path, 'r').extractall(audiodata_dir)

        # Delete zip file
        os.remove(zip_path)

        for audio_file in os.listdir(os.path.join(audiodata_dir, "ESC-50-master", "audio")):
            audio_file_path = os.path.join(audiodata_dir, "ESC-50-master", "audio", audio_file)
            shutil.move(audio_file_path, audiodata_dir)

        # Remove redundant files
        shutil.rmtree(os.path.join(audiodata_dir, "ESC-50-master"))

        print(f"Downloaded audio data to {audiodata_dir}")

    def __split_audios(self):
        # Create directories for split audio files
        os.makedirs(DATASET_DIR_BASE / "train_audio", exist_ok=True)
        os.makedirs(DATASET_DIR_BASE / "val_audio", exist_ok=True)
        os.makedirs(DATASET_DIR_BASE / "test_audio", exist_ok=True)

        train_metadata_df = pd.read_csv(DATASET_DIR_BASE / "train.csv")
        val_metadata_df = pd.read_csv(DATASET_DIR_BASE / "val.csv")
        test_metadata_df = pd.read_csv(DATASET_DIR_BASE / "test.csv")

        # Move each audio file to the correct folder
        for audio_name in train_metadata_df["filename"]:
            shutil.move(DATASET_DIR_BASE / "audio" / audio_name, DATASET_DIR_BASE / "train_audio" / audio_name)

        for audio_name in val_metadata_df["filename"]:
            shutil.move(DATASET_DIR_BASE / "audio" / audio_name, DATASET_DIR_BASE / "val_audio" / audio_name)

        for audio_name in test_metadata_df["filename"]:
            shutil.move(DATASET_DIR_BASE / "audio" / audio_name, DATASET_DIR_BASE / "test_audio" / audio_name)

        # Remove the now empty audio folder
        shutil.rmtree(DATASET_DIR_BASE / "audio")

    def __download_metadata(self, metadata_path):
        filename = "raw/master/meta/esc50.csv"

        self.download_file(BASE_URL + filename, metadata_path, f"Downloading {self.kind} metadata")
        print(f"Downloaded metadata to {metadata_path}")

    @staticmethod
    def __split_metadata(metadata_df: pd.DataFrame):
        """Split the metadata dataframe into train, validation and test set."""
        # Select the first 4 folds as train data and split the 5th fold for validation and test data
        train_metadata_df = metadata_df.iloc[:1600]
        val_metadata_df = metadata_df.iloc[1600:1750]
        test_metadata_df = metadata_df.iloc[1750:]

        # Save the split dataframes to csv files
        train_metadata_df.to_csv(DATASET_DIR_BASE / "train.csv", index=False)
        val_metadata_df.to_csv(DATASET_DIR_BASE / "val.csv", index=False)
        test_metadata_df.to_csv(DATASET_DIR_BASE / "test.csv", index=False)
