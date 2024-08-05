import os

from pathlib import Path

import subprocess

import torchaudio

from .audio_dataset import AudioDataset

import yt_dlp

import pandas as pd

from glob import glob


BASE_URL = "https://raw.githubusercontent.com/cdjkim/audiocaps/master/dataset/"
DATASET_DIR_BASE = Path(__file__).parent / "audiocaps"


class AudioCaps(AudioDataset):
    def get_samples(self):
        metadata_path = os.path.join(DATASET_DIR_BASE, f"{self.kind}.csv")
        audiodata_dir = os.path.join(DATASET_DIR_BASE, f"{self.kind}_audio")

        # Download metadata and audios if necessary
        if self.download:
            self.__download_dataset(DATASET_DIR_BASE / "train.csv", DATASET_DIR_BASE / "train_audio")
            self.__download_dataset(DATASET_DIR_BASE / "val.csv", DATASET_DIR_BASE / "val_audio")
            self.__download_dataset(DATASET_DIR_BASE / "test.csv", DATASET_DIR_BASE / "test_audio")

        metadata_df = pd.read_csv(metadata_path)

        audio_paths = sorted(glob(os.path.join(audiodata_dir, "*.wav")), key=lambda x: int(os.path.basename(x)[:-4]))
        ids = [int(os.path.basename(path)[:-4]) for path in audio_paths]
        captions = [metadata_df[metadata_df["audiocap_id"] == audiocap_id]["caption"].item() for audiocap_id in ids]

        return audio_paths, captions

    def __download_dataset(self, metadata_path: str | Path, audiodata_dir: str | Path):
        # Download metadata and create directory if necessary
        os.makedirs(DATASET_DIR_BASE, exist_ok=True)
        self.__download_metadata(metadata_path)
        metadata_df = pd.read_csv(metadata_path)
        # Copy metadata for removal of invalid samples
        metadata_df_new = metadata_df.copy()
        corrupted_sample = None

        # Download audios and create directories if necessary
        os.makedirs(audiodata_dir, exist_ok=True)
        download_dir = os.path.join(DATASET_DIR_BASE, "full_audios")
        os.makedirs(download_dir, exist_ok=True)
        for youtube_id, audiocap_id, start_time in zip(metadata_df["youtube_id"], metadata_df["audiocap_id"],
                                                       metadata_df["start_time"]):
            if not os.path.exists(os.path.join(audiodata_dir, f'{audiocap_id}.wav')):
                successful_download = True
                if not os.path.exists(os.path.join(download_dir, f'{youtube_id}.wav')):
                    successful_download = self.__download_audio(youtube_id, download_dir)
                if successful_download:
                    # Extract audio segment using the given start time in the metadata
                    self.__extract_audio_segment(
                        os.path.join(download_dir, f'{youtube_id}.wav'),
                        start_time,
                        os.path.join(audiodata_dir, f'{audiocap_id}.wav')
                    )
                    try:
                        # Try to load wav file
                        torchaudio.load(os.path.join(audiodata_dir, f'{audiocap_id}.wav'))
                    except RuntimeError:
                        corrupted_sample = os.path.join(audiodata_dir, f'{audiocap_id}.wav')

                else:
                    # Remove unavailable samples from csv file
                    metadata_df_new = metadata_df_new[metadata_df_new["youtube_id"] != youtube_id]
                    metadata_df_new.to_csv(metadata_path, index=False)

            # Remove corrupted samples
            if corrupted_sample is not None:
                print(corrupted_sample)
                metadata_df_new = metadata_df_new[metadata_df_new["audiocap_id"] != audiocap_id]
                metadata_df_new.to_csv(metadata_path, index=False)
                print(f"Removing corrupted sample: {corrupted_sample}")
                os.remove(corrupted_sample)
                corrupted_sample = None

        print(f"Downloaded {self.kind} audio data to {audiodata_dir}")

    def __download_metadata(self, metadata_path: str):
        if not os.path.exists(metadata_path):
            filename = self.kind + ".csv"
            self.download_file(BASE_URL + filename, metadata_path, f"Downloading audio metadata")
            print(f"Downloaded {self.kind} metadata to {metadata_path}")

    @staticmethod
    def __download_audio(youtube_id: str, output_dir: str) -> bool:
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(output_dir, f'{youtube_id}.%(ext)s'),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'quiet': True
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([f'https://www.youtube.com/watch?v={youtube_id}'])
            return True
        except yt_dlp.utils.DownloadError:
            return False

    @staticmethod
    def __extract_audio_segment(file_path: str, start_time: int, output_path: str):
        command = [
            'ffmpeg',
            '-i', file_path,
            '-ss', str(start_time),
            '-t', "10",
            "-ac", "1",
            output_path
        ]

        subprocess.run(command, check=True)
