import os

from glob import glob

import yaml

from pathlib import Path


BASE_PATH = Path(__file__).parent


def create_clap_config(audio: dict, text: dict, projection: dict, training: dict):
    """Creates a YAML configuration file from the provided dictionaries.

    This function takes four dictionaries containing configuration settings for
    audio, text, projection, and training. It creates a YAML file with these
    settings, named based on the encoder name fields in the audio and text dictionaries.

    Parameters
    ----------
    audio : dict
        A dictionary containing audio configuration settings. Must include a "name" key.
    text : dict
        A dictionary containing text configuration settings. Must include a "name" key.
    projection : dict
        A dictionary containing projection configuration settings.
    training : dict
        A dictionary containing training configuration settings.

    Examples
    --------
    >>> audio_dict = {
    ...     "name": 'htsat-tiny',
    ...     "pretrained_path": 'clap/model/layers/audio_encoders/htsat/HTSAT_AudioSet_Saved_1.ckpt',
    ...     "sampling_rate": 32000,
    ...     "duration": 10,
    ...     "use_fusion": True,
    ...     "window_size": 1024,
    ...     "hop_size": 320,
    ...     "mel_bins": 64,
    ...     "f_min": 50,
    ...     "f_max": 14000,
    ...     "num_classes": 527,
    ...     "depths": [2, 2, 6, 2],
    ...     "embed_dim": 96,
    ...     "out_size": 768
    ... }
    >>> text_dict = {
    ...     "name": 'gpt2',
    ...     "max_len": 77,
    ...     "out_size": 768
    ... }
    >>> projection_dict = {
    ...     "hidden_size": 1024,
    ...     "out_size": 1024
    ... }
    >>> training_dict = {
    ...     "learning_rate": 0.0001,
    ...     "batch_size": 64,
    ...     "epochs": 5
    ... }
    >>> create_clap_config(audio_dict,text_dict,projection_dict,training_dict)
    """
    audio_encoder = audio["name"]
    text_encoder = text["name"]

    config = {
        "audio": audio,
        "text": text,
        "projection": projection,
        "training": training,
    }

    with open(BASE_PATH / f"clap_{audio_encoder}_{text_encoder}.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def load_clap_config(audio_encoder: str, text_encoder: str, version: str | int) -> dict:
    """Load configuration settings from a YAML configuration file based on encoder names and the specified version."""
    config = None
    config_paths = glob(os.path.join(BASE_PATH, "*.yml"))
    for config_path in config_paths:
        if audio_encoder in config_path and text_encoder in config_path and "v" + str(version) in config_path:
            config = config_path

    if config is None:
        raise FileNotFoundError(f"Config file not found for {audio_encoder} and {text_encoder}."
                                f"Available configs: {config_paths}")

    with open(config, "r") as f:
        config = yaml.safe_load(f)

    return config
