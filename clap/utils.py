from pathlib import Path

import yaml

import torch


def load_config(config: dict | Path | str) -> dict:
    if isinstance(config, Path) or isinstance(config, str):
        with open(config, "r") as f:
            config = yaml.safe_load(f)

    return config


def get_target_device():
    """Get the target device where training takes place."""
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
