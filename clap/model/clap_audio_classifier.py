import torch
from torch import nn

from .clap import Clap
from ..utils import load_clap_config, load_clf_ckpt


ACTIVATION_FUNCTIONS = {None: None, "relu": nn.ReLU, "sigmoid": nn.Sigmoid, "tanh": nn.Tanh}


class ClapAudioClassifier(nn.Module):
    """A classifier model that uses CLAP for audio classification.

    Attributes
    ----------
    clap : Clap
        The CLAP model instance.
    clf_config : dict
        Configuration for the classifier.
    in_dim : int
        Input dimension for the classifier (this must have the same dimension as the output of the audio encoder).
    out_dim : int
        Output dimension for the classifier.
    act_fn : nn.Module or None
        Activation function to apply after the final linear layer.
    fc : nn.Linear
        The final linear layer of the classifier.
    """
    def __init__(self, clap: Clap, config: dict):
        """Initializes the audio classifier model.

        Parameters
        ----------
        clap : Clap
            An instance of CLAP containing the audio and text encoders.
        config : dict
            A configuration dictionary with the classifier parameters.
        """
        super().__init__()

        self.clap = clap
        self.clf_config = config["classifier"]
        self.in_dim = self.clf_config["in_dim"]
        self.out_dim = self.clf_config["out_dim"]
        self.act_fn = ACTIVATION_FUNCTIONS[self.clf_config["act_fn"]]
        self.fc = nn.Linear(self.in_dim, self.out_dim)

        if self.clf_config["freeze_text"]:
            self.clap.freeze_text_encoder()

        if self.clf_config["freeze_audio"]:
            self.clap.freeze_audio_encoder()

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        out_emb = self.clap.audio_encoder(x)
        out = self.fc(out_emb)

        if self.act_fn is not None:
            out = self.act_fn(out)

        return out

    @classmethod
    def from_ckpt(
            cls,
            audio_encoder: str,
            text_encoder: str,
            clap_cfg_version: str | int,
            clf_ckpt_version: str | int
    ) -> 'ClapAudioClassifier':
        """Create an instance of ClapAudioClassifier from a checkpoint.

        Parameters
        ----------
        audio_encoder : str
            Name of the audio encoder used in the checkpoint.
        text_encoder : str
            Name of the text encoder used in the checkpoint.
        clap_cfg_version : str or int
            Version identifier for the CLAP configuration file.
        clf_ckpt_version : str or int
            Version identifier for the classifier checkpoint file.

        Returns
        -------
        ClapAudioClassifier
            An instance of the ClapAudioClassifier class loaded with the specified checkpoint.
        """
        config = load_clap_config(audio_encoder, text_encoder, clap_cfg_version)
        clap = Clap(config)

        clf = cls(clap, config)
        ckpt = load_clf_ckpt(audio_encoder, text_encoder, clf_ckpt_version)

        clf.load_state_dict(ckpt["model"])

        return clf
