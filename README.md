**Table of content:**
- [Overview](#custom-contrastive-language-audio-pretraining)
- [Setup](#setup)
- [Usage](#usage)
- [Datasets](#datasets)
- [References](#references)

# Custom Contrastive Language-Audio Pretraining
This is a custom (unofficial) implementation of the model from the paper [CLAP: Learning Audio Concepts from Natural Language Supervision](https://doi.org/10.1109/ICASSP49357.2023.10095889).
CLAP learns acoustic concepts from natural language supervision and excels in various tasks when fine-tuned as well as enables "Zero-Shot" inference.
In order to improve the base implementation, I have included **improvements** like using a [Hierarchical Token Semantic Audio Transformer](https://doi.org/10.1109/ICASSP43922.2022.9746312) (**HTS-AT**) as audio encoder or **RoBERTa / GPT-2** as text encoder and [Attentional Feature Fusion](https://doi.org/10.1109/WACV48630.2021.00360) (**iAFF**) for effectively processing longer inputs.
(Not implemented yet: Additionally, I will probably include caption enhancements for the implemented datasets using an appropriate model.)

## Setup
This package can be easily installed with Conda. For this download this repository and do the following:
```shell
# Move to the root directory
cd path_to_root_directory

# Create a conda environment using either env_cuda.yml or env_cpu.yml
conda env create -f env_cpu.yml
```
Alternatively, you can first install the dependencies listed in the .yml file manually or with any other package manager and then do the following:
```shell
# Move to the root directory
cd path_to_root_directory

# Install the clap package using pip (-e for editable mode)
pip install -e .
```

## Usage
This package can be used to train and fine-tune a clap model with a pre-defined ClapTrainer class and the SymmetricCrossEntropyLoss.
For further insights into the ClapDataset used for training have a look at [Datasets](#Datasets).
Example usage:
- Using the [Experiments Notebook](experiments.ipynb).
- Using a python script:

```python
from clap import Clap, ClapDataset, SymmetricCrossEntropyLoss, ClapTrainer
from torch.utils.data import DataLoader
from torch.optim import Adam

# Load the model using a path to a config file (primarily for training)
# The path to the pretrained audio encoder has to be specified in the config
config_path = "clap/configs/clap_cnn14_distilroberta-base_v1.yml"
clap_from_config = Clap(config=config_path)

# Alternatively, load a trained model from a checkpoint file
# Be aware the checkpoint and the config file must have the same name
clap_from_ckpt = Clap.from_ckpt(ckpt="clap/checkpoints/clap_cnn14_roberta.ckpt")

# Initialize ClapDataset and DataLoaders
train_dataset = ClapDataset(datasets=["AudioCaps", "Clotho"], kind="train", download=True, config=config_path,
                            preprocess_audio=True)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = ClapDataset(datasets=["AudioCaps", "Clotho"], kind="val", download=True, config=config_path,
                          preprocess_audio=True)
val_loader = DataLoader(train_dataset, batch_size=32)

test_dataset = ClapDataset(datasets=["AudioCaps", "Clotho"], kind="test", download=True, config=config_path,
                           preprocess_audio=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Specify loss and optimizer
criterion = SymmetricCrossEntropyLoss()
optimizer = Adam(clap_from_config.parameters())

# Load trainer and start training
trainer = ClapTrainer(train_loader, val_loader, test_loader, clap_from_config, optimizer, criterion, epochs=10)
metrics = trainer.train_and_eval(out_path="checkpoints/clap_cnn14_roberta.ckpt")
```

## Datasets
I implemented a `ClapDataset` that should be used for training.
For now, it only supports the [Clotho](https://doi.org/10.1109/ICASSP40776.2020.9052990) and [AudioCaps](https://doi.org/10.18653/v1/N19-1011) datasets.
Each ClapDataset represents either train, validation or test dataset and can include either AudioCaps or Clotho or both.
If need be, the datasets including the metadata will be downloaded to a specific directory.
The audio needs to be preprocessed first, thus setting `preprocess_audio=True` and specifing a config file is necessary when first initialising a `ClapDataset`.
The user does should not rename/move these files, as there location is important for the `ClapDataset`.

Examples:

```python
from clap import ClapDataset

config_path = "clap/configs/clap_cnn14_distilroberta-base_v1.yml"
clotho_train_dataset = ClapDataset(datasets=["Clotho"], kind="train", download=True, config=config_path,
                                   preprocess_audio=True)
audiocaps_test_dataset = ClapDataset(datasets=["AudioCaps"], kind="test", download=True, config=config_path,
                                     preprocess_audio=True)
combined_val_dataset = ClapDataset(datasets=["AudioCaps", "Clotho"], kind="val", download=True, config=config_path,
                                   preprocess_audio=True)
```

## References
- B. Elizalde, S. Deshmukh, M. A. Ismail and H. Wang, "CLAP Learning Audio Concepts from Natural Language Supervision," ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Rhodes Island, Greece, 2023, pp. 1-5, doi: [10.1109/ICASSP49357.2023.10095889](https://doi.org/10.1109/ICASSP49357.2023.10095889).
- Y. Wu, K. Chen, T. Zhang, Y. Hui, T. Berg-Kirkpatrick and S. Dubnov, "Large-Scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation," ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Rhodes Island, Greece, 2023, pp. 1-5, doi: [10.1109/ICASSP49357.2023.10095969](https://doi.org/10.1109/ICASSP49357.2023.10095969).
- K. Chen, X. Du, B. Zhu, Z. Ma, T. Berg-Kirkpatrick and S. Dubnov, "HTS-AT: A Hierarchical Token-Semantic Audio Transformer for Sound Classification and Detection," ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Singapore, Singapore, 2022, pp. 646-650, doi: [10.1109/ICASSP43922.2022.9746312](https://doi.org/10.1109/ICASSP43922.2022.9746312).
- Y. Dai, F. Gieseke, S. Oehmcke, Y. Wu and K. Barnard, "Attentional Feature Fusion," 2021 IEEE Winter Conference on Applications of Computer Vision (WACV), Waikoloa, HI, USA, 2021, pp. 3559-3568, doi: [10.1109/WACV48630.2021.00360](https://doi.org/10.1109/WACV48630.2021.00360).
