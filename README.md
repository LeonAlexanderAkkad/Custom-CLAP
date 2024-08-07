**Table of content:**

[Overview](#custom-contrastive-language-audio-pretraining) | [Setup](#setup) | [Example usage](#example-usage) | [Datasets](#datasets) | [Results](#results) | [References](#references)

# Custom Contrastive Language-Audio Pretraining
This is a custom (unofficial) implementation of the model from the paper [CLAP: Learning Audio Concepts from Natural Language Supervision](https://doi.org/10.1109/ICASSP49357.2023.10095889).
CLAP learns acoustic concepts from natural language supervision and excels in various tasks when fine-tuned as well as enables "Zero-Shot" inference.
In order to improve the base implementation, I have included **improvements** like using a [Hierarchical Token Semantic Audio Transformer](https://doi.org/10.1109/ICASSP43922.2022.9746312) (**HTS-AT**) as audio encoder or **RoBERTa / GPT-2** as text encoder and [Attentional Feature Fusion](https://doi.org/10.1109/WACV48630.2021.00360) (**iAFF**) for effectively processing longer inputs.

Please note: In order to use a pretrained audio encoder for training, please download the official checkpoints yourself (links are below) and change the absolute path in the corresponding configuration file.

**Pretrained Audio Encoders:**

[HTS-AT checkpoint](https://drive.google.com/file/d/1Nj2OrJiTrn4F7mZxIW1JTUJVpF8MdFB3/view?usp=drive_link) | [CNN14 checkpoint](https://zenodo.org/records/3987831/files/Cnn14_mAP%3D0.431.pth?download=1)

## Setup
This package can be easily installed with Conda. To do so, download this repository and do the following:
```shell
# Move to the root directory
cd path_to_root_directory

# Create a conda environment using either env_cuda.yml or env_cpu.yml
conda env create -f env_cpu.yml
```
Alternatively, you can first install the dependencies listed in the [Environment YAML](env_cpu.yml) file manually or with any other package manager and then do the following:
```shell
# Move to the root directory
cd path_to_root_directory

# Install the clap package using pip (-e for editable mode)
pip install -e .
```

## Example Usage
[Training](#training-clap-on-audiocaps-and-clothov2) | [Fine-Tuning](#fine-tuning-clap-on-esc-50) | [Retrieval performance](#evaluating-the-retrieval-performance-of-clap-on-audiocaps-and-clothov2) | [Zero-Shot Audio Classification](#evaluating-the-zero-shot-performance-of-clap-on-the-esc-50-dataset) | [Fine-Tuned Audio Classification](#evaluating-the-fine-tuned-clap-audio-classifier-on-the-esc-50-dataset)

This package can be used to train and evaluate a CLAP model with a pre-defined `ClapTrainer` class and the `SymmetricCrossEntropyLoss` on the AudioCaps and ClothoV2 dataset.
Additionally, one can fine-tune a CLAP model using the implemented `ClaupAudioClassifier` on the ESC-50 dataset.
For logging, if needed, weights and biases can be used.
For further insights into the `ClapDataset` used for training have a look at [Datasets](#Datasets).

### Training CLAP on AudioCaps and ClothoV2
For training, I employ the described loss from the paper, Adam optimizer and a learning rate scheduler that uses a warm-up and a cosine annealing phase.
The exact configurations can be seen in the Training Notebook.
- Jupyter Notebook: [Training Notebook](examples/training.ipynb)
- Python script:

```python
from torch.utils.data import DataLoader
from torch import optim
from clap import Clap
from clap.datasets import ClapDataset
from clap.training import ClapTrainer, create_scheduler, SymmetricCrossEntropyLoss
from clap.utils import get_target_device, load_clap_config, set_random_seed

# Load config for audio processing and get target device
audio_encoder = "htsat-tiny"
text_encoder = "gpt2"
cfg_version = 1
ckpt_version = 1
config = load_clap_config(audio_encoder=audio_encoder, text_encoder=text_encoder, version=cfg_version)
device = get_target_device()

# Load Datasets
seed = set_random_seed(None)
train_dataset = ClapDataset(config=config, kinds=["train"])
val_dataset = ClapDataset(config=config, kinds=["val"])
test_dataset = ClapDataset(config=config, kinds=["test"])

# Define data loaders
train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"])
test_loader = DataLoader(test_dataset, batch_size=config["training"]["batch_size"])

# Define model, optimizer, scheduler and loss function
clap = Clap(config).to(device)
print(f"Number of parameters to train: {sum(p.numel() for p in clap.parameters())}")
optimizer = optim.Adam(clap.parameters(), lr=config["training"]["learning_rate"])
scheduler = create_scheduler(optimizer, warmup_steps=1000, T_max=len(train_loader)*config["training"]["epochs"], milestones=[1000])
loss_fn = SymmetricCrossEntropyLoss()
trainer = ClapTrainer(
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    model=clap,
    optimizer=optimizer,
    scheduler=scheduler,
    loss_fn=loss_fn,
    epochs=config["training"]["epochs"]
)

train_metrics, val_metrics, test_metrics = trainer.train_and_eval(audio_encoder=audio_encoder, text_encoder=text_encoder, version=1, early_stopping=False)
```

### Fine-Tuning CLAP on ESC-50
For fine-tuning, I froze the text encoder and trained a single fully connected linear layer and the audio encoder on the ESC-50 dataset.
The ESC-50 dataset was simply split in the following way: The first 4 folds were used as training, the final fold was split so that the validation set comprises 150 samples and the test set comprises 250 samples.
I made sure that no samples from the same audio source ended up in different splits.
The Adam optimizer with learning rate scheduling (30 steps warm-up stage, followed by cosine learning rate annealing) was employed.
The exact configurations can be seen in the Fine-Tuning Notebook.
- Jupyter Notebook: [Fine-Tuning Notebook](examples/finetuning.ipynb)
- Python script:

```python
import torch
from torch import optim
from torch.utils.data import DataLoader
from clap import Clap, ClapAudioClassifier
from clap.training import create_scheduler, ClapFinetuner
from clap.datasets import ClapDataset
from clap.utils import get_target_device, load_clap_config, set_random_seed

# Load config for audio processing and get target device
audio_encoder = "htsat-tiny"
text_encoder = "gpt2"
cfg_version = 1
ckpt_version = 2
config = load_clap_config(audio_encoder=audio_encoder, text_encoder=text_encoder, version=cfg_version)
device = get_target_device()

# Load Datasets
seed = set_random_seed(None)
train_dataset = ClapDataset(config=config, kinds=["train"], datasets=["ESC50"])
val_dataset = ClapDataset(config=config, kinds=["val"], datasets=["ESC50"])
test_dataset = ClapDataset(config=config, kinds=["test"], datasets=["ESC50"])

# Define data loaders
train_loader = DataLoader(train_dataset, batch_size=config["fine-tuning"]["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config["fine-tuning"]["batch_size"])
test_loader = DataLoader(test_dataset, batch_size=config["fine-tuning"]["batch_size"])

# Define model, optimizer, scheduler and loss function
clap = Clap.from_ckpt(audio_encoder=audio_encoder, text_encoder=text_encoder, ckpt_version=ckpt_version, cfg_version=cfg_version)
clap_clf = ClapAudioClassifier(clap=clap, config=config).to(device)
print(f"Number of parameters to train: {sum(p.numel() for p in clap_clf.parameters())}")
optimizer = optim.Adam(clap.parameters(), lr=config["fine-tuning"]["learning_rate"])
scheduler = create_scheduler(optimizer, warmup_steps=31, T_max=len(train_loader)*config["fine-tuning"]["epochs"], milestones=[31])
loss_fn = torch.nn.CrossEntropyLoss()
trainer = ClapFinetuner(
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    model=clap_clf,
    optimizer=optimizer,
    scheduler=scheduler,
    loss_fn=loss_fn,
    epochs=config["fine-tuning"]["epochs"]
)

train_metrics, val_metrics, test_metrics = trainer.finetune_and_eval(audio_encoder=audio_encoder, text_encoder=text_encoder, version=1, early_stopping=False)
```

### Evaluating the retrieval performance of CLAP on AudioCaps and ClothoV2
For retrieval evaluation, I only test the trained model on the test sets of the respective datasets.
- Jupyter Notebook: [Retrieval Notebook](examples/retrieval.ipynb)
- Python script:

```python
from torch.utils.data import DataLoader
from clap import Clap
from clap.evaluate import eval_retrieval
from clap.datasets import ClapDataset
from clap.utils import get_target_device, load_clap_config

# Load config for audio processing and get target device
audio_encoder = "htsat-tiny"
text_encoder = "gpt2"
cfg_version = 1
ckpt_version = 3
config = load_clap_config(audio_encoder=audio_encoder, text_encoder=text_encoder, version=cfg_version)
device = get_target_device()

# Initialize evaluation datasets and dataloaders
audio_caps_eval_dataset = ClapDataset(config=config, datasets=["AudioCaps"], kinds=["test"])
audio_caps_loader = DataLoader(audio_caps_eval_dataset, batch_size=64, shuffle=False)
clotho_eval_dataset = ClapDataset(config=config, datasets=["Clotho"], kinds=["test"])
clotho_loader = DataLoader(clotho_eval_dataset, batch_size=64, shuffle=False)

# Load trained model
clap = Clap.from_ckpt(audio_encoder=audio_encoder, text_encoder=text_encoder, cfg_version=cfg_version, ckpt_version=ckpt_version).to(device)

# Get metrics
audio_caps_metrics = eval_retrieval(model=clap, test_loader=audio_caps_loader)
clotho_metrics = eval_retrieval(model=clap, test_loader=clotho_loader)

print("Audio Caps:\n")
for name, score in audio_caps_metrics.items():
    print(f"{name:14}: {score:.4f}")

print("Clotho:\n")
for name, score in clotho_metrics.items():
    print(f"{name:14}: {score:.4f}")
```

### Evaluating the Zero-Shot performance of CLAP on the ESC-50 dataset
For Zero-Shot evaluation, I evaluate the trained model on the whole dataset and generate prompts from the dataset classes and compute the similarity between the embeddings of these prompts and the audios.
- Jupyter Notebook: [Zero-Shot Audio classification Notebook](examples/zero_shot_classification.ipynb)
- Python script:

```python
from torch.utils.data import DataLoader
from clap import Clap
from clap.evaluate import eval_zero_shot_classification
from clap.datasets import ClapDataset
from clap.utils import get_target_device, load_clap_config

# Load config for audio processing and get target device
audio_encoder = "htsat-tiny"
text_encoder = "gpt2"
cfg_version = 1
ckpt_version = 3
config = load_clap_config(audio_encoder=audio_encoder, text_encoder=text_encoder, version=cfg_version)
device = get_target_device()

# Load Dataset and DataLoader
esc50_dataset = ClapDataset(config=config, kinds=["train", "val", "test"], datasets=["ESC50"])
esc50_dataloader = DataLoader(esc50_dataset, batch_size=64, shuffle=False)
class_to_idx, _ = ClapDataset.load_esc50_class_mapping()

# Load pretrained model
model = Clap.from_ckpt(audio_encoder=audio_encoder, text_encoder=text_encoder, ckpt_version=ckpt_version, cfg_version=cfg_version).to(device)

# Generate prompts
prompt = "This is the sound of "
prompts = [prompt + y for y in class_to_idx.keys()]

# Compute text embedding for the prompts
class_embeddings = model.get_text_embeddings(prompts)

# Get ZS accuracy
acc = eval_zero_shot_classification(model=model, eval_loader=esc50_dataloader, class_embeddings=class_embeddings)

print(f'ESC50 Accuracy {acc}')
```

### Evaluating the Fine-Tuned CLAP audio classifier on the ESC-50 dataset
In order to compare it to the Zero-Shot evaluation, I evaluated the fine-tuned classifier on the whole ESC-50 datset.
- Jupyter Notebook: [Fine-Tuned Audio Classification Notebook](examples/fine-tuned_classification.ipynb)
- Python script:

```python
from torch.utils.data import DataLoader
from clap import  ClapAudioClassifier
from clap.evaluate import eval_fine_tuned_classification
from clap.datasets import ClapDataset
from clap.utils import get_target_device, load_clap_config

# Load config for audio processing and get target device
audio_encoder = "htsat-tiny"
text_encoder = "gpt2"
cfg_version = 1
ckpt_version = 1
config = load_clap_config(audio_encoder=audio_encoder, text_encoder=text_encoder, version=cfg_version)
device = get_target_device()

# Load Dataset and DataLoader
esc50_dataset = ClapDataset(config=config, kinds=["train", "val", "test"], datasets=["ESC50"])
esc50_dataloader = DataLoader(esc50_dataset, batch_size=64, shuffle=False)

# Load pretrained model
clf = ClapAudioClassifier.from_ckpt(audio_encoder=audio_encoder, text_encoder=text_encoder, clap_cfg_version=cfg_version, clf_ckpt_version=ckpt_version).to(device)

acc = eval_fine_tuned_classification(model=clf, eval_loader=esc50_dataloader)

print(f'ESC50 Accuracy: {acc}')
```

## Datasets
I implemented a `ClapDataset` that should be used for training and evaluation.
For now, it only supports the [ClothoV2](https://doi.org/10.1109/ICASSP40776.2020.9052990), [AudioCaps](https://doi.org/10.18653/v1/N19-1011) and [ESC-50](https://doi.org/10.1145/2733373.2806390) datasets.
Each ClapDataset can load either train, validation, test or any combination of these three dataset splits, and it can include either AudioCaps, ClothoV2 or ESC-50 or any combination of these three.
If need be, the datasets including the metadata will be downloaded to a specific directory.
The user should not rename/move the configuration and checkpoint files, as there location is important for the `ClapDataset` and `Clap` respectively.

Examples:

```python
from clap.utils import load_clap_config
from clap.datasets import ClapDataset

audio_encoder = "htsat-tiny"
text_encoder = "gpt2"
version = 1
config = load_clap_config(audio_encoder=audio_encoder, text_encoder=text_encoder, version=version)
clotho_train_dataset = ClapDataset(datasets=["Clotho"], kinds=["train"], download=True, config=config)
audiocaps_test_dataset = ClapDataset(datasets=["AudioCaps"], kinds=["test"], download=True, config=config)
combined_val_dataset = ClapDataset(datasets=["AudioCaps", "Clotho"], kinds=["val"], download=True, config=config)
```

## Results
These performances were achieved by using this [configuration](clap/configs/clap_htsat-tiny_gpt2_v1.yml) and this [notebook](examples/training.ipynb).
ClAP was briefly trained according to the aforementioned configuration on both AudioCaps and ClothoV2, employing feature fusion for longer audios and simple repeat padding for shorter audios.
The fine-tuning hyperparameters can be found in the same configuration file.
A singly fully connected linear layer was trained with the audio encoder for fine-tuning.

### Retrieval performance

#### AudioCaps:

|                | Recall@1 | Recall@5 | Recall@10 | mAP@10 |
|----------------|:--------:|:--------:|:---------:|:------:|
| **Audio-Text** |  0.7211  |  0.9649  |  0.9880   | 0.8270 |
| **Text-Audio** |  0.7360  |  0.9690  |  0.9909   | 0.8388 |

#### ClothoV2:

|                | Recall@1 | Recall@5 | Recall@10 | mAP@10 |
|----------------|:--------:|:--------:|:---------:|:------:|
| **Audio-Text** |  0.1562  |  0.6303  |  0.8283   | 0.3542 |
| **Text-Audio** |  0.1402  |  0.6794  |  0.8545   | 0.3467 |


### Zero-Shot Audio classification

|        | Top-1 Accuracy |
|--------|:--------------:|
| ESC-50 |     0.7465     |


### Fine-Tuned Audio Classification

|        | Top-1 Accuracy |
|--------|:--------------:|
| ESC-50 |     0.9341     |

## References
> B. Elizalde, S. Deshmukh, M. A. Ismail and H. Wang, "CLAP Learning Audio Concepts from Natural Language Supervision," ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Rhodes Island, Greece, 2023, pp. 1-5
>
> Doi: [10.1109/ICASSP49357.2023.10095889](https://doi.org/10.1109/ICASSP49357.2023.10095889).

> Y. Wu, K. Chen, T. Zhang, Y. Hui, T. Berg-Kirkpatrick and S. Dubnov, "Large-Scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation," ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Rhodes Island, Greece, 2023, pp. 1-5
>
> Doi: [10.1109/ICASSP49357.2023.10095969](https://doi.org/10.1109/ICASSP49357.2023.10095969).

> K. Chen, X. Du, B. Zhu, Z. Ma, T. Berg-Kirkpatrick and S. Dubnov, "HTS-AT: A Hierarchical Token-Semantic Audio Transformer for Sound Classification and Detection," ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Singapore, Singapore, 2022, pp. 646-650
>
> Doi: [10.1109/ICASSP43922.2022.9746312](https://doi.org/10.1109/ICASSP43922.2022.9746312).

> Y. Dai, F. Gieseke, S. Oehmcke, Y. Wu and K. Barnard, "Attentional Feature Fusion," 2021 IEEE Winter Conference on Applications of Computer Vision (WACV), Waikoloa, HI, USA, 2021, pp. 3559-3568
> 
> Doi: [10.1109/WACV48630.2021.00360](https://doi.org/10.1109/WACV48630.2021.00360).

> Chris Dongjoo Kim, Byeongchang Kim, Hyunmin Lee, and Gunhee Kim. 2019. AudioCaps: Generating Captions for Audios in The Wild. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 119–132, Minneapolis, Minnesota. Association for Computational Linguistics.
> 
> Doi: [10.18653/v1/N19-1011](https://doi.org/10.18653/v1/N19-1011)

> K. Drossos, S. Lipping and T. Virtanen, "Clotho: an Audio Captioning Dataset," ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Barcelona, Spain, 2020, pp. 736-740.
> 
> Doi: [10.1109/ICASSP40776.2020.9052990](https://doi.org/10.1109/ICASSP40776.2020.9052990)

> Karol J. Piczak. 2015. ESC: Dataset for Environmental Sound Classification. In Proceedings of the 23rd ACM international conference on Multimedia (MM '15). Association for Computing Machinery, New York, NY, USA, 1015–1018. 
> 
> Doi: [10.1145/2733373.2806390](https://doi.org/10.1145/2733373.2806390)
