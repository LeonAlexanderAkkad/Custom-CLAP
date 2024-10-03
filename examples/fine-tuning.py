import argparse

import wandb

import torch
from torch import optim
from torch.utils.data import DataLoader
from clap import Clap, ClapAudioClassifier
from clap.training import create_scheduler, ClapFinetuner
from clap.datasets import ClapDataset
from clap.utils import get_target_device, load_clap_config, set_random_seed


# Define command line arg parser
parser = argparse.ArgumentParser()

# Add args
parser.add_argument(
    "config_path",
    type=str,
    help="The path to the config file to use."
)
parser.add_argument(
    "clap_ckpt_path",
    type=str,
    help="The path to the file where the clap model checkpoint is saved."
)
parser.add_argument(
    "clf_ckpt_path",
    type=str,
    help="The path to the file where the classifier checkpoints will be saved to."
)
parser.add_argument(
    "datasets",
    type=str,
    nargs="+",
    choices=["ESC50"],
    help="The datasets to be used for training and evaluation. Options: [ESC50]"
)
parser.add_argument(
    "dataset_paths",
    type=str,
    nargs="+",
    help="The paths to the specified datasets. The order matters and should be similar to the datasets argument."
)
parser.add_argument(
    "--download",
    action="store_true",
    help="Whether to download the datasets or not.",
    default=False
)
parser.add_argument(
    "-wandb",
    "--use_wandb",
    action="store_true",
    help="Use logging to weights and biases.",
    default=False
)
parser.add_argument(
    "--project",
    type=str,
    help="The project where the results will be logged to on wandb."
)
parser.add_argument(
    "--name",
    type=str,
    help="The name of the run that will be logged to wandb."
)
parser.add_argument(
    "--start_from_checkpoint",
    action="store_true",
    help="Specifies if training should start from a pretrained model stored in ckpt_path.",
    default=False
)
parser.add_argument(
    "--early_stopping",
    action="store_true",
    help="Specifies if early stopping should be used during training.",
    default=False
)

args = parser.parse_args()

# Load config for audio processing and get target device
config = load_clap_config(config_path=args.config_path)

# Initialize wandb
if args.use_wandb:
    wandb.init(
        # Set the wandb project where this run will be logged
        project=args.project,
        name=args.name,
        # Track hyperparameters
        config=config
    )
    config = wandb.config

config_fine_tune = config["fine-tuning"]
device = get_target_device()

# Load Datasets
seed = set_random_seed(config_fine_tune["seed"])
train_dataset = ClapDataset(config=config, kinds=["train"], datasets=args.datasets, datasets_paths=args.dataset_paths, download=args.download)
val_dataset = ClapDataset(config=config, kinds=["val"], datasets=args.datasets, datasets_paths=args.dataset_paths, download=args.download)
test_dataset = ClapDataset(config=config, kinds=["test"], datasets=args.datasets, datasets_paths=args.dataset_paths, download=args.download)

# Define data loaders
train_loader = DataLoader(train_dataset, batch_size=config_fine_tune["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config_fine_tune["batch_size"], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=config_fine_tune["batch_size"], shuffle=False)

# Define model, optimizer, scheduler, loss function and fine-tuner
if args.start_from_checkpoint:
    clap_clf = ClapAudioClassifier.from_ckpt(args.config_path, args.clf_ckpt_path)
    print(f"Number of parameters to train: {sum(p.numel() for p in clap_clf.parameters())}")
    trainer = ClapFinetuner.from_ckpt(
        model=clap_clf,
        ckpt_path=args.clf_ckpt_path,
        config_fine_tune=config_fine_tune,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=config_fine_tune["epochs"],
        enable_wandb_logging=args.use_wandb
    )
else:
    clap = Clap.from_ckpt(config_path=args.config_path, ckpt_path=args.clap_ckpt_path)
    clap_clf = ClapAudioClassifier(clap=clap, config=config).to(device)
    print(f"Number of parameters to train: {sum(p.numel() for p in clap.parameters())}")
    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        clap.parameters(),
        lr=config_fine_tune["learning_rate"],
        betas=config_fine_tune["betas"],
        weight_decay=config_fine_tune["weight_decay"]
    )
    scheduler = create_scheduler(
        optimizer,
        warmup_steps=config_fine_tune["warmup_epochs"],
        T_max=len(train_loader)*config_fine_tune["annealing_epochs"]-1,
        min_lr=1e-6
    )

    trainer = ClapFinetuner(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        model=clap_clf,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        epochs=config_fine_tune["epochs"],
        enable_wandb_logging=args.use_wandb
    )

trainer.finetune_and_eval(args.clf_ckpt_path, early_stopping=args.early_stopping)

if args.use_wandb:
    wandb.finish()
