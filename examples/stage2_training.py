import argparse

import wandb

from torch.utils.data import DataLoader
from torch import optim

from clap import Clap
from clap.datasets import ClapDataset
from clap.training import ClapTrainer, create_scheduler, SymmetricCrossEntropyLoss
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
    "ckpt_path",
    type=str,
    help="The path to the file where model checkpoints are saved to."
)
parser.add_argument(
    "datasets",
    type=str,
    nargs="+",
    choices=["AudioCaps", "Clotho"],
    help="The datasets to be used for training and evaluation. Options: [AudioCaps, Clotho]"
)
parser.add_argument(
    "--dataset_paths",
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
    "--distill_weight",
    type=float,
    default=0,
    help="The distillation weight used for computing the weighted some between the stage 1 and stage 2 loss."
)
parser.add_argument(
    "--distill_config_paths",
    nargs="+",
    type=str,
    default=[],
    help="The path to the config files of the models used for computing the soft targets."
)
parser.add_argument(
    "--distill_ckpt_paths",
    nargs="+",
    type=str,
    default=[],
    help="The path to the checkpoint files of the models used for computing the soft targets."
)
parser.add_argument(
    "--parameters",
    type=str,
    help="The path to the checkpoint file for initializing the parameters of the model to be trained."
)
parser.add_argument(
    "-wandb",
    "--use_wandb",
    action="store_true",
    default=False,
    help="Use logging to weights and biases."
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
    default=False,
    help="Specifies if early stopping should be used during training."
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

# Get target device and training config
config_train = config["training"]["stage2"]
device = get_target_device()

# Load Datasets
seed = set_random_seed(config_train["seed"])
train_dataset = ClapDataset(config=config, kinds=["train"], datasets=args.datasets, datasets_paths=args.dataset_paths, download=args.download)
val_dataset = ClapDataset(config=config, kinds=["val"], datasets=args.datasets, datasets_paths=args.dataset_paths, download=args.download)
test_dataset = ClapDataset(config=config, kinds=["test"], datasets=args.datasets, datasets_paths=args.dataset_paths, download=args.download)

# Define data loaders
train_loader = DataLoader(train_dataset, batch_size=config_train["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config_train["batch_size"])
test_loader = DataLoader(test_dataset, batch_size=config_train["batch_size"])

# Get models to distill soft labels
distill_models = []
for config_path, ckpt_path in zip(args.distill_config_paths, args.distill_ckpt_paths):
    distill_model = Clap.from_ckpt(config_path, ckpt_path).to(device)
    distill_model.freeze_model()
    distill_model.eval()
    distill_models.append(distill_model)

distill_from = distill_models if len(distill_models) else None

# Define model, optimizer, scheduler, loss function and trainer
if args.start_from_checkpoint:
    clap = Clap.from_ckpt(args.config_path, args.ckpt_path)
    print(f"Number of parameters to train: {sum(p.numel() for p in clap.parameters())}")
    trainer = ClapTrainer.from_ckpt(
        model=clap,
        ckpt_path=args.ckpt_path,
        config_train=config_train,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=config_train["epochs"],
        enable_wandb_logging=args.use_wandb,
        distill_from=distill_from,
        distill_weight=args.distill_weight
    )
else:
    if args.parameters is not None:
        clap = Clap.from_ckpt(args.config_path, args.parameters).to(device)
    else:
        clap = Clap(config).to(device)
    print(f"Number of parameters to train: {sum(p.numel() for p in clap.parameters())}")
    loss_fn = SymmetricCrossEntropyLoss()

    optimizer = optim.AdamW(
        clap.parameters(),
        lr=config_train["learning_rate"],
        betas=config_train["betas"],
        weight_decay=config_train["weight_decay"]
    )

    scheduler = create_scheduler(
        optimizer,
        warmup_steps=len(train_loader) * config_train["warmup_epochs"],
        T_max=len(train_loader) * config_train["annealing_epochs"] - 1,
        min_lr=1e-6
    )

    trainer = ClapTrainer(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        model=clap,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        epochs=config_train["epochs"],
        enable_wandb_logging=args.use_wandb
    )

trainer.train_and_eval(args.ckpt_path, early_stopping=args.early_stopping)

if args.use_wandb:
    wandb.finish()
