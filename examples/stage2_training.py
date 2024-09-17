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
    "--early_stopping",
    action="store_true",
    default=False,
    help="Specifies if early stopping should be used during training."
)

args = parser.parse_args()

# Load config for audio processing and get target device
config = load_clap_config(config_path=args.config_path)
device = get_target_device()

# Load Datasets
seed = set_random_seed(config["training"]["seed"])
train_dataset = ClapDataset(config=config, kinds=["train"], datasets=args.datasets, datasets_paths=args.dataset_paths, download=args.download)
val_dataset = ClapDataset(config=config, kinds=["val"], datasets=args.datasets, datasets_paths=args.dataset_paths, download=args.download)
test_dataset = ClapDataset(config=config, kinds=["test"], datasets=args.datasets, datasets_paths=args.dataset_paths, download=args.download)

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

# Define data loaders
train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"])
test_loader = DataLoader(test_dataset, batch_size=config["training"]["batch_size"])

# Define model, optimizer, scheduler and loss function
if args.parameters is not None:
    clap = Clap.from_ckpt(args.config_path, args.parameters).to(device)
else:
    clap = Clap(config).to(device)
print(f"Number of parameters to train: {sum(p.numel() for p in clap.parameters())}")
optimizer = optim.Adam(clap.parameters(), lr=config["training"]["learning_rate"])
scheduler = create_scheduler(
    optimizer,
    warmup_steps=config["training"]["warmup_steps"],
    T_max=len(train_loader)*config["training"]["stage2_epochs"],
    milestones=config["training"]["milestones"]
)
loss_fn = SymmetricCrossEntropyLoss()

# Get models to distill soft labels
distill_models = []
for config, ckpt in zip(args.distill_config_paths, args.distill_ckpt_paths):
    distill_model = Clap.from_ckpt(config, ckpt).to(device)
    distill_model.freeze_encoders()
    distill_model.eval()
    distill_models.append(distill_model)

distill_from = distill_models if len(distill_models) else None

trainer = ClapTrainer(
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    model=clap,
    optimizer=optimizer,
    scheduler=scheduler,
    loss_fn=loss_fn,
    epochs=config["training"]["stage2_epochs"],
    enable_wandb_logging=args.use_wandb,
    distill_from=distill_from,
    distill_weight=args.distill_weight
)

trainer.train_and_eval(args.ckpt_path, early_stopping=args.early_stopping)

if args.use_wandb:
    wandb.finish()
