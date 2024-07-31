from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR


def create_scheduler(optimizer: Optimizer, warmup_epochs: int, T_max: int, milestones: list[int]):
    """Creates a learning rate scheduler with warmup and cosine annealing."""
    def __warmup_lambda(epoch: int):
        return epoch / warmup_epochs

    warm_up = LambdaLR(optimizer=optimizer, lr_lambda=__warmup_lambda)
    decay = CosineAnnealingLR(optimizer=optimizer, T_max=T_max)

    return SequentialLR(optimizer=optimizer, schedulers=[warm_up, decay], milestones=milestones)
