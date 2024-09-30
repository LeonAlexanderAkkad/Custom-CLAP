from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR, MultiplicativeLR


def create_scheduler(optimizer: Optimizer, warmup_steps: int, T_max: int, min_lr: float):
    """Creates a learning rate scheduler with warm-up and cosine annealing phases.

    This function returns a scheduler that first warms up the learning rate
    over a specified number of steps and then follows a cosine annealing schedule finishing with a constant learning rate.

    Parameters
    ----------
    optimizer : Optimizer
        The optimizer for which to schedule the learning rate.
    warmup_steps : int
        The number of steps over which to warm up the learning rate.
    T_max : int
        The maximum number of steps for cosine annealing.
    min_lr : float
        The minimum learning rate for cosine annealing constant lr scheduler.

    Returns
    -------
    SequentialLR
        A learning rate scheduler that first warms up the learning rate and then applies cosine annealing.

    Example
    -------
    >>> import torch
    >>> from clap import Clap
    >>> from clap.utils import load_clap_config
    >>> config = load_clap_config('path/to/config.yaml')
    >>> model = Clap(config)
    >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    >>> warmup_steps = 5
    >>> T_max = 50
    >>> min_lr = 1e-6
    >>> scheduler = create_scheduler(optimizer, warmup_steps, T_max, min_lr)
    >>> for epoch in range(100):
    ...     optimizer.step()
    ...     scheduler.step()
    """
    def __warmup_lambda(step: int):
        if warmup_steps == 0:
            return 1.0

        return step / warmup_steps

    warm_up = LambdaLR(optimizer=optimizer, lr_lambda=__warmup_lambda)
    decay = CosineAnnealingLR(optimizer=optimizer, T_max=T_max, eta_min=min_lr)
    plateau = MultiplicativeLR(optimizer=optimizer, lr_lambda=lambda step: 1)

    return SequentialLR(optimizer=optimizer, schedulers=[warm_up, decay, plateau], milestones=[warmup_steps, warmup_steps + T_max + 1])
