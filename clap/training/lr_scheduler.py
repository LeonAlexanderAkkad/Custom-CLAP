from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR


def create_scheduler(optimizer: Optimizer, warmup_steps: int, T_max: int, milestones: list[int]):
    """Creates a learning rate scheduler with warm-up and cosine annealing phases.

    This function returns a scheduler that first warms up the learning rate
    over a specified number of steps and then follows a cosine annealing schedule.

    Parameters
    ----------
    optimizer : Optimizer
        The optimizer for which to schedule the learning rate.
    warmup_steps : int
        The number of steps over which to warm up the learning rate.
    T_max : int
        The maximum number of steps for cosine annealing.
    milestones : list of int
        List of step indices at which the scheduler switches from warm-up to cosine annealing.

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
    >>> milestones = [warmup_steps]
    >>> scheduler = create_scheduler(optimizer, warmup_steps, T_max, milestones)
    >>> for epoch in range(100):
    ...     optimizer.step()
    ...     scheduler.step()
    """
    def __warmup_lambda(step: int):
        if warmup_steps == 0:
            return 1.0

        return step / warmup_steps

    warm_up = LambdaLR(optimizer=optimizer, lr_lambda=__warmup_lambda)
    decay = CosineAnnealingLR(optimizer=optimizer, T_max=T_max, eta_min=1e-6)

    return SequentialLR(optimizer=optimizer, schedulers=[warm_up, decay], milestones=milestones)
