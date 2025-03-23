"""
In VPT repo, they handle bias term very carefully when weight decay > 0 for different optimizers.
They also used a special implementation of Adamw from huggingface with weight decay fix. Check their repo's optimizer.py
for additional information.
We skip them for now as AdaptFormer and ConvPass skip this as either

"""

import torch.optim as optim
from typing import Any, Callable, Iterable, List, Tuple, Optional

from utils.setup_logging import get_logger

logger = get_logger("Prompt_CAM")


def make_optimizer(tune_parameters, params):
    if params.optimizer == 'adam':
        optimizer = optim.Adam(
            tune_parameters,
            lr=params.lr,
            weight_decay=params.wd,
        )

    elif params.optimizer == 'adamw':
        optimizer = optim.AdamW(
            tune_parameters,
            lr=params.lr,
            weight_decay=params.wd,
        )
    else:
        optimizer = optim.SGD(
            tune_parameters,
            lr=params.lr,
            weight_decay=params.wd,
            momentum=params.momentum,
        )
    return optimizer





