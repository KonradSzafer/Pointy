import os
import json
import logging
import random

import numpy as np
import torch
from torch import nn
from omegaconf import OmegaConf
from tabulate import tabulate

from src.paths import PATHS

logger = logging.getLogger(__name__)


def setup_logger(filename: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(filename)


def log_config(config: OmegaConf, logger: logging.Logger) -> None:
    logger.info("Config:\n%s", OmegaConf.to_yaml(config))


def log_table(data: dict, logger: logging.Logger) -> None:
    """ Log a dictionary as a table """
    # Convert single values to lists
    if not isinstance(next(iter(data.values())), list):
        data = {k: [v] for k, v in data.items()}
    table = tabulate(data, headers="keys", tablefmt="grid")
    logger.info("\n%s", table)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
