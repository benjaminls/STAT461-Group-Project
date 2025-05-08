# This file contains the model classes we will use for training and evaluation.

# General
import sys
import os
import argparse

# Logging
import logging
import colorama
from utils.log_formatter import ColoredModuleFormatter

# GNN
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score

# Custom utils

# Set up logging
def _init_logger() -> logging.Logger:
    """Set up logger for a script.

    Args:
        args (argparse.Namespace): _description_
    """

    console_handler = logging.StreamHandler()
    level = getattr(logging, os.environ.get("PYTHON_GNN_LOG_LEVEL", "INFO").upper(), None)
    if level is None:
        raise ValueError(f"Invalid log level: {os.environ.get('PYTHON_GNN_LOG_LEVEL')}")
    console_handler.setLevel(level)
    formatter = ColoredModuleFormatter("%(message)s")
    console_handler.setFormatter(formatter)
    new_logger = logging.getLogger(__name__)
    new_logger.addHandler(console_handler)
    new_logger.setLevel(level)
    return new_logger

logger = _init_logger()
