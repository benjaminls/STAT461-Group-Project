import os
import logging
import argparse
import numpy as np

import torch
import torch_geometric
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import StepLR

from models.



from torch_geometric.data import Data, DataLoader

from standin import StandInDataset
from dataset import GraphDataset


# ==========
# Some parts of this code are adapted, inspired, or borrowed from:
# https://github.com/GageDeZoort/interaction_network_paper/tree/pytorch_geometric
# The novel and substantive parts of this code are our own.
# ==========

partion = {
    "train": graph_files[IDs:1000],
    "val": graph_files[1000:1400],
    "test": graph_files[1400:1500],
}

params = {
    "batch_size": 1,
    "shuffle": False,
    # "num_workers": 0, # options mentioned in IN code, comment out, understand later
    # "pin_memory": True, # understand later
}

train_set = StandInDataset()
train_loader = DataLoader(train_set, **params)
val_set = StandInDataset()
val_loader = DataLoader(val_set, **params)
test_set = StandInDataset()
test_loader = DataLoader(test_set, **params)