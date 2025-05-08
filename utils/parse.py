import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


def load_hits(hits_csv_path: str) -> pd.DataFrame:
    """Load hits CSV with columns [hit_id,x,y,z,...]."""
    return pd.read_csv(hits_csv_path)


def load_truth(truth_csv: str) -> pd.DataFrame:
    """Load truth CSV with columns [hit_id,particle_id,...]."""
    return pd.read_csv(truth_csv)


