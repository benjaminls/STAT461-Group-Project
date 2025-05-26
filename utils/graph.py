"""
Functions for constructing and manipulating graphs from detector hit data.

This module provides utilities to build graph representations of particle detector data,
including clustering hits into connected components and creating edge connections between nodes.
The resulting graphs are compatible with PyTorch Geometric's Data format for use in GNN models.

Types of graphs:
- Baseline: 
    - Nodes: Hits
    - Node features: (x,y,z)
    - Edges: All pairs in cluster
    - Edge features: 1 if particles match, 0 otherwise
    - Split: 70/15/15 train/val/test
"""


import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from utils.parse import load_hits, load_truth
from utils.log_formatter import ColoredModuleFormatter

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


def build_graph(hits_df: pd.DataFrame):
    """
    Run DBSCAN on (x,y,z), assign cluster labels,
    then fully connect hits within each cluster (skip noise).
    Returns:
      x           Tensor[#nodes×3] of node coords
      edge_index Tensor[2×#edges] of undirected edges
      hits_df    DataFrame with an added 'cluster' column
    """
    coords = hits_df[['x', 'y', 'z']].values
    db = DBSCAN()  # default eps=0.5, min_samples=5
    clusters = db.fit_predict(coords)
    hits_df['cluster'] = clusters

    x = torch.tensor(coords, dtype=torch.float)

    # Build undirected edges
    edge_list = []
    for cl in np.unique(clusters):
        if cl == -1:
            continue  # noise
        idxs = np.where(clusters == cl)[0]
        for i in idxs:
            for j in idxs:
                if i < j:
                    edge_list.append([i, j])
                    edge_list.append([j, i])

    edge_index = (
        torch.tensor(edge_list, dtype=torch.long)
             .t()
             .contiguous()
    )
    return x, edge_index, hits_df


def create_edge_labels(edge_index: torch.Tensor,
                       hits_df: pd.DataFrame,
                       truth_df: pd.DataFrame) -> torch.Tensor:
    """
    For each edge (i→j), label as 1.0 if hits_df.particle_id[i] ==
    hits_df.particle_id[j], else 0.0.
    """
    # Map hit_id → node index
    hit_to_idx = {hid: idx for idx, hid in enumerate(hits_df['hit_id'].values)}
    # Map node index → particle_id
    particle_map = {}
    for _, row in truth_df.iterrows():
        hid = int(row['hit_id'])
        if hid in hit_to_idx:
            particle_map[hit_to_idx[hid]] = int(row['particle_id'])

    src, dst = edge_index
    labels = []
    for i, j in zip(src.tolist(), dst.tolist()):
        pid_i = particle_map.get(i, -1)
        pid_j = particle_map.get(j, -2)
        labels.append(1.0 if pid_i == pid_j else 0.0)

    return torch.tensor(labels, dtype=torch.float)


def create_data(hits_csv: str, truth_csv: str) -> Data:
    """
    Builds a PyG Data object with:
      - x: node features [#nodes×3]
      - edge_index: [2×#edges]
      - y: edge labels [#edges]
      - train/val/test masks on edges (70/15/15)
    """
    hits_df  = load_hits(hits_csv)
    truth_df = load_truth(truth_csv)

    logger.debug(f"Hits DataFrame: \n{hits_df.head()}")
    logger.debug(f"Truth DataFrame: \n{truth_df.head()}")

    x, edge_index, hits_df = build_graph(hits_df)
    y = create_edge_labels(edge_index, hits_df, truth_df)

    logger.debug(f"Node features (x): \n{x}")
    logger.debug(f"Edge index: \n{edge_index}")
    logger.debug(f"Edge labels (y): \n{y}")

    data = Data(x=x, edge_index=edge_index, y=y)

    # Random edge split
    num = y.size(0)
    perm = torch.randperm(num)
    n_train = int(0.70 * num)
    n_val   = int(0.15 * num)

    train_idx = perm[:n_train]
    val_idx   = perm[n_train:n_train+n_val]
    test_idx  = perm[n_train+n_val:]

    data.train_mask = torch.zeros(num, dtype=torch.bool)
    data.val_mask   = torch.zeros(num, dtype=torch.bool)
    data.test_mask  = torch.zeros(num, dtype=torch.bool)

    data.train_mask[train_idx] = True
    data.val_mask[val_idx]     = True
    data.test_mask[test_idx]   = True

    return data
