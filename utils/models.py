# This file contains the model classes we will use for training and evaluation.

# General
import sys
import os
import argparse
import numpy as np

# Logging
import logging
import colorama
from utils.log_formatter import ColoredModuleFormatter

# PyTorch
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
from torch_geometric.nn import EdgeConv
from torch_geometric.data import Data
from torch_cluster import knn_graph

# Other
import itertools
from sklearn.cluster import DBSCAN
from sklearn.metrics import roc_auc_score

# Custom utils
import utils.common as common

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


"""
- data.x -> node feature matrix of shape [num_nodes, num_node_features] <=> R^{N x 3}
- data.pos -> raw 3D coorindates, used to define edges via DBSCAN
- edge_index -> a long tensor of shape [2, num_edges] <=> R^{2 x E}
    - (node1, node2) is an edge from node1 to node2
- edge_attr -> (todo) edge features tensor of shape [num_edges, num_edge_features] <=> R^{E x F_e}
    - ex. edge features could be the distance between the two nodes, or the angle between them, etc.
    - explore this ASAP, other papers make use of edge features
- batch -> a 1D long tensor of shape [num_nodes * num_batches] <=> R^{N x 1}
    - batch[n] = g tells you that node index n belongs to graph g
    - g = 0, 1, 2, ..., num_batches-1
"""


# data.x -> node feature matrix of shape [num_nodes, num_node_features] <=> R^{N x 3}
# data.pos -> raw 3D coorindates, used to define edges via DBSCAN
# edge_index -> a long tensor of shape [2, num_edges] <=> R^{2 x E}
#   - (node1, node2) is an edge from node1 to node2
# edge_attr -> (todo) edge features tensor of shape [num_edges, num_edge_features] <=> R^{E x F_e}
#   - ex. edge features could be the distance between the two nodes, or the angle between them, etc.
#   - explore this ASAP, other papers make use of edge features
# batch -> a 1D long tensor of shape [num_nodes * num_batches] <=> R^{N x 1}
#   - batch[n] = g tells you that node index n belongs to graph g
#   - g = 0, 1, 2, ..., 

def _dbscan_edge_index(pos, batch, eps, min_samples, metric='euclidean'):
    """
    Build edge_index by running DBSCAN separately on each graph in the batch.
    Connect every pair of nodes within the same DBSCAN cluster (ignoring noise points).

    - Batch index prevents cross-graph edges.
    - Specify custom metric function or use string name for a built-in sklearn pairwise metrics.

    Args:
        pos (torch.Tensor): Node positions of shape (N, 3).
        batch (torch.Tensor): Batch vector of shape (N,).
        eps (float): DBSCAN eps parameter.
        min_samples (int): DBSCAN min_samples parameter.
        metric (str): Distance metric for DBSCAN.
    """

    edges = []
    # Iteate over each graph in the batch
    for graph_id in batch.unique().tolist():
        mask = (batch == graph_id)
        idxs = mask.nonzero(as_tuple=False).view(-1)
        # pos is (N, 3) torch.Tensor, move to CPU numpy
        coords = pos[mask].cpu().numpy() # (N, 3) x y z
        # cluster in eta-phi space, but apply cluster to 3D cartesian coords
        labels = DBSCAN(eps=eps, min_samples=min_samples, metric=metric).fit_predict(coords)
        # For each cluster, connect every pair of nodes
        for cluster_id in set(labels):
            if cluster_id == -1:
                continue # skip noise points
            members = idxs[labels == cluster_id].tolist()
            # Add directed edges for every pair in this cluster
            for i, j in itertools.permutations(members, 2):
                edges.append((i, j))
    if not edges: # len==0
        # Fallback to no edges
        logger.warning("No edges found in this batch.")
        edge_index = torch.empty((2, 0), dtype=torch.long, device=pos.device)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long, device=pos.device).t().contiguous()
    return edge_index



# if cluster has n nodes, there will be n(n-1) directed edges
class GCNEdgeClassifier(torch.nn.Module):
    """2‐layer GCN producing node embeddings; edges scored by dot‐product."""
    def __init__(self, hidden_channels=64, eps=0.1, min_samples=5, **kwargs):
        super().__init__()
        self.eps = eps
        self.min_samples = min_samples
        self.conv1 = GCNConv(3, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels * 2, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, 1)
        )
        self.metric = kwargs.get('metric', 'euclidean') # DBSCAN metric
        
    def forward(self, data):
        """Forward pass for the GCN edge classifier.

        Args:
            data (_type_): _description_

        Returns:
            _type_: _description_
        """
        x, pos, batch = data.x, data.pos, data.batch

        # 1) Build edges within each DBSCAN cluster
        edge_index = _dbscan_edge_index(pos, batch, self.eps, self.min_samples, self.metric)

        # 2) Standard GCN layers on this custom graph
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))

        # 3) Edge-classification MLP
        row, col = edge_index
        edge_features = torch.cat([h[row], h[col]], dim=1)
        return self.edge_mlp(edge_features).view(-1) # shape (E, 1)

        
class GATEdgeClassifier(torch.nn.Module):
    """GAT model for edge classification.
    This model uses a GATConv layer to compute node embeddings, and then
    classifies edges based on the concatenation of the embeddings of the two
    nodes connected by the edge.
    """
    def __init__(self, hidden_channels=32, eps=0.1, min_samples=5, heads=4, **kwargs):
        super().__init__()
        self.eps = eps
        self.min_samples = min_samples
        self.gat1 = GATConv(3, hidden_channels, heads=heads, concat=True)
        self.gat2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False)
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, 1)
        )
        self.metric = kwargs.get('metric', 'euclidean')

    def forward(self, data):
        x, pos, batch = data.x, data.pos, data.batch

        # 1) Cluster with DBSCAN
        edge_index = _dbscan_edge_index(pos, batch, self.eps, self.min_samples, self.metric)

        # 2) Node embeddings via GAT
        h = F.relu(self.gat1(x, edge_index))
        h = F.relu(self.gat2(h, edge_index))

        # 3) Build edge features and classify
        row, col = edge_index
        edge_features = torch.cat([h[row], h[col]], dim=1)
        return self.edge_mlp(edge_features).view(-1)


class EdgeConvEdgeClassifier(torch.nn.Module):
    """EdgeConv model for edge classification.
    This model uses an EdgeConv layer to compute node embeddings, and then
    classifies edges based on the concatenation of the embeddings of the two
    nodes connected by the edge.
    """
    def __init__(self, hidden_channels=32, eps=0.1, min_samples=5, **kwargs):
        super().__init__()
        self.eps = eps
        self.min_samples = min_samples
        self.conv1 = EdgeConv(nn=torch.nn.Sequential(
            torch.nn.Linear(2*3, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels),
        ))
        self.conv2 = EdgeConv(nn=torch.nn.Sequential(
            torch.nn.Linear(2*hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels),
        ))
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, 1)
        )
        self.metric = kwargs.get('metric', 'euclidean')

    def forward(self, data):
        x, pos, batch = data.x, data.pos, data.batch

        # 1) Cluster with DBSCAN
        edge_index = _dbscan_edge_index(pos, batch, self.eps, self.min_samples, self.metric)

        # 2) Node embeddings via EdgeConv
        h = F.relu(self.conv1(x, edge_index))

        # 3) Re-cluster with DBSCAN
        edge_index = _dbscan_edge_index(h, batch, self.eps, self.min_samples, self.metric)

        # 4) Node embeddings via EdgeConv
        h = F.relu(self.conv2(h, edge_index))

        # 3) Build edge features and classify
        row, col = edge_index
        edge_features = torch.cat([h[row], h[col]], dim=1)
        return self.edge_mlp(edge_features).view(-1)