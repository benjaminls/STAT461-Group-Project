import os
import logging
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Dataset, Data
from torch_geometric.utils import to_undirected
from torch_geometric.transforms import ToUndirected

# ==========
# Some parts of this code are adapted, inspired, or borrowed from:
# https://github.com/GageDeZoort/interaction_network_paper/tree/pytorch_geometric
# The novel and substantive parts of this code are our own.
# ==========


class GraphDataset(Dataset):
    """Dataset for graph data."""

    def __init__(self, transform=None, pre_transform=None, graph_files=[]):
        super().__init__(None, transform, pre_transform)
        self.graph_files = graph_files

    @property
    def raw_file_names(self):
        return self.graph_files

    @property
    def processed_file_names(self):
        return []

    def len(self):
        return len(self.graph_files)

    def get(self, idx):
        with np.load(self.graph_files[idx]) as file:
            x = torch.from_numpy(file["x"])
            edge_index = torch.from_numpy(file["edge_index"])
            edge_attr = torch.from_numpy(file["edge_attr"])
            y = torch.from_numpy(file["y"])
            pid = torch.from_numpy(file["pid"])
            pt = torch.from_numpy(file["pt"]) if "pt" in file else 0
            eta = torch.from_numpy(file["eta"]) if "eta" in file else 0

            # Convert to undirected graph
            row, col = edge_index
            row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
            edge_index = torch.stack([row, col], dim=0)
            edge_attr = torch.cat([edge_attr, -1 * edge_attr], dim=1)
            y = torch.cat([y, y])

            # Create Data object
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=torch.transpose(edge_attr, 0, 1),
                y=y,
                pid=pid,
                pt=pt,
                eta=eta,
            )
            data.num_nodes = len(x)

        return data
