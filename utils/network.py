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
from torch_geometric.nn import GCNConv

from utils.graph import create_data
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


class TrackGNN(torch.nn.Module):
    """3‐layer GCN producing node embeddings; edges scored by dot‐product."""
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(3, 64)
        self.conv2 = GCNConv(64, 32)
        self.conv3 = GCNConv(32, 16)

    def forward(self, x, edge_index):
        # Node embeddings
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        h = self.conv3(h, edge_index)

        # Score each edge by dot(src, dst)
        src, dst = edge_index
        score = (h[src] * h[dst]).sum(dim=1)
        return score


def train_epoch(model, data, optimizer, criterion, device):
    model.train()
    optimizer.zero_grad()
    scores = model(data.x.to(device), data.edge_index.to(device))
    loss = criterion(scores[data.train_mask],
                     data.y[data.train_mask].to(device))
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate(model, data, mask, device):
    model.eval()
    with torch.no_grad():
        scores = model(data.x.to(device), data.edge_index.to(device))
        logger.debug(f"Scores: {scores[mask].cpu()}")
        probs = torch.sigmoid(scores.cpu())
        logger.debug(f"Predicted probabilities: {probs[mask]}")
        logger.debug(f"True labels: {data.y[mask].cpu()}")
        y_true = data.y[mask].cpu().numpy()
        y_pred = probs[mask].cpu().numpy()
        # auc = roc_auc_score(data.y[mask].cpu(), probs[mask])
        auc = roc_auc_score(y_true, y_pred)
        logger.debug(f"Evaluating AUC: {auc:.4f}")
    return auc


def run_training(hits_csv: str,
                 truth_csv: str,
                 epochs: int = 50,
                 lr: float = 1e-3):
    """Full training loop with periodic AUC reporting."""
    data = create_data(hits_csv, truth_csv)
    logger.info(f"Data: {data}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    model = TrackGNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    for ep in range(1, epochs + 1):
        loss = train_epoch(model, data, optimizer, criterion, device)
        if ep == 1 or ep % 10 == 0:
            train_auc = evaluate(model, data, data.train_mask, device)
            val_auc   = evaluate(model, data, data.val_mask,   device)
            print(f"Epoch {ep:03d} | Loss {loss:.4f} | "
                  f"Train AUC {train_auc:.4f} | Val AUC {val_auc:.4f}")

    test_auc = evaluate(model, data, data.test_mask, device)
    print(f"\nFinal Test ROC AUC: {test_auc:.4f}")