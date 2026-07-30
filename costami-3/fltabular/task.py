"""fltabular: Flower ClientApp/ServerApp — GLM for cost regression with central DP."""

from functools import lru_cache

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

SEED = 42


class GLM(nn.Module):
    """Single-layer GLM == linear regression with identity link (cost ~ features)."""

    def __init__(self, num_features: int):
        super().__init__()
        self.linear = nn.Linear(num_features, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


@lru_cache(maxsize=4)
def _load_full_dataframe(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


@lru_cache(maxsize=4)
def _global_scaler(csv_path: str, feature_cols: tuple) -> StandardScaler:
    """Fit one StandardScaler on the full dataset so every client normalises
    features consistently. NOTE: this is a simulation-only convenience — in a
    real deployment each client would only see its own local statistics, or
    you'd use a federated/precomputed global mean & std instead."""
    df = _load_full_dataframe(csv_path)
    scaler = StandardScaler()
    scaler.fit(df[list(feature_cols)].values)
    return scaler


def _partition_groups(csv_path: str, group_col: str, num_partitions: int) -> list:
    df = _load_full_dataframe(csv_path)
    groups = sorted(df[group_col].unique().tolist())
    if num_partitions > len(groups):
        raise ValueError(
            f"num_partitions ({num_partitions}) > number of unique "
            f"'{group_col}' values ({len(groups)})"
        )
    # Contiguous chunks of sorted group IDs -> partitions are naturally
    # non-uniform in size, matching your real grouping distribution.
    return [c.tolist() for c in np.array_split(np.array(groups), num_partitions)]


def load_data(
    csv_path: str,
    group_col: str,
    target_col: str,
    partition_id: int,
    num_partitions: int,
    batch_size: int = 8,
):
    """Return train/val/test DataLoaders for one client.

    Split policy applied within each client's local slice:
      80/20 -> train_full / test   (test held out, used only for final round)
      80/20 -> train / val         (val used for per-round evaluation)
    """
    df = _load_full_dataframe(csv_path)
    feature_cols = tuple(
        c for c in df.columns if c not in (group_col, target_col)
    )

    group_ids = _partition_groups(csv_path, group_col, num_partitions)[partition_id]
    local_df = df[df[group_col].isin(group_ids)].reset_index(drop=True)

    train_full, test_df = train_test_split(
        local_df, test_size=0.2, random_state=SEED, shuffle=True
    )
    train_df, val_df = train_test_split(
        train_full, test_size=0.2, random_state=SEED, shuffle=True
    )

    scaler = _global_scaler(csv_path, feature_cols)

    def to_loader(split_df: pd.DataFrame, shuffle: bool) -> DataLoader:
        X = scaler.transform(split_df[list(feature_cols)].values).astype(np.float32)
        y = split_df[target_col].values.astype(np.float32)
        ds = TensorDataset(torch.tensor(X), torch.tensor(y))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader = to_loader(train_df, shuffle=True)
    val_loader = to_loader(val_df, shuffle=False)
    test_loader = to_loader(test_df, shuffle=False)
    return train_loader, val_loader, test_loader, len(feature_cols)


def train_fn(net: nn.Module, trainloader: DataLoader, epochs: int, lr: float, device):
    net.to(device)
    net.train()
    criterion = nn.L1Loss()  # MAE loss, as requested
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)

    running_loss, n_batches = 0.0, 0
    for _ in range(epochs):
        for X, y in trainloader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(net(X), y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1
    return running_loss / max(n_batches, 1)


@torch.no_grad()
def test_fn(net: nn.Module, dataloader: DataLoader, device):
    net.to(device)
    net.eval()
    criterion = nn.L1Loss()

    preds, targets = [], []
    total_loss, n_batches = 0.0, 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        out = net(X)
        total_loss += criterion(out, y).item()
        n_batches += 1
        preds.append(out.cpu())
        targets.append(y.cpu())

    y_pred = torch.cat(preds).numpy()
    y_true = torch.cat(targets).numpy()

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss, {"mae": mae, 
                      "rmse": rmse, 
                      "r2": r2, 
                      "num_examples": len(y_true)}