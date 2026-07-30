"""fltabular: Flower Example on Adult Census Income Tabular Dataset (Clean FL Version)."""

from pathlib import Path
import math
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.regression import (
    MeanAbsoluteError,
    MeanSquaredError,
    R2Score,
)


# =========================================================
# GLOBAL PREPROCESSING (FIT ONCE — IMPORTANT FOR FL)
# =========================================================

DATA_PATH = Path("/workspaces/flower/fl-tabular/db.csv")
RAW_DF = pd.read_csv(DATA_PATH)
RAW_DF, test_df = train_test_split(RAW_DF, 
                                   test_size=0.2, 
                                #    random_state=42,
                                   stratify=RAW_DF["CRF01"])

PARTITION_BY = "CRF01"
CLIENTS = sorted(RAW_DF[PARTITION_BY].unique())

# Remove partition key globally AFTER defining clients
RAW_DF = RAW_DF.drop(columns=[PARTITION_BY])

# Split features/target
X_FULL = RAW_DF.drop("COST_BL", axis=1)
y_FULL = RAW_DF["COST_BL"]

# Identify categorical columns
CATEGORICAL_COLS = X_FULL.select_dtypes(include=["object"]).columns

# Fit encoders ONCE globally
ENCODER = OrdinalEncoder()
X_FULL[CATEGORICAL_COLS] = ENCODER.fit_transform(X_FULL[CATEGORICAL_COLS])

SCALER = StandardScaler()
X_FULL[X_FULL.columns] = SCALER.fit_transform(X_FULL)


# =========================================================
# PARTITIONING
# =========================================================

def _load_local_partition(partition_id: int, num_partitions: int):
    if len(CLIENTS) != num_partitions:
        raise ValueError(
            f"Expected {num_partitions} clients, found {len(CLIENTS)}"
        )

    client_id = CLIENTS[partition_id]
    df_client = RAW_DF.copy()

    # rebuild client partition using original data
    full_df = pd.read_csv(DATA_PATH)
    df_client = full_df[full_df[PARTITION_BY] == client_id].drop(columns=[PARTITION_BY])

    return df_client


def load_data(partition_id: int, num_partitions: int):

    dataset = _load_local_partition(partition_id, num_partitions)
    dataset = dataset.dropna()

    X = dataset.drop("COST_BL", axis=1)
    y = dataset["COST_BL"]

    # Apply GLOBAL preprocessing (IMPORTANT)
    X[CATEGORICAL_COLS] = ENCODER.transform(X[CATEGORICAL_COLS])
    X[X.columns] = SCALER.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, 
        # random_state=42
    )

    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)

    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    train_loader = DataLoader(
        TensorDataset(X_train_tensor, y_train_tensor),
        batch_size=8,
        shuffle=True,
    )

    test_loader = DataLoader(
        TensorDataset(X_test_tensor, y_test_tensor),
        batch_size=8,
        shuffle=False,
    )

    return train_loader, test_loader


# =========================================================
# MODEL (LINEAR REGRESSION — YOUR CHOICE)
# =========================================================

class CostRegressor(nn.Module):
    def __init__(self, input_dim: int = 13):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.layer1(x)


# =========================================================
# TRAINING
# =========================================================

def trainer(model, train_loader, num_epochs=10):
    criterion = nn.L1Loss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, 
                           momentum=0.9
                           )

    model.train()

    for _ in range(num_epochs):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()


# =========================================================
# EVALUATION (CLEAN + STABLE)
# =========================================================

def evaluator(model, test_loader):
    model.eval()

    mae = MeanAbsoluteError()
    mse = MeanSquaredError(squared=True)
    rmse = MeanSquaredError(squared=False)
    r2 = R2Score()

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            preds = model(X_batch).reshape(-1)
            target = y_batch.reshape(-1)

            mae.update(preds, target)
            mse.update(preds, target)
            rmse.update(preds, target)
            r2.update(preds, target)

    return (
        mae.compute().item(),
        mse.compute().item(),
        rmse.compute().item(),
        r2.compute().item(),
    )