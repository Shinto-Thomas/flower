"""fltabular: Flower Example on Adult Census Income Tabular Dataset."""

import os
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

DATA_PATH_ENV = "FL_TABULAR_DATA_PATH"
TARGET_COLUMN_ENV = "FL_TABULAR_TARGET_COLUMN"
PARTITION_COLUMN = "CRF01"
DEFAULT_TARGET_COLUMN = "COST_BL"
_partition_info_logged = False


def _get_dataset_path() -> Path:
    env_path = os.environ.get(DATA_PATH_ENV)
    if env_path:
        return Path(env_path).expanduser().resolve()

    candidates = sorted(Path.cwd().glob("*.pkl"))
    if len(candidates) == 1:
        return candidates[0]

    if not candidates:
        raise FileNotFoundError(
            f"No .pkl file found in {Path.cwd()}. Set {DATA_PATH_ENV} to the file path."
        )

    raise RuntimeError(
        f"Found multiple .pkl files in {Path.cwd()}. Set {DATA_PATH_ENV} to the file path."
    )


def _get_target_column(dataset: pd.DataFrame) -> str:
    target_column = os.environ.get(TARGET_COLUMN_ENV, DEFAULT_TARGET_COLUMN)
    if target_column not in dataset.columns:
        available_columns = ", ".join(map(str, dataset.columns))
        raise KeyError(
            f"Missing target column '{target_column}'. Available columns: {available_columns}"
        )
    return target_column


def get_input_dim() -> int:
    dataset = pd.read_pickle(_get_dataset_path())
    target_column = _get_target_column(dataset)
    return len(dataset.columns.difference([PARTITION_COLUMN, target_column]))


def load_data(partition_id: int, num_partitions: int):
    global _partition_info_logged

    dataset = pd.read_pickle(_get_dataset_path())

    if PARTITION_COLUMN not in dataset.columns:
        raise KeyError(f"Missing partition column '{PARTITION_COLUMN}'.")

    target_column = _get_target_column(dataset)
    partition_values = sorted(dataset[PARTITION_COLUMN].dropna().unique())
    if not partition_values:
        raise ValueError(f"No partition values found in '{PARTITION_COLUMN}'.")

    if not _partition_info_logged:
        print(
            f"Detected {len(partition_values)} unique '{PARTITION_COLUMN}' IDs: {partition_values}"
        )
        if num_partitions != len(partition_values):
            print(
                f"Simulation has {num_partitions} nodes, but data has {len(partition_values)} "
                f"'{PARTITION_COLUMN}' IDs. Partition mapping will cycle across IDs."
            )
        _partition_info_logged = True

    partition_index = partition_id % len(partition_values)
    partition_value = partition_values[partition_index]
    dataset = dataset[dataset[PARTITION_COLUMN] == partition_value].copy()

    dataset.dropna(inplace=True)
    dataset.drop(columns=[PARTITION_COLUMN], inplace=True)

    categorical_cols = dataset.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    if target_column in categorical_cols:
        categorical_cols.remove(target_column)
    ordinal_encoder = OrdinalEncoder()
    dataset[categorical_cols] = ordinal_encoder.fit_transform(dataset[categorical_cols])

    X = dataset.drop(target_column, axis=1)
    y = dataset[target_column]

    if PARTITION_COLUMN in X.columns:
        X = X.drop(columns=[PARTITION_COLUMN])

    y = pd.to_numeric(y, errors="coerce")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, numeric_features)]
    )

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    return train_loader, test_loader


class CostRegressor(nn.Module):
    def __init__(self, input_dim: int):
        super(CostRegressor, self).__init__()
        self.layer1 = nn.Linear(input_dim, 1)
        # self.layer2 = nn.Linear(128, 64)
        # self.output = nn.Linear(64, 1)
        # self.relu = nn.ReLU()

    def forward(self, x):
        # x = self.relu(self.layer1(x))
        # x = self.relu(self.layer2(x))
        return self.layer1(x)


def trainer(model, train_loader, num_epochs=10):
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.5)
    model.train()
    for epoch in range(num_epochs):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()


def evaluator(model, test_loader):
    model.eval()
    criterion = nn.L1Loss()
    loss = 0.0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            batch_loss = criterion(outputs, y_batch)
            loss += batch_loss.item()
            total += y_batch.size(0)
    mae = loss / len(test_loader)
    return mae, mae
