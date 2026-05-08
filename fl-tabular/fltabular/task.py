"""fltabular: Flower Example on Adult Census Income Tabular Dataset."""

import os
import math
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

TRAIN_DATA_PATH_ENV = "FL_TABULAR_TRAIN_DATA_PATH"
VAL_DATA_PATH_ENV = "FL_TABULAR_VAL_DATA_PATH"
TARGET_COLUMN_ENV = "FL_TABULAR_TARGET_COLUMN"
DEFAULT_TARGET_COLUMN = "COST_BL"
IGNORED_COLUMNS = {"CRF01"}
_preprocessors: dict[int, ColumnTransformer] = {}


def _get_dataset_path(split: str) -> Path:
    env_name = TRAIN_DATA_PATH_ENV if split == "train" else VAL_DATA_PATH_ENV
    default_name = "train_db.pkl" if split == "train" else "val_db.pkl"

    env_path = os.environ.get(env_name)
    if env_path:
        return Path(env_path).expanduser().resolve()

    default_path = Path.cwd() / default_name
    if default_path.exists():
        return default_path.resolve()

    candidates = sorted(Path.cwd().glob("*.pkl"))
    if len(candidates) == 1:
        return candidates[0]

    if not candidates:
        raise FileNotFoundError(
            f"No .pkl file found in {Path.cwd()}. Set {env_name} or place {default_name} next to the app."
        )

    raise RuntimeError(
        f"Found multiple .pkl files in {Path.cwd()}. Set {env_name} to the file path."
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
    dataset = pd.read_pickle(_get_dataset_path("train"))
    target_column = _get_target_column(dataset)
    feature_columns = [column for column in dataset.columns if column not in {target_column, *IGNORED_COLUMNS}]
    return len(feature_columns)


def _load_split_dataframe(split: str) -> tuple[pd.DataFrame, str]:
    dataset = pd.read_pickle(_get_dataset_path(split))
    dataset.dropna(inplace=True)
    target_column = _get_target_column(dataset)
    return dataset, target_column


def _partition_rows(dataset: pd.DataFrame, partition_id: int, num_partitions: int) -> pd.DataFrame:
    if num_partitions < 1:
        raise ValueError("num_partitions must be at least 1")

    shuffled = dataset.sample(frac=1, random_state=42).reset_index(drop=True)
    partition_indices = np.array_split(np.arange(len(shuffled)), num_partitions)
    selected_indices = partition_indices[partition_id % len(partition_indices)]
    return shuffled.iloc[selected_indices].copy()


def _build_preprocessor(feature_frame: pd.DataFrame) -> ColumnTransformer:
    categorical_cols = feature_frame.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()
    numeric_cols = feature_frame.select_dtypes(include=["number", "bool"]).columns.tolist()

    transformers = []
    if categorical_cols:
        transformers.append(
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                categorical_cols,
            )
        )
    if numeric_cols:
        transformers.append(("num", StandardScaler(), numeric_cols))

    return ColumnTransformer(transformers=transformers)


def load_data(split: str, partition_id: int, num_partitions: int):
    dataset, target_column = _load_split_dataframe(split)
    dataset = _partition_rows(dataset, partition_id, num_partitions)

    feature_frame = dataset.drop(columns=[target_column], errors="ignore")
    feature_frame = feature_frame.drop(columns=list(IGNORED_COLUMNS), errors="ignore")
    y = pd.to_numeric(dataset[target_column], errors="coerce")
    valid_rows = y.notna()
    feature_frame = feature_frame.loc[valid_rows].reset_index(drop=True)
    y = y.loc[valid_rows].reset_index(drop=True)

    cache_key = partition_id
    if split == "train":
        preprocessor = _build_preprocessor(feature_frame)
        X = preprocessor.fit_transform(feature_frame)
        _preprocessors[cache_key] = preprocessor
    else:
        preprocessor = _preprocessors.get(cache_key)
        if preprocessor is None:
            train_dataset, train_target = _load_split_dataframe("train")
            train_dataset = _partition_rows(train_dataset, partition_id, num_partitions)
            train_features = train_dataset.drop(columns=[train_target], errors="ignore")
            train_features = train_features.drop(columns=list(IGNORED_COLUMNS), errors="ignore")
            train_y = pd.to_numeric(train_dataset[train_target], errors="coerce")
            train_valid_rows = train_y.notna()
            train_features = train_features.loc[train_valid_rows].reset_index(drop=True)
            preprocessor = _build_preprocessor(train_features)
            preprocessor.fit(train_features)
            _preprocessors[cache_key] = preprocessor
        X = preprocessor.transform(feature_frame)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)

    dataset_tensor = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset_tensor, batch_size=8, shuffle=(split == "train"))


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
    optimizer = optim.Adam(model.parameters(), lr=1)
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
    absolute_error = 0.0
    squared_error = 0.0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            absolute_error += torch.sum(torch.abs(outputs - y_batch)).item()
            squared_error += torch.sum((outputs - y_batch) ** 2).item()
            total += y_batch.size(0)
    mae = absolute_error / total
    rmse = math.sqrt(squared_error / total)
    return mae, rmse
