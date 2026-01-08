import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class LogisticRegression(nn.Module):
    def __init__(self, input_size: int, num_classes: int):
        """
        Multiclass Logistic Regression model.

        Args:
            input_size: Number of input features (e.g., 10 voting questions)
            num_classes: Number of output classes (e.g., 4 types of answers for the label)
        """
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x: Input tensor of shape (batch_size, input_size)

        Returns:
            Tensor of shape (batch_size, num_classes) with raw logits
        """
        features = x.view(x.size(0), -1)
        logits = self.linear(features)
        return features, logits

class MLPVote(nn.Module):
    def __init__(self, input_size: int, num_classes: int):
        """
        MLP model for voting data with 3 hidden layers.

        Args:
            input_size: Number of input features
            num_classes: Number of output classes
        """
        super(MLPVote, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 16)
        self.fc4 = nn.Linear(16, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_size)

        Returns:
            features: Output of fc3 layer (batch_size, 16)
            logits: Output logits (batch_size, num_classes)
        """
        # x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        features = self.relu(self.fc3(x))
        logits = self.fc4(features)
        return features, logits


def load_voting_csv(csv_file: str, feature_cols: list, label_col: str):
    """
    Load voting CSV file and convert to PyTorch tensors for logistic regression.

    All categorical features and labels are one-hot encoded.

    Args:
        csv_file: Path to CSV file
        feature_cols: List of column names for features (e.g., 10 questions)
        label_col: Column name of the label question

    Returns:
        X: Tensor of shape (num_samples, num_features * num_classes) (one-hot encoded)
        y: Tensor of shape (num_samples,) with class indices (0..num_classes-1)
    """
    df = pd.read_csv(csv_file)

    # One-hot encode features
    X_onehot_list = []
    for col in feature_cols:
        X_onehot = pd.get_dummies(df[col], prefix=col)
        X_onehot_list.append(X_onehot)
    X_df = pd.concat(X_onehot_list, axis=1)
    X = torch.tensor(X_df.values, dtype=torch.float32)

    # Convert label to integer class indices
    label_mapping = {val: idx for idx, val in enumerate(sorted(df[label_col].unique()))}
    y = df[label_col].map(label_mapping).values
    y = torch.tensor(y, dtype=torch.long)

    return X, y


def subsample_from_partitions(X, y, k, a, b, seed=None):
    """
    Uniformly subsample `a` examples from [0, k] and `b` examples from [k+1, N-1].

    Args:
        X (torch.Tensor): Feature tensor of shape (N, num_features).
        y (torch.Tensor): Label tensor of shape (N,).
        k (int): Cutpoint index (0-based).
        a (int): Number of samples to pick from the first partition [0, k].
        b (int): Number of samples to pick from the second partition [k+1, N-1].
        seed (int, optional): Random seed for reproducibility.

    Returns:
        X_sub (torch.Tensor): Subsampled features.
        y_sub (torch.Tensor): Subsampled labels.
    """
    N = len(X)
    if seed is not None:
        torch.manual_seed(seed)

    if k < 0 or k >= N:
        raise ValueError(f"Cutpoint k={k} must be between 0 and N-1.")

    # Define partition index ranges
    first_partition = torch.arange(0, k + 1)
    second_partition = torch.arange(k + 1, N)

    if a > len(first_partition):
        raise ValueError(f"Not enough samples in first partition to pick {a}.")
    if b > len(second_partition):
        raise ValueError(f"Not enough samples in second partition to pick {b}.")

    a_indices = first_partition[torch.randperm(len(first_partition))[:a]]
    b_indices = second_partition[torch.randperm(len(second_partition))[:b]]

    selected_indices = torch.cat([a_indices, b_indices])
    #selected_indices = selected_indices[torch.randperm(len(selected_indices))]  # optional shuffle

    X_sub = X[selected_indices]
    y_sub = y[selected_indices]

    return X_sub, y_sub, selected_indices


def create_dataloader(X, y, batch_size=64, shuffle=False):
    """
    Convert X, y tensors into a DataLoader.

    Args:
        X: Feature tensor
        y: Label tensor
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle data

    Returns:
        DataLoader object
    """
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


def extract_partition_dataloader(X, y, partition_array, partition_index, batch_size=64, shuffle=False):
    """
    Extract data belonging to a specific partition and return a DataLoader.

    Args:
        X: Feature tensor of shape (N, num_features)
        y: Label tensor of shape (N,)
        partition_array: 1D array of integers of length N, where each entry indicates which partition that data point belongs to
        partition_index: The partition index to extract
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle data

    Returns:
        DataLoader containing only data from the specified partition
    """
    mask = partition_array == partition_index

    X_partition = X[mask]
    y_partition = y[mask]

    dataloader = create_dataloader(X_partition, y_partition, batch_size, shuffle)

    return dataloader
