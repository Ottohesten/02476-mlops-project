import os
from pathlib import Path

import torch
from torch.utils.data import Dataset

from mlops_project.data import MyDataset, corrupt_mnist


def test_my_dataset():
    """Test the MyDataset class."""
    dataset = MyDataset(Path("data/processed"))
    assert isinstance(dataset, Dataset)


def test_data():
    """Test the data loading and preprocessing."""
    N_train = 30000
    N_test = 5000
    dataset_train, dataset_test = corrupt_mnist()
    assert len(dataset_train) == N_train, "Dataset did not have expected number of training samples"
    assert len(dataset_test) == N_test, "Dataset did not have expected number of test samples"
    for dataset in [dataset_train, dataset_test]:
        for x, y in dataset:
            assert x.shape == (1, 28, 28), f"Expected image shape (1, 28, 28), got {x.shape}"
            assert y in range(10), f"Expected target in range 0-9, got {y}"
    train_targets = torch.unique(dataset_train.tensors[1])
    assert (train_targets == torch.arange(0, 10)).all(), "Training targets do not cover all classes 0-9"
    test_targets = torch.unique(dataset_test.tensors[1])
    assert (test_targets == torch.arange(0, 10)).all(), "Test targets do not cover all classes 0-9"
