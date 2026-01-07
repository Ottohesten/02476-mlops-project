from pathlib import Path

from torch.utils.data import Dataset

from mlops_project.data import MyDataset


def test_my_dataset():
    """Test the MyDataset class."""
    dataset = MyDataset(Path("data/raw"))
    assert isinstance(dataset, Dataset)
