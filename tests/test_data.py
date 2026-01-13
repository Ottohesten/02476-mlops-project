import torch
from torch.utils.data import Dataset
import pytest
from mlops_project.data import MyDataset, corrupt_mnist

@pytest.fixture
def mock_data(tmp_path):
    """Create mock data for testing."""
    # Create a small dataset
    N_train = 100
    N_test = 20
    
    # Create random images: (N, 1, 28, 28)
    train_images = torch.randn(N_train, 1, 28, 28)
    test_images = torch.randn(N_test, 1, 28, 28)
    
    # Create target (0-9) ensuring all classes are present
    train_target = torch.randint(0, 10, (N_train,))
    # Force first 10 to be 0-9 to ensure all classes exist for the test
    train_target[:10] = torch.arange(0, 10)
    
    test_target = torch.randint(0, 10, (N_test,))
    test_target[:10] = torch.arange(0, 10)
    
    # Save to tmp_path
    torch.save(train_images, tmp_path / "train_images.pt")
    torch.save(train_target, tmp_path / "train_target.pt")
    torch.save(test_images, tmp_path / "test_images.pt")
    torch.save(test_target, tmp_path / "test_target.pt")
    
    return tmp_path

def test_my_dataset(mock_data):
    """Test the MyDataset class."""
    dataset = MyDataset(data_path=mock_data, train=True)
    assert isinstance(dataset, Dataset)
    assert len(dataset) == 100
    
    sample, target = dataset[0]
    assert sample.shape == (1, 28, 28)
    
    # Test test split
    dataset_test = MyDataset(data_path=mock_data, train=False)
    assert len(dataset_test) == 20

def test_corrupt_mnist(mock_data):
    """Test the corrupt_mnist function."""
    dataset_train, dataset_test = corrupt_mnist(data_path=str(mock_data))
    
    assert len(dataset_train) == 100
    assert len(dataset_test) == 20
    
    # Check shapes
    x, y = dataset_train[0]
    assert x.shape == (1, 28, 28)
    assert y in range(10)
    
    # Check all classes are present
    train_targets = torch.unique(dataset_train.tensors[1])
    # Sort to compare
    train_targets, _ = torch.sort(train_targets)
    assert (train_targets == torch.arange(0, 10)).all(), "Training targets do not cover all classes 0-9"
