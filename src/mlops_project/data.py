import os
from pathlib import Path

import lightning as L
import torch
import typer
from torch.utils.data import Dataset


def normalize(images: torch.Tensor) -> torch.Tensor:
    """Normalize images."""
    return (images - images.mean()) / images.std()


class MyDataset(Dataset):
    """My custom dataset."""

    def __init__(self, data_path: Path, train: bool = True) -> None:
        self.data_path = data_path
        self.train = train

        prefix = "train" if self.train else "test"
        self.images = torch.load(data_path / f"{prefix}_images.pt")
        self.target = torch.load(data_path / f"{prefix}_target.pt")

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.images)

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""
        return self.images[index], self.target[index]

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""


def preprocess(data_path: Path, output_folder: Path) -> None:
    print("Preprocessing data...")
    dataset = MyDataset(data_path)
    dataset.preprocess(output_folder)


def preprocess_data(raw_dir: str, processed_dir: str) -> None:
    """Process raw data and save it to processed directory."""
    train_images, train_target = [], []
    for i in range(6):
        train_images.append(torch.load(f"{raw_dir}/train_images_{i}.pt"))
        train_target.append(torch.load(f"{raw_dir}/train_target_{i}.pt"))
    train_images = torch.cat(train_images)
    train_target = torch.cat(train_target)

    test_images: torch.Tensor = torch.load(f"{raw_dir}/test_images.pt")
    test_target: torch.Tensor = torch.load(f"{raw_dir}/test_target.pt")

    train_images = train_images.unsqueeze(1).float()
    test_images = test_images.unsqueeze(1).float()
    train_target = train_target.long()
    test_target = test_target.long()

    train_images = normalize(train_images)
    test_images = normalize(test_images)

    os.makedirs(processed_dir, exist_ok=True)
    torch.save(train_images, f"{processed_dir}/train_images.pt")
    torch.save(train_target, f"{processed_dir}/train_target.pt")
    torch.save(test_images, f"{processed_dir}/test_images.pt")
    torch.save(test_target, f"{processed_dir}/test_target.pt")


def corrupt_mnist(
    data_path: str = "data/processed",
) -> tuple[torch.utils.data.TensorDataset, torch.utils.data.TensorDataset]:
    """Return train and test datasets for corrupt MNIST."""
    train_images = torch.load(f"{data_path}/train_images.pt")
    train_target = torch.load(f"{data_path}/train_target.pt")
    test_images = torch.load(f"{data_path}/test_images.pt")
    test_target = torch.load(f"{data_path}/test_target.pt")

    train_set = torch.utils.data.TensorDataset(train_images, train_target)
    test_set = torch.utils.data.TensorDataset(test_images, test_target)
    return train_set, test_set


class CorruptMNISTDataModule(L.LightningDataModule):
    """LightningDataModule for Corrupt MNIST dataset."""

    def __init__(self, data_path: str = "data/processed", batch_size: int = 64, num_workers: int = 4):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str | None = None) -> None:
        """Setup datasets."""
        train_set, test_set = corrupt_mnist(self.data_path)
        self.train_set = train_set
        self.test_set = test_set

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Return training dataloader."""
        return torch.utils.data.DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """Return validation dataloader."""
        return torch.utils.data.DataLoader(
            self.test_set, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        """Return test dataloader."""
        return torch.utils.data.DataLoader(
            self.test_set, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True
        )


if __name__ == "__main__":
    # typer.run(preprocess)
    typer.run(preprocess_data)
