import os
from pathlib import Path

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


def corrupt_mnist(data_path: str = "data/processed") -> tuple[torch.utils.data.TensorDataset, torch.utils.data.TensorDataset]:
    """Return train and test datasets for corrupt MNIST."""
    train_images = torch.load(f"{data_path}/train_images.pt")
    train_target = torch.load(f"{data_path}/train_target.pt")
    test_images = torch.load(f"{data_path}/test_images.pt")
    test_target = torch.load(f"{data_path}/test_target.pt")

    train_set = torch.utils.data.TensorDataset(train_images, train_target)
    test_set = torch.utils.data.TensorDataset(test_images, test_target)
    return train_set, test_set


if __name__ == "__main__":
    # typer.run(preprocess)
    typer.run(preprocess_data)
