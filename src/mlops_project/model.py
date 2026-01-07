import torch
from torch import nn


class MyAwesomeModel(nn.Module):
    """Just a dummy model to show how to structure your code"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = torch.relu(self.conv1(x))  # 28x28 -> 26x26
        x = torch.max_pool2d(x, 2, 2)  # 26x26 -> 13x13
        x = torch.relu(self.conv2(x))  # 13x13 -> 11x11
        x = torch.max_pool2d(x, 2, 2)  # 11x11 -> 5x5
        x = torch.relu(self.conv3(x))  # 5x5 -> 3x3
        x = torch.max_pool2d(x, 2, 2)  # 3x3 -> 1x1
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)


class ModelSequential(nn.Module):
    """A simple model using nn.Sequential."""

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)


if __name__ == "__main__":
    model = MyAwesomeModel()
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
