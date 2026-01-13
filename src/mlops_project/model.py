import pytorch_lightning as pl
import torch
from torch import nn


class MyAwesomeModel(pl.LightningModule):
    """Just a dummy model to show how to structure your code"""

    def __init__(self, learning_rate: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128, 10)

        self.loss_fn = nn.CrossEntropyLoss()

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

    def training_step(self, batch, batch_idx):
        """Training step."""
        img, target = batch
        y_pred = self(img)
        loss = self.loss_fn(y_pred, target)
        accuracy = (y_pred.argmax(dim=1) == target).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_accuracy", accuracy, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        img, target = batch
        y_pred = self(img)
        loss = self.loss_fn(y_pred, target)
        accuracy = (y_pred.argmax(dim=1) == target).float().mean()
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("validation_accuracy", accuracy, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        """Configure optimizers."""
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


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
