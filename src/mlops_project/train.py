import torch
from lightning.pytorch.cli import LightningCLI

from mlops_project.data import CorruptMNISTDataModule
from mlops_project.model import MyAwesomeModel


def cli_main():
    cli = LightningCLI(
        MyAwesomeModel, CorruptMNISTDataModule, seed_everything_default=42, save_config_kwargs={"overwrite": True}
    )


if __name__ == "__main__":
    cli_main()


# def train_model(learning_rate: float = 1e-3, batch_size: int = 64, epochs: int = 100) -> None:
#     """Train a model on MNIST."""
#     print("Training day and night")
#     print(f"{learning_rate=}, {batch_size=}, {epochs=}")

#     # Initialize model
#     model = MyAwesomeModel(learning_rate=learning_rate)

#     # Initialize DataLoader
#     train_set = MyDataset(data_path=Path("data/processed"), train=True)
#     val_set = MyDataset(data_path=Path("data/processed"), train=False)

#     train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
#     val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size)

#     # Initialize WandbLogger
#     wandb_logger = WandbLogger(
#         project="mlops_project",
#         config={
#             "learning_rate": learning_rate,
#             "batch_size": batch_size,
#             "epochs": epochs,
#         },
#     )

#     # Early stopping callback
#     early_stopping_callback = EarlyStopping(monitor="val_loss", patience=3, verbose=True, mode="min")

#     # Model checkpoint callback
#     checkpoint_callback = ModelCheckpoint(
#         monitor="val_loss",
#         dirpath="models/",
#         filename="best-checkpoint",
#         save_top_k=1,
#         mode="min",
#     )

#     # Print callback
#     print_callback = PrintCallback()

#     # Initialize Trainer
#     trainer = Trainer(
#         max_epochs=epochs,
#         accelerator="auto",
#         devices="auto",
#         logger=wandb_logger,
#         # callbacks=[early_stopping_callback, checkpoint_callback],
#         callbacks=[early_stopping_callback, checkpoint_callback, print_callback, TQDMProgressBar(refresh_rate=20)],
#         # precision="bf16-true",
#         # profiler="simple",
#         log_every_n_steps=1,
#         check_val_every_n_epoch=1,
#         enable_progress_bar=True,
#     )

#     # Train
#     trainer.fit(model, train_dataloader, val_dataloader)

#     print("Training complete")

#     # Save model locally
#     os.makedirs("models", exist_ok=True)
#     torch.save(model.state_dict(), "models/model.pth")

#     # Log artifact
#     artifact = wandb.Artifact(
#         name="corrupt_mnist_model",
#         type="model",
#         description="A model trained to classify corrupt MNIST images",
#         metadata={"learning_rate": learning_rate, "epochs": epochs},
#     )
#     artifact.add_file("models/model.pth")
#     wandb_logger.experiment.log_artifact(artifact)


# if __name__ == "__main__":
#     typer.run(train_model)
