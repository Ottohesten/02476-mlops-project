import os
from pathlib import Path

# import matplotlib.pyplot as plt
import torch
import typer
from sklearn.metrics import RocCurveDisplay, accuracy_score, f1_score, precision_score, recall_score

import wandb

# from mlops_project.model import ModelSequential as MyAwesomeModel
from mlops_project.data import MyDataset, corrupt_mnist
from mlops_project.model import MyAwesomeModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
# DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")


app = typer.Typer()

sweep_configuration = {
    "method": "random",
    "name": "sweep",
    "metric": {"goal": "minimize", "name": "loss"},
    "parameters": {
        "batch_size": {"values": [32, 64, 128]},
        "epochs": {"values": [10, 15, 20]},
    },
}


def train_model(lr: float = 1e-3, batch_size: int = 64, epochs: int = 10) -> None:
    """Train a model on MNIST."""
    print("Training day and night")
    print(f"{lr=}, {batch_size=}, {epochs=}")

    with wandb.init(
        project="mlops_project",
        config={
            "learning_rate": lr,
            "batch_size": batch_size,
            "epochs": epochs,
        },
    ) as run:
        config = run.config
        lr = config.learning_rate
        batch_size = config.batch_size
        epochs = config.epochs

        model = MyAwesomeModel().to(DEVICE)
        # train_set, _ = corrupt_mnist()
        train_set = MyDataset(data_path=Path("data/processed"), train=True)

        train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        statistics = {"train_loss": [], "train_accuracy": []}
        for epoch in range(epochs):
            model.train()
            preds = []
            targets = []
            for i, (img, target) in enumerate(train_dataloader):
                img, target = img.to(DEVICE), target.to(DEVICE)
                optimizer.zero_grad()
                y_pred = model(img)
                loss = loss_fn(y_pred, target)
                loss.backward()
                optimizer.step()
                statistics["train_loss"].append(loss.item())

                accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
                statistics["train_accuracy"].append(accuracy)
                wandb.log({"train_loss": loss.item(), "train_accuracy": accuracy})

                preds.append(y_pred.detach().cpu())
                targets.append(target.detach().cpu())

                if i % 100 == 0:
                    print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")

                    # log an img to wandb
                    # wandb.log(
                    #     {
                    #         "examples": [
                    #             wandb.Image(
                    #                 img[j], caption=f"Pred: {y_pred.argmax(dim=1)[j].item()}, True: {target[j].item()}"
                    #             )
                    #             for j in range(min(5, img.shape[0]))
                    #         ]
                    #     }
                    # )
            # add a custom matplotlib plot of the ROC curves
            preds = torch.cat(preds, 0)
            targets = torch.cat(targets, 0)

            # Softmax to get probabilities for ROC
            probs = torch.softmax(preds, dim=1)

            # for class_id in range(10):
            #     one_hot = (targets == class_id).float()
            #     _ = RocCurveDisplay.from_predictions(
            #         one_hot,
            #         probs[:, class_id],
            #         name=f"ROC curve for {class_id}",
            #         plot_chance_level=(class_id == 2),
            #     )

            # alternatively use wandb.log({"roc": wandb.Image(plt)}
            # wandb.log({"roc": wandb.Image(plt)})
            # plt.close()  # close the plot to avoid memory leaks and overlapping figures

        print("Training complete")
        # os.makedirs("models", exist_ok=True)
        # torch.save(model.state_dict(), "models/model.pth")
        # fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        # axs[0].plot(statistics["train_loss"])
        # axs[0].set_title("Train loss")
        # axs[1].plot(statistics["train_accuracy"])
        # axs[1].set_title("Train accuracy")
        # os.makedirs("reports/figures", exist_ok=True)
        # fig.savefig("reports/figures/training_statistics.png")

        final_accuracy = accuracy_score(targets, preds.argmax(dim=1))
        final_precision = precision_score(targets, preds.argmax(dim=1), average="weighted")
        final_recall = recall_score(targets, preds.argmax(dim=1), average="weighted")
        final_f1 = f1_score(targets, preds.argmax(dim=1), average="weighted")

        artifact = wandb.Artifact(
            name="corrupt_mnist_model",
            type="model",
            description="A model trained to classify corrupt MNIST images",
            metadata={"accuracy": final_accuracy, "precision": final_precision, "recall": final_recall, "f1": final_f1},
        )
        artifact.add_file("models/model.pth")
        run.log_artifact(artifact)


@app.command()
def train(lr: float = 1e-3, batch_size: int = 64, epochs: int = 10) -> None:
    train_model(lr, batch_size, epochs)


@app.command()
def sweep(count: int = 5):
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="mlops_project")
    wandb.agent(sweep_id, function=train_model, count=count)


if __name__ == "__main__":
    app()
