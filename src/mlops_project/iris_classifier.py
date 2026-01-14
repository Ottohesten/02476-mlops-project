import os
import pickle
from typing import Annotated, Literal

import typer
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

app = typer.Typer()
train_app = typer.Typer()
app.add_typer(train_app, name="train")


# Load the dataset
x, y = load_breast_cancer(return_X_y=True)

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


@train_app.command()
def svm(
    output_path: Annotated[str, typer.Option("--output", "-o")] = "model.ckpt",
    kernel: Literal["linear", "poly", "rbf", "sigmoid", "precomputed"] = "linear",
):
    """Train and evaluate the model."""

    # Train a Support Vector Machine (SVM) model
    model = SVC(kernel=kernel, random_state=42)
    model.fit(x_train, y_train)

    if os.path.dirname(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {output_path}")


@train_app.command()
def knn(n_neighbors: int = 5, output_path: Annotated[str, typer.Option("--output", "-o")] = "model.ckpt"):
    """
    Train a K-Nearest Neighbors (KNN) model.
    """
    if os.path.dirname(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(x_train, y_train)
    with open(output_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {output_path}")


@app.command()
def evaluate(model_path):
    """Evaluate the model on the test set."""

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)


# this "if"-block is added to enable the script to be run from the command line
if __name__ == "__main__":
    app()
