import os
import pickle
from collections.abc import Generator
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import BackgroundTasks, FastAPI


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and classes, and create database file."""
    global model, classes
    classes = ["Iris-Setosa", "Iris-Versicolour", "Iris-Virginica"]
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)

    # check if file exists, if not create it with header
    if not os.path.exists("prediction_database.csv"):
        with open("prediction_database.csv", "w") as file:
            file.write("time,sepal_length,sepal_width,petal_length,petal_width,target\n")
    yield

    del model


app = FastAPI(lifespan=lifespan)


def add_to_database(
    now: str,
    sepal_length: float,
    sepal_width: float,
    petal_length: float,
    petal_width: float,
    prediction: int,
) -> None:
    """Simple function to add prediction to database."""
    with open("prediction_database.csv", "a") as file:
        file.write(f"{now},{sepal_length},{sepal_width},{petal_length},{petal_width},{prediction}\n")


@app.post("/predict")
async def iris_inference(
    sepal_length: float,
    sepal_width: float,
    petal_length: float,
    petal_width: float,
    background_tasks: BackgroundTasks,
):
    """Version 2 of the iris inference endpoint."""
    prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = prediction.item()

    now = str(datetime.now(tz=timezone.utc))
    background_tasks.add_task(add_to_database, now, sepal_length, sepal_width, petal_length, petal_width, prediction)
    return {"prediction": classes[prediction], "prediction_int": prediction}
