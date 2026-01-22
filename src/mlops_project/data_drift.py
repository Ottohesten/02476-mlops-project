import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset
from sklearn import datasets

reference_data = datasets.load_iris(as_frame=True).frame
reference_data = reference_data.rename(
    columns={
        "sepal length (cm)": "sepal_length",
        "sepal width (cm)": "sepal_width",
        "petal length (cm)": "petal_length",
        "petal width (cm)": "petal_width",
        "target": "target",
    }
)


current_data = pd.read_csv("prediction_database.csv")
current_data = current_data.drop(columns=["time", "timestamp"], errors="ignore")
current_data = current_data.rename(columns={"prediction": "target"})

print("Reference columns:", reference_data.columns)
print("Current columns:", current_data.columns)

# Ensure types in current_data match reference_data
for col in reference_data.columns:
    if col in current_data.columns:
        current_data[col] = current_data[col].astype(reference_data[col].dtype)

if len(current_data) == 0:
    print("Warning: prediction_database.csv is empty. Cannot generate data drift report.")
    exit(0)

# make the data into evidently datasets
data_definition = DataDefinition(
    numerical_columns=["sepal_length", "sepal_width", "petal_length", "petal_width"],
    categorical_columns=["target"],
)

ref_ds = Dataset.from_pandas(reference_data, data_definition=data_definition)
curr_ds = Dataset.from_pandas(current_data, data_definition=data_definition)

report = Report(metrics=[DataDriftPreset()])
my_eval = report.run(
    reference_data=ref_ds,
    current_data=curr_ds,
)
my_eval.save_html("report.html")
print("Data drift report saved to report.html")
