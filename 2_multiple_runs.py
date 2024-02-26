import argparse
import logging
import os
import warnings
from typing import Tuple

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.entities.experiment import Experiment
from mlflow.tracking.fluent import ActiveRun
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

EXP_NAME = "Demo_with_ElasticNet"
RUN_TAGS: dict[str, str] = {
    "source": "create_experiment.py",
    "version": "v1",
    "priority": "p1",
    "type": "classification",
}
# get arguments from command
parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, required=False, default=0.2)
parser.add_argument("--l1_ratio", type=float, required=False, default=0.2)
args = parser.parse_args()


# evaluation function
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


# Function to create or get an existing MLflow experiment
def get_or_create_experiment(name) -> Tuple[Experiment, str]:
    exp: Experiment | None = mlflow.get_experiment_by_name(name)
    if exp:
        exp_id: str = exp.experiment_id
    else:
        exp_id: str = mlflow.create_experiment(
            name=name, tags={"exp_version": "v1", "priority": "p1"}
        )
    exp: Experiment = mlflow.get_experiment(experiment_id=exp_id)
    return exp, exp_id


warnings.filterwarnings("ignore")
np.random.seed(40)

# Read the wine-quality csv file from the URL
df = pd.read_csv("red-wine-quality.csv")
# make dir named "data" allow exists
os.makedirs("data", exist_ok=True)
df.to_csv("data/wine-quality.csv", index=False)

# Split the data into training and test sets. (0.75, 0.25) split.
train, test = train_test_split(df)

# The predicted column is "quality" which is a scalar from [3, 9]
train_x = train.drop(["quality"], axis=1)
test_x = test.drop(["quality"], axis=1)
train_y = train[["quality"]]
test_y = test[["quality"]]

print("The set tracking uri is ", mlflow.get_tracking_uri())

# Get or create an experiment
exp, exp_id = get_or_create_experiment(EXP_NAME)
print("Name: {}".format(exp.name))
print("Experiment_id: {}".format(exp.experiment_id))
print("Artifact Location: {}".format(exp.artifact_location))
print("Tags: {}".format(exp.tags))
print("Lifecycle_stage: {}".format(exp.lifecycle_stage))
print("Creation timestamp: {}".format(exp.creation_time))

for i in range(3):
    alpha = args.alpha * (i + 1)
    l1_ratio = args.l1_ratio * (i + 1)

    run: ActiveRun = mlflow.start_run(
        experiment_id=exp_id, run_name=f"based_alpha_{args.alpha}_run{i+1}", tags=RUN_TAGS
    )
    print("Run started")

    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    lr.fit(train_x, train_y)

    predicted_qualities = lr.predict(test_x)

    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

    print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)
    print(f"Active run id is {mlflow.active_run().info.run_id}")
    print(f"Active run name is {mlflow.active_run().info.run_name}")

    mlflow.set_tags({"new_tag": "new_value"})

    mlflow.log_param("alpha", alpha)
    mlflow.log_param("l1_ratio", l1_ratio)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)
    mlflow.sklearn.log_model(lr, "model")
    mlflow.log_artifacts(local_dir="data", artifact_path="logged_data")

    mlflow.end_run()
    print("Run ended")

    print(f"Last active run id is {mlflow.last_active_run().info.run_id}")
