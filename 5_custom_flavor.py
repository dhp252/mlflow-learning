import argparse
import logging
import os
import warnings
from typing import Tuple

import cloudpickle
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
from mlflow.entities.experiment import Experiment
from mlflow.models.signature import ModelSignature, infer_signature
from mlflow.tracking.fluent import ActiveRun
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")
np.random.seed(40)

EXP_NAME = os.path.basename(__file__).replace(".py", "")

RUN_TAGS: dict[str, str] = {
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
    r2 = r2_score(actual, pred)
    return rmse, r2


# Function to create or get an existing MLflow experiment
def setup_experiment(name, verbose=True) -> Tuple[Experiment, str]:
    exp: Experiment | None = mlflow.get_experiment_by_name(name)
    if exp:
        exp_id: str = exp.experiment_id
    else:
        exp_id: str = mlflow.create_experiment(
            name=name, tags={"exp_version": "v1", "priority": "p1"}
        )
    exp: Experiment = mlflow.get_experiment(experiment_id=exp_id)

    if verbose:
        print("Name: {}".format(exp.name))
        print("Experiment_id: {}".format(exp.experiment_id))
        print("Artifact Location: {}".format(exp.artifact_location))
        print("Tags: {}".format(exp.tags))
        print("Lifecycle_stage: {}".format(exp.lifecycle_stage))
        print("Creation timestamp: {}".format(exp.creation_time))

    return exp, exp_id


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
exp, exp_id = setup_experiment(EXP_NAME)

alpha = args.alpha
l1_ratio = args.l1_ratio

run: ActiveRun = mlflow.start_run(
    experiment_id=exp_id, run_name="autolog_sklearn", tags=RUN_TAGS
)

mlflow.sklearn.autolog(
    log_input_examples=False,
    log_model_signatures=False,
    log_models=False,
    log_post_training_metrics=True,
    max_tuning_runs=10,
)

print("Run started")

lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
lr.fit(train_x, train_y)

predicted_qualities = lr.predict(test_x)

signature: ModelSignature = infer_signature(
    model_input=train_x, model_output=predicted_qualities
)
# mlflow.sklearn.log_model(
#     sk_model=lr,
#     artifact_path="model",
#     signature=signature,
#     input_example=train_x.iloc[0:2],
#     code_paths=[os.path.basename(__file__)],
# )

# ! <start focus> ! #
custom_model_path = "sklearn_model.pkl"
joblib.dump(lr, custom_model_path)

class MyMCustomeModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.model = joblib.load(filename=context.artifacts["custom_model_path"])

    def predict(self, context, model_input):
        return self.model.predict(model_input)


# infer conda env
conda_env = mlflow.sklearn.get_default_conda_env()

mlflow.pyfunc.log_model(
    artifact_path="model",
    python_model=MyMCustomeModel(),
    artifacts={"custom_model_path": custom_model_path},
    code_path=[os.path.basename(__file__)],
    conda_env=conda_env,
)
# ! <end focus> ! #

predicted_qualities = lr.predict(test_x)

(rmse, r2) = eval_metrics(test_y, predicted_qualities)

print(f"Active run id is {mlflow.active_run().info.run_id}")
print(f"Active run name is {mlflow.active_run().info.run_name}")

mlflow.set_tags({"new_tag": "new_value"})

mlflow.log_artifacts(local_dir="data", artifact_path="logged_data")

mlflow.end_run()
print("Run ended")

print(f"Last active run id is {mlflow.last_active_run().info.run_id}")


####### PREDICTION ########

# model_uri = mlflow.get_run(loaded_model).info.artifact_uri + "/model"
model_uri = "runs:/d0886a5c81b14a0c9969298f28d9e9e4/model"
loaded_model = mlflow.pyfunc.load_model(model_uri)
print("Model URI: ", model_uri)
predicted_qualities_from_mlflow = loaded_model.predict(test_x)
