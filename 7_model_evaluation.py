import argparse
import logging
import os
import warnings
from typing import Tuple

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mlflow.entities.experiment import Experiment
from mlflow.models import MetricThreshold, make_metric
from mlflow.tracking.fluent import ActiveRun
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")
np.random.seed(40)

EXP_NAME = os.path.basename(__file__).replace(".py", "")
MODEL_ARTIFACT_PATH = "model"
BASELINE_MODEL_ARTIFACT_PATH = "baseline_model"

RUN_TAGS: dict[str, str] = {
    "type": "classification",
}
# get arguments from command
parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, required=False, default=0.7)
parser.add_argument("--l1_ratio", type=float, required=False, default=0.7)
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
    experiment_id=exp_id, run_name="model_evaluation_with_baseline", tags=RUN_TAGS
)

mlflow.sklearn.autolog(
    log_input_examples=True,
    log_model_signatures=True,
    log_models=False,
    log_post_training_metrics=True,
    max_tuning_runs=10,
)

print("Run started")

lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
lr.fit(train_x, train_y)

predicted_qualities = lr.predict(test_x)

mlflow.sklearn.log_model(
    sk_model=lr,
    artifact_path="model",
    input_example=train_x.iloc[0:2],
    code_paths=[os.path.basename(__file__)],
)

# sklearn_model_path = "sklearn_model.pkl"
# joblib.dump(lr, sklearn_model_path)
# artifact = {"sklearn_model": sklearn_model_path, "data": data_dir}

# Create baseline model
baseline_model = DummyRegressor()
baseline_model.fit(train_x, train_y)

baseline_predicted_qualities = baseline_model.predict(test_x)

rmse, r2 = eval_metrics(test_y, predicted_qualities)

mlflow.log_metrics({"rmse": rmse, "r2": r2})

baseline_model_path = "baseline_model.pkl"
joblib.dump(baseline_model, baseline_model_path)
baseline_artifacts = {"baseline_model": baseline_model_path}


class SklearnModel(mlflow.pyfunc.PythonModel):

    def __init__(self, artifact_name):
        self.artifact_name = artifact_name

    def load_context(self, context):
        self.model = joblib.load(filename=context.artifacts[self.artifact_name])

    def predict(self, context, model_input):
        return self.model.predict(model_input)


mlflow.pyfunc.log_model(
    artifact_path=BASELINE_MODEL_ARTIFACT_PATH,
    python_model=SklearnModel(BASELINE_MODEL_ARTIFACT_PATH),
    artifacts=baseline_artifacts,
    # code_paths=[os.path.basename(__file__)],
)


# ! <start focus> ! #
# create custom metrics
def my_random_calculation_1(eval_df, builtin_metrics):
    return eval_df["prediction"] - eval_df["target"] + 100


def my_random_calculation_2(eval_df, builtin_metrics):
    return builtin_metrics["r2_score"] + 100


my_metric_1 = make_metric(
    eval_fn=my_random_calculation_1, greater_is_better=False, name="diff_adds_100"
)

my_metric_2 = make_metric(
    eval_fn=my_random_calculation_2, greater_is_better=True, name="r2_adds_100"
)


# create custom artifacts
def prediction_target_scatter(
    eval_df, _builtin_metrics, artifacts_dir
) -> dict[str, str]:
    plt.scatter(eval_df["prediction"], eval_df["target"])
    plt.xlabel("Targets")
    plt.ylabel("Predictions")
    plt.title("Targets vs Predictions")
    plot_path = os.path.join(artifacts_dir, "scatter_plot.png")
    plt.savefig(plot_path)
    return {"scatter_plot": plot_path}


# define thresholds
thresholds = {
    "mean_squared_error": MetricThreshold(
        threshold=0.1,  # maximum allowed MSE
        min_absolute_change=0.1,  # minimum absolute improvement compared to the baseline
        min_relative_change=0.05,  # minimum relative improvement compared to the baseline
        greater_is_better=False,  # lower MSE is better
    ),
}

# artifact_uri = run.info.artifact_uri + "/model"
artifact_uri = mlflow.get_artifact_uri(MODEL_ARTIFACT_PATH)
baseline_artifact_uri = mlflow.get_artifact_uri(BASELINE_MODEL_ARTIFACT_PATH)
mlflow.evaluate(
    model=artifact_uri,
    data=test,
    targets="quality",
    model_type="regressor",
    evaluators=["default"],
    custom_metrics=[  # optional
        my_metric_1,
        my_metric_2,
    ],
    custom_artifacts=[prediction_target_scatter],  # optional
    # validation_thresholds=thresholds,  # optional
    baseline_model=baseline_artifact_uri,  # optional
)
# ! <end focus> ! #

mlflow.end_run()
