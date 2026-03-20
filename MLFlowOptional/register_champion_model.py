import argparse
import time
from pathlib import Path
import sys

import mlflow
from mlflow.tracking import MlflowClient

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
DEFAULT_TRACKING_URI = f"sqlite:///{(REPO_ROOT / 'MLFlowOptional' / 'mlflow.db').as_posix()}"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Find best MLflow run by metric, register its model, and assign champion alias."
    )
    parser.add_argument(
        "--experiment-names",
        default="breast-cancer-model-comparison-lr,breast-cancer-model-comparison-rf",
        help="Comma-separated MLflow experiment names to search.",
    )
    parser.add_argument(
        "--ranking-metric",
        default="recall",
        help="Metric key used to rank runs. Default is recall.",
    )
    parser.add_argument(
        "--model-name",
        default="breast-cancer-champion-model",
        help="Registered model name where the new version will be created.",
    )
    parser.add_argument(
        "--alias",
        default="champion",
        help="Alias to set on the registered model version.",
    )
    parser.add_argument(
        "--description",
        default="Champion model selected automatically by highest recall from logged runs.",
        help="Short description to add to the registered model version.",
    )
    parser.add_argument(
        "--tracking-uri",
        default=DEFAULT_TRACKING_URI,
        help="MLflow tracking URI.",
    )
    return parser.parse_args()


def wait_until_ready(client: MlflowClient, model_name: str, model_version: str, timeout_seconds: int = 120):
    start = time.time()
    while True:
        version_info = client.get_model_version(name=model_name, version=model_version)
        status_value = str(version_info.status)
        if status_value == "READY" or status_value.endswith(".READY"):
            return version_info

        if time.time() - start > timeout_seconds:
            raise TimeoutError(
                f"Timed out waiting for model {model_name} version {model_version} to be READY. "
                f"Current status: {version_info.status}"
            )

        time.sleep(1)


def _get_experiment_ids(client: MlflowClient, experiment_names: list[str]) -> list[str]:
    ids: list[str] = []
    for name in experiment_names:
        exp = mlflow.get_experiment_by_name(name)
        if exp is None:
            raise ValueError(f"Experiment '{name}' was not found.")
        ids.append(exp.experiment_id)
    return ids


def find_best_run_id(client: MlflowClient, experiment_ids: list[str], ranking_metric: str) -> tuple[str, float]:
    runs = client.search_runs(
        experiment_ids=experiment_ids,
        order_by=[f"metrics.{ranking_metric} DESC"],
        max_results=1,
    )
    if not runs:
        raise ValueError(
            f"No runs with metric '{ranking_metric}' were found in experiments {experiment_ids}."
        )
    best = runs[0]
    metric_val = best.data.metrics.get(ranking_metric)
    if metric_val is None:
        raise ValueError(f"Best run is missing metric '{ranking_metric}'.")
    return best.info.run_id, float(metric_val)


def main():
    args = parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()

    experiment_names = [n.strip() for n in args.experiment_names.split(",") if n.strip()]
    experiment_ids = _get_experiment_ids(client, experiment_names)

    run_id, metric_value = find_best_run_id(client, experiment_ids, args.ranking_metric)
    run = mlflow.get_run(run_id)
    run_name = run.data.tags.get("mlflow.runName", "")
    model_family = run.data.tags.get("model_family", "")

    artifact_path = run.data.tags.get("model_artifact_path", "model")

    model_uri = f"runs:/{run_id}/{artifact_path}"
    registered = mlflow.register_model(model_uri=model_uri, name=args.model_name)

    ready_version = wait_until_ready(client, args.model_name, registered.version)
    client.update_model_version(
        name=args.model_name,
        version=ready_version.version,
        description=args.description,
    )
    client.set_model_version_tag(
        name=args.model_name,
        version=ready_version.version,
        key="role",
        value="champion",
    )
    client.set_registered_model_alias(
        name=args.model_name,
        alias=args.alias,
        version=ready_version.version,
    )

    print("=" * 70)
    print(f"Experiments       : {', '.join(experiment_names)}")
    print(f"Best metric       : {args.ranking_metric}={metric_value:.6f}")
    print(f"Best run id       : {run_id}")
    print(f"Best run name     : {run_name}")
    print(f"Model family      : {model_family}")
    print(f"Model artifact    : {artifact_path}")
    print(f"Source model URI  : {model_uri}")
    print(f"Registered model  : {args.model_name}")
    print(f"New version       : {ready_version.version}")
    print("Version tag       : role=champion")
    print(f"Alias set         : {args.alias}")
    print(f"Load URI          : models:/{args.model_name}@{args.alias}")
    print("=" * 70)


if __name__ == "__main__":
    main()
