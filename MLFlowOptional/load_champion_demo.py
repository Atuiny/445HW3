import argparse
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.train_lib import DEFAULT_TRACKING_URI, load_breast_cancer_data


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load champion model by alias, predict on sample data, and print metadata checks."
    )
    parser.add_argument(
        "--model-name",
        default="breast-cancer-champion-model",
        help="Registered model name.",
    )
    parser.add_argument("--alias", default="champion", help="Registered model alias to load.")
    parser.add_argument("--batch-size", type=int, default=5, help="Number of test rows to predict.")
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI, help="MLflow tracking URI.")
    return parser.parse_args()


def main():
    args = parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()

    model_uri = f"models:/{args.model_name}@{args.alias}"
    champion_model = mlflow.sklearn.load_model(model_uri)

    _, X_test, _, y_test = load_breast_cancer_data(test_size=0.3, split_random_state=42)
    X_batch = X_test[: args.batch_size]
    y_batch = y_test[: args.batch_size]

    predictions = champion_model.predict(X_batch)

    champion_version = client.get_model_version_by_alias(args.model_name, args.alias)
    champion_tags = champion_version.tags or {}
    champion_run = mlflow.get_run(champion_version.run_id)

    print("=" * 70)
    print(f"Loaded URI                : {model_uri}")
    print(f"Champion model            : {args.model_name}")
    print(f"Champion alias            : {args.alias}")
    print(f"Champion version          : {champion_version.version}")
    print(f"Champion run_id           : {champion_version.run_id}")
    print(f"Champion source           : {champion_version.source}")
    print(f"Champion version tags     : {champion_tags}")
    print("-" * 70)
    print("Sample predictions")
    for idx, (pred, actual) in enumerate(zip(predictions.tolist(), y_batch.tolist()), start=1):
        print(f"row {idx}: pred={pred} actual={actual}")
    print("-" * 70)
    print("Original run metadata check")
    print(f"run_id (from run)         : {champion_run.info.run_id}")
    print(f"run_name                  : {champion_run.data.tags.get('mlflow.runName', '')}")
    print(f"model_family              : {champion_run.data.tags.get('model_family', '')}")
    print(f"recall metric             : {champion_run.data.metrics.get('recall', 'N/A')}")
    print(f"matches champion run_id   : {champion_run.info.run_id == champion_version.run_id}")
    print(f"has role=champion tag     : {champion_tags.get('role') == 'champion'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
