import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict

import joblib
import mlflow
import mlflow.sklearn
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.train_lib import DEFAULT_TRACKING_URI, load_breast_cancer_data


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Export the MLflow Model Registry champion alias to a standalone model.pkl file "
            "(for container inference without MLflow)."
        )
    )
    p.add_argument(
        "--model-name",
        default="breast-cancer-champion-model",
        help="Registered model name.",
    )
    p.add_argument(
        "--alias",
        default="champion",
        help="Alias to export (default: champion).",
    )
    p.add_argument(
        "--tracking-uri",
        default=DEFAULT_TRACKING_URI,
        help="MLflow tracking URI.",
    )
    p.add_argument(
        "--output",
        default=str(REPO_ROOT / "model_registry" / "champion" / "model.pkl"),
        help=(
            "Where to write the exported model pickle "
            "(default: ./model_registry/champion/model.pkl)."
        ),
    )
    p.add_argument(
        "--write-sample-request",
        action="store_true",
        help="If set, writes ./sample_request.json using real dataset values.",
    )
    return p.parse_args()


def _as_python_floats(x: np.ndarray) -> list[list[float]]:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    return x.tolist()


def main() -> None:
    args = parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)

    model_uri = f"models:/{args.model_name}@{args.alias}"
    model = mlflow.sklearn.load_model(model_uri)

    output_path = Path(args.output).resolve()
    joblib.dump(model, output_path)

    meta: Dict[str, Any] = {
        "exported_from": model_uri,
        "output": str(output_path),
    }

    meta_path = output_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(f"Loaded champion model URI: {model_uri}")
    print(f"Wrote model pickle       : {output_path}")
    print(f"Wrote metadata           : {meta_path}")

    if args.write_sample_request:
        # Use a deterministic split and take the first test row.
        _, X_test, _, _ = load_breast_cancer_data(test_size=0.3, split_random_state=42)
        sample = _as_python_floats(X_test[0])
        sample_payload = {"instances": sample}
        sample_path = REPO_ROOT / "sample_request.json"
        sample_path.write_text(json.dumps(sample_payload, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote sample request JSON: {sample_path}")


if __name__ == "__main__":
    main()
