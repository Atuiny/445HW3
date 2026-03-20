"""Training utilities for the sklearn breast cancer dataset.

This file intentionally contains reusable helpers (TrainConfig, train_and_evaluate, etc.)
so that different entrypoints (CI workflow, local scripts, etc.) can share the same logic.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import joblib
import numpy as np
import yaml
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRACKING_URI = f"sqlite:///{(REPO_ROOT / 'MLFlowOptional' / 'mlflow.db').as_posix()}"


ModelType = Literal["logistic_regression", "random_forest"]


@dataclass(frozen=True)
class TrainConfig:
    dataset: str = "breast_cancer"
    model_type: ModelType = "logistic_regression"
    test_size: float = 0.2
    random_state: int = 42

    # Logistic Regression
    lr_c: float = 1.0
    lr_max_iter: int = 500
    lr_solver: str = "liblinear"

    # Random Forest
    rf_n_estimators: int = 300
    rf_max_depth: Optional[int] = None
    rf_min_samples_split: int = 2


def _read_yaml_params(params_path: str) -> Dict[str, Any]:
    if not params_path or not os.path.exists(params_path):
        return {}
    with open(params_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in params file: {params_path}")
    return data


def _coerce_config(params: Dict[str, Any]) -> TrainConfig:
    train_params = params.get("train", {}) if isinstance(params, dict) else {}
    if train_params is None:
        train_params = {}
    if not isinstance(train_params, dict):
        raise ValueError("params.yaml: 'train' must be a mapping")

    allowed = {field.name for field in TrainConfig.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    unknown = set(train_params.keys()) - allowed
    if unknown:
        raise ValueError(f"Unknown train params: {sorted(unknown)}")

    return TrainConfig(**train_params)


def _load_dataset(dataset: str) -> tuple[np.ndarray, np.ndarray]:
    dataset = dataset.strip().lower()
    if dataset != "breast_cancer":
        raise ValueError(
            "Unsupported dataset. Use 'breast_cancer' (sklearn.datasets.load_breast_cancer)."
        )

    bunch = load_breast_cancer()
    X = bunch.data.astype(np.float32, copy=False)
    y = bunch.target.astype(np.int64, copy=False)
    return X, y


def load_breast_cancer_data(
    test_size: float = 0.2,
    split_random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convenience loader for the breast cancer dataset with a fixed split.

    Returns: (X_train, X_test, y_train, y_test)
    """

    X, y = _load_dataset("breast_cancer")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(test_size),
        random_state=int(split_random_state),
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


def _build_model(config: TrainConfig):
    if config.model_type == "logistic_regression":
        clf = LogisticRegression(
            C=float(config.lr_c),
            max_iter=int(config.lr_max_iter),
            solver=str(config.lr_solver),
            random_state=int(config.random_state),
        )
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    if config.model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=int(config.rf_n_estimators),
            max_depth=config.rf_max_depth,
            min_samples_split=int(config.rf_min_samples_split),
            random_state=int(config.random_state),
            n_jobs=-1,
        )

    raise ValueError(f"Unsupported model_type: {config.model_type}")


def _predict_scores(model, X: np.ndarray) -> Optional[np.ndarray]:
    """Return probability scores for ROC-AUC when available."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if isinstance(proba, np.ndarray) and proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        if isinstance(scores, np.ndarray):
            return scores
    return None


def train_and_evaluate(config: TrainConfig) -> tuple[Any, Dict[str, Any]]:
    X, y = _load_dataset(config.dataset)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(config.test_size),
        random_state=int(config.random_state),
        stratify=y,
    )

    model = _build_model(config)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics: Dict[str, Any] = {
        "dataset": config.dataset,
        "model_type": config.model_type,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_features": int(X_train.shape[1]),
        "test_size": float(config.test_size),
        "random_state": int(config.random_state),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    scores = _predict_scores(model, X_test)
    if scores is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_test, scores))

    return model, metrics


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    _ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a sklearn binary classifier.")
    p.add_argument(
        "--params",
        default="params.yaml",
        help="Path to params.yaml (expects a top-level 'train' mapping).",
    )
    p.add_argument(
        "--output-dir",
        default=os.path.join("data", "processed"),
        help="Directory to write model + metrics artifacts.",
    )
    p.add_argument(
        "--dataset",
        default=None,
        help="Override dataset name (default comes from params.yaml).",
    )
    p.add_argument(
        "--model-type",
        choices=["logistic_regression", "random_forest"],
        default=None,
        help="Override model type (default comes from params.yaml).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    params = _read_yaml_params(args.params)
    config = _coerce_config(params)

    if args.dataset is not None:
        config = TrainConfig(**{**config.__dict__, "dataset": args.dataset})
    if args.model_type is not None:
        config = TrainConfig(**{**config.__dict__, "model_type": args.model_type})

    model, metrics = train_and_evaluate(config)

    output_dir = args.output_dir
    _ensure_dir(output_dir)

    model_path = os.path.join(output_dir, "model.joblib")
    metrics_path = os.path.join(output_dir, "metrics.json")

    joblib.dump(model, model_path)
    _write_json(metrics_path, metrics)

    print(f"Wrote model: {model_path}")
    print(f"Wrote metrics: {metrics_path}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()
