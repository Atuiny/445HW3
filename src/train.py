from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib

from train_lib import TrainConfig, train_and_evaluate


REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = REPO_ROOT / "outputs"
REGISTRY_DIR = REPO_ROOT / "registry"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Teacher-style training + selection script. "
            "Trains models on sklearn breast cancer dataset, selects best by recall, "
            "writes outputs/, and updates registry/champion.json."
        )
    )

    p.add_argument("--metric", default="recall", help="Selection metric (default: recall)")
    p.add_argument("--force", action="store_true", help="Overwrite champion even if not improved")

    # Data split / seed
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)

    # LR hyperparams
    p.add_argument("--lr-c", type=float, default=1.0)
    p.add_argument("--lr-max-iter", type=int, default=500)
    p.add_argument("--lr-solver", type=str, default="liblinear")

    # RF hyperparams
    p.add_argument("--rf-n-estimators", type=int, default=300)
    p.add_argument("--rf-max-depth", type=int, default=None)
    p.add_argument("--rf-min-samples-split", type=int, default=2)

    return p.parse_args()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def _get_metric(metrics: Dict[str, Any], metric_name: str) -> float:
    v = metrics.get(metric_name)
    if v is None:
        raise KeyError(f"Missing required metric '{metric_name}' in metrics.json")
    if not isinstance(v, (int, float)):
        raise TypeError(f"Metric '{metric_name}' must be numeric")
    return float(v)


def _train(cfg: TrainConfig) -> Tuple[Any, Dict[str, Any]]:
    model, metrics = train_and_evaluate(cfg)
    metrics = dict(metrics)
    metrics["config"] = asdict(cfg)
    return model, metrics


def main() -> None:
    args = parse_args()

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)

    metric_name = str(args.metric)

    lr_cfg = TrainConfig(
        model_type="logistic_regression",
        test_size=float(args.test_size),
        random_state=int(args.random_state),
        lr_c=float(args.lr_c),
        lr_max_iter=int(args.lr_max_iter),
        lr_solver=str(args.lr_solver),
    )
    rf_cfg = TrainConfig(
        model_type="random_forest",
        test_size=float(args.test_size),
        random_state=int(args.random_state),
        rf_n_estimators=int(args.rf_n_estimators),
        rf_max_depth=args.rf_max_depth,
        rf_min_samples_split=int(args.rf_min_samples_split),
    )

    lr_model, lr_metrics = _train(lr_cfg)
    rf_model, rf_metrics = _train(rf_cfg)

    lr_score = _get_metric(lr_metrics, metric_name)
    rf_score = _get_metric(rf_metrics, metric_name)

    if rf_score > lr_score:
        best_model, best_metrics = rf_model, rf_metrics
        best_score = rf_score
    else:
        best_model, best_metrics = lr_model, lr_metrics
        best_score = lr_score

    outputs_model_path = OUTPUTS_DIR / "model.pkl"
    outputs_metrics_path = OUTPUTS_DIR / "metrics.json"
    joblib.dump(best_model, outputs_model_path)
    _write_json(outputs_metrics_path, best_metrics)

    # Instructor Dockerfile expects model.pkl at repo root
    root_model_path = REPO_ROOT / "model.pkl"
    joblib.dump(best_model, root_model_path)

    champion_path = REGISTRY_DIR / "champion.json"
    champion = _read_json(champion_path) or {}
    previous = champion.get("metric_value")
    previous_val = float(previous) if isinstance(previous, (int, float)) else None

    should_promote = bool(args.force) or previous_val is None or (best_score > previous_val)

    new_champion = {
        "updated_at": _utc_now_iso(),
        "metric_name": metric_name,
        "metric_value": best_score,
        "model_type": best_metrics.get("model_type"),
        "outputs": {
            "model": "outputs/model.pkl",
            "metrics": "outputs/metrics.json",
        },
        "config": best_metrics.get("config"),
        "previous_metric_value": previous_val,
        "promoted": should_promote,
    }

    if should_promote:
        _write_json(champion_path, new_champion)

    print("Training complete")
    print(f"- best {metric_name}: {best_score:.6f}")
    print(f"- wrote: {outputs_model_path}")
    print(f"- wrote: {outputs_metrics_path}")
    print(f"- wrote: {root_model_path}")
    if should_promote:
        print(f"- updated: {champion_path}")
    else:
        print(f"- kept existing champion: {champion_path}")


if __name__ == "__main__":
    main()
