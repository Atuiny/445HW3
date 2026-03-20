import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Tuple

import joblib

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.train_lib import TrainConfig, train_and_evaluate

REGISTRY_DIR = REPO_ROOT / "model_registry"
CHAMPION_DIR = REGISTRY_DIR / "champion"
CANDIDATES_DIR = REGISTRY_DIR / "candidates"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Train a candidate model and (optionally) promote it to the local dummy registry. "
            "This mimics an MLflow-style champion registry using files."
        )
    )
    p.add_argument(
        "--model-type",
        choices=["logistic_regression", "random_forest"],
        default="logistic_regression",
        help="Which model family to train.",
    )

    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)

    # LR
    p.add_argument("--lr-c", type=float, default=1.0)
    p.add_argument("--lr-max-iter", type=int, default=500)
    p.add_argument("--lr-solver", type=str, default="liblinear")

    # RF
    p.add_argument("--rf-n-estimators", type=int, default=300)
    p.add_argument("--rf-max-depth", type=int, default=None)
    p.add_argument("--rf-min-samples-split", type=int, default=2)

    p.add_argument(
        "--metric",
        default="recall",
        help="Metric used for promotion comparison (default: recall).",
    )
    p.add_argument(
        "--promote",
        action="store_true",
        help="If set, promotes candidate to champion when it beats the current champion.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="If set, promotes candidate even if it is not better.",
    )
    return p.parse_args()


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def _get_best_metric(metrics: Dict[str, Any], metric_name: str) -> float:
    v = metrics.get(metric_name)
    if v is None:
        raise KeyError(f"Metrics missing required key '{metric_name}'")
    if not isinstance(v, (int, float)):
        raise TypeError(f"Metric '{metric_name}' must be numeric, got {type(v)}")
    return float(v)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _candidate_slug(cfg: TrainConfig) -> str:
    # Short, file-system-safe identifier.
    parts = [cfg.model_type, f"seed{cfg.random_state}"]
    if cfg.model_type == "logistic_regression":
        parts.append(f"C{cfg.lr_c}")
        parts.append(f"solver{cfg.lr_solver}")
        parts.append(f"maxiter{cfg.lr_max_iter}")
    else:
        parts.append(f"trees{cfg.rf_n_estimators}")
        parts.append(f"depth{cfg.rf_max_depth}")
        parts.append(f"minsplit{cfg.rf_min_samples_split}")

    slug = "__".join(parts)
    slug = slug.replace(" ", "")
    slug = slug.replace("/", "-")
    slug = slug.replace("\\", "-")
    slug = slug.replace(":", "-")
    return slug


def _build_config(args: argparse.Namespace) -> TrainConfig:
    if args.model_type == "logistic_regression":
        return TrainConfig(
            model_type=args.model_type,
            test_size=float(args.test_size),
            random_state=int(args.random_state),
            lr_c=float(args.lr_c),
            lr_max_iter=int(args.lr_max_iter),
            lr_solver=str(args.lr_solver),
        )

    return TrainConfig(
        model_type=args.model_type,
        test_size=float(args.test_size),
        random_state=int(args.random_state),
        rf_n_estimators=int(args.rf_n_estimators),
        rf_max_depth=args.rf_max_depth,
        rf_min_samples_split=int(args.rf_min_samples_split),
    )


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())


def _write_candidate_artifacts(
    cfg: TrainConfig,
    model: Any,
    metrics: Dict[str, Any],
    promotion_metric: str,
) -> Tuple[Path, Path, Path]:
    slug = _candidate_slug(cfg)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = CANDIDATES_DIR / f"{ts}__{slug}"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "model.pkl"
    metrics_path = out_dir / "metrics.json"
    meta_path = out_dir / "metadata.json"

    joblib.dump(model, model_path)
    _write_json(metrics_path, metrics)

    metric_value = _get_best_metric(metrics, promotion_metric)

    meta = {
        "created_at": _utc_now_iso(),
        "config": asdict(cfg),
        "metric_for_promotion": promotion_metric,
        "metric_value": metric_value,
    }
    # Optionally include git SHA if available.
    git_head = REPO_ROOT / ".git" / "HEAD"
    if git_head.exists():
        meta["git_present"] = True
    _write_json(meta_path, meta)

    return model_path, metrics_path, meta_path


def main() -> None:
    args = parse_args()

    CHAMPION_DIR.mkdir(parents=True, exist_ok=True)
    CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)

    cfg = _build_config(args)

    model, metrics = train_and_evaluate(cfg)
    candidate_metric = _get_best_metric(metrics, args.metric)

    # Always write a candidate record for traceability.
    cand_model_path, cand_metrics_path, cand_meta_path = _write_candidate_artifacts(
        cfg, model, metrics, promotion_metric=args.metric
    )

    print("Candidate trained")
    print(f"- candidate_dir : {cand_model_path.parent}")
    print(f"- {args.metric}      : {candidate_metric:.6f}")

    champion_metrics_path = CHAMPION_DIR / "metrics.json"
    champion_model_path = CHAMPION_DIR / "model.pkl"
    champion_meta_path = CHAMPION_DIR / "metadata.json"

    champion_metrics = _read_json(champion_metrics_path) or {}
    champion_metric = None
    if champion_metrics:
        try:
            champion_metric = _get_best_metric(champion_metrics, args.metric)
        except Exception:
            champion_metric = None

    if not args.promote:
        print("Promotion skipped (use --promote to enable).")
        return

    should_promote = args.force or champion_metric is None or (candidate_metric > champion_metric)

    if not should_promote:
        print("Not promoting: candidate did not beat current champion.")
        print(f"- champion {args.metric}: {champion_metric:.6f}")
        print(f"- candidate {args.metric}: {candidate_metric:.6f}")
        return

    _copy_file(cand_model_path, champion_model_path)
    _copy_file(cand_metrics_path, champion_metrics_path)

    # Compatibility with instructor demo Dockerfile (expects ./model.pkl at repo root)
    root_model_path = REPO_ROOT / "model.pkl"
    _copy_file(cand_model_path, root_model_path)

    meta = {
        "promoted_at": _utc_now_iso(),
        "source_candidate_dir": str(cand_model_path.parent),
        "config": asdict(cfg),
        "metric_for_promotion": args.metric,
        "candidate_metric": candidate_metric,
        "previous_champion_metric": champion_metric,
    }
    _write_json(champion_meta_path, meta)

    print("Promoted to champion")
    print(f"- champion_model  : {champion_model_path}")
    print(f"- champion_metrics: {champion_metrics_path}")
    print(f"- root_model_copy : {root_model_path}")


if __name__ == "__main__":
    main()
