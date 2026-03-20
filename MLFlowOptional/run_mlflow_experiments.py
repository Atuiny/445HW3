import argparse
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Optional

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.train_lib import DEFAULT_TRACKING_URI, TrainConfig, train_and_evaluate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run multiple MLflow experiments/runs on sklearn breast-cancer dataset, "
            "log recall + artifacts, and optionally register the best model as champion."
        )
    )
    p.add_argument(
        "--tracking-uri",
        default=DEFAULT_TRACKING_URI,
        help="MLflow tracking URI (default uses local sqlite mlflow.db).",
    )
    p.add_argument(
        "--experiment-name",
        default="breast-cancer-model-comparison",
        help=(
            "Best-practice mode: logs runs into a single, stable experiment name. "
            "Re-running the script with different hyperparameters creates additional runs in the same experiment."
        ),
    )
    p.add_argument(
        "--experiment-names",
        nargs=2,
        default=None,
        metavar=("LR_EXPERIMENT", "RF_EXPERIMENT"),
        help=(
            "Optional compatibility mode: use two separate experiments (LR + RF), "
            "logging one run into each. If omitted, a single experiment is used."
        ),
    )
    p.add_argument(
        "--run-lr",
        action="store_true",
        help="If set, logs one Logistic Regression run in the LR experiment.",
    )
    p.add_argument(
        "--run-rf",
        action="store_true",
        help="If set, logs one Random Forest run in the RF experiment.",
    )

    # Hyperparameter overrides (one run per experiment)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)

    p.add_argument("--lr-c", type=float, default=1.0)
    p.add_argument("--lr-max-iter", type=int, default=500)
    p.add_argument("--lr-solver", type=str, default="liblinear")

    p.add_argument("--rf-n-estimators", type=int, default=300)
    p.add_argument(
        "--rf-max-depth",
        type=int,
        default=None,
        help="Max depth for RandomForest (omit for None).",
    )
    p.add_argument("--rf-min-samples-split", type=int, default=2)
    p.add_argument(
        "--ranking-metric",
        default="recall",
        help="Metric used to pick the best run (default: recall).",
    )
    p.add_argument(
        "--register",
        action="store_true",
        help="If set, registers the best model and assigns champion alias.",
    )
    p.add_argument(
        "--model-name",
        default="breast-cancer-champion-model",
        help="Registered model name for champion registration.",
    )
    p.add_argument(
        "--alias",
        default="champion",
        help="Alias to assign to the best registered model version.",
    )
    p.add_argument(
        "--description",
        default="Champion model selected automatically by highest recall from logged runs.",
        help="Description added to the registered model version.",
    )
    return p.parse_args()


def _ensure_experiment(name: str) -> str:
    exp = mlflow.get_experiment_by_name(name)
    if exp is not None:
        return exp.experiment_id
    return mlflow.create_experiment(name)


def _log_numeric_metrics(metrics: Dict[str, Any]) -> None:
    numeric: Dict[str, float] = {}
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            numeric[k] = float(v)
    if numeric:
        mlflow.log_metrics(numeric)


def _run_configs(experiment_name: str, configs: Iterable[TrainConfig]) -> None:
    mlflow.set_experiment(experiment_name)

    for cfg in configs:
        artifact_path = _model_artifact_path(cfg)
        run_name = artifact_path
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tag("dataset", cfg.dataset)
            mlflow.set_tag("model_family", cfg.model_type)
            mlflow.set_tag("model_artifact_path", artifact_path)
            mlflow.log_params(asdict(cfg))

            model, metrics = train_and_evaluate(cfg)

            _log_numeric_metrics(metrics)
            mlflow.log_dict(metrics, "metrics.json")
            # Log the model under a distinctive artifact path so it's easy to identify in the UI.
            mlflow.sklearn.log_model(model, artifact_path=artifact_path)


def _search_best_run(
    client: MlflowClient,
    experiment_ids: List[str],
    ranking_metric: str,
) -> tuple[str, float, str]:
    runs = client.search_runs(
        experiment_ids=experiment_ids,
        order_by=[f"metrics.{ranking_metric} DESC"],
        max_results=1,
    )
    if not runs:
        raise ValueError(
            f"No runs found in experiments {experiment_ids}. "
            f"Did you run this script without errors?"
        )
    best = runs[0]
    val = best.data.metrics.get(ranking_metric)
    if val is None:
        raise ValueError(f"Best run is missing metric '{ranking_metric}'.")
    artifact_path = best.data.tags.get("model_artifact_path", "model")
    return best.info.run_id, float(val), artifact_path


def _safe_token(value: Any) -> str:
    s = str(value)
    s = s.replace(" ", "")
    s = s.replace("/", "-")
    s = s.replace("\\", "-")
    s = s.replace(":", "-")
    s = s.replace("=", "-")
    s = s.replace(".", "p")
    return s


def _model_artifact_path(cfg: TrainConfig) -> str:
    if cfg.model_type == "logistic_regression":
        return "__".join(
            [
                "model",
                "lr",
                f"C-{_safe_token(cfg.lr_c)}",
                f"solver-{_safe_token(cfg.lr_solver)}",
                f"maxiter-{_safe_token(cfg.lr_max_iter)}",
                f"seed-{_safe_token(cfg.random_state)}",
            ]
        )

    if cfg.model_type == "random_forest":
        depth = "None" if cfg.rf_max_depth is None else cfg.rf_max_depth
        return "__".join(
            [
                "model",
                "rf",
                f"trees-{_safe_token(cfg.rf_n_estimators)}",
                f"depth-{_safe_token(depth)}",
                f"minsplit-{_safe_token(cfg.rf_min_samples_split)}",
                f"seed-{_safe_token(cfg.random_state)}",
            ]
        )

    # Fallback (should not happen)
    return f"model__{_safe_token(cfg.model_type)}"


def _wait_until_ready(client: MlflowClient, model_name: str, version: str, timeout_seconds: int = 120):
    import time

    start = time.time()
    while True:
        info = client.get_model_version(name=model_name, version=version)
        status_value = str(info.status)
        if status_value == "READY" or status_value.endswith(".READY"):
            return info
        if time.time() - start > timeout_seconds:
            raise TimeoutError(
                f"Timed out waiting for model {model_name} version {version} to be READY. "
                f"Current status: {info.status}"
            )
        time.sleep(1)


def _register_champion(
    client: MlflowClient,
    run_id: str,
    artifact_path: str,
    model_name: str,
    alias: str,
    description: str,
) -> None:
    model_uri = f"runs:/{run_id}/{artifact_path}"
    registered = mlflow.register_model(model_uri=model_uri, name=model_name)

    ready = _wait_until_ready(client, model_name, registered.version)
    client.update_model_version(name=model_name, version=ready.version, description=description)
    client.set_model_version_tag(name=model_name, version=ready.version, key="role", value="champion")
    client.set_registered_model_alias(name=model_name, alias=alias, version=ready.version)

    print("=" * 70)
    print(f"Registered model  : {model_name}")
    print(f"New version       : {ready.version}")
    print(f"Alias set         : {alias}")
    print(f"Load URI          : models:/{model_name}@{alias}")
    print("=" * 70)


def main() -> None:
    args = parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()

    separate_experiments = args.experiment_names is not None
    if separate_experiments:
        exp_lr, exp_rf = args.experiment_names
        exp_lr_id = _ensure_experiment(exp_lr)
        exp_rf_id = _ensure_experiment(exp_rf)
        experiment_ids = [exp_lr_id, exp_rf_id]
    else:
        exp_name = str(args.experiment_name)
        exp_id = _ensure_experiment(exp_name)
        experiment_ids = [exp_id]

    # One run per experiment. If neither flag is specified, run both.
    run_lr = bool(args.run_lr)
    run_rf = bool(args.run_rf)
    if not run_lr and not run_rf:
        run_lr = True
        run_rf = True

    lr_cfg: Optional[TrainConfig] = None
    rf_cfg: Optional[TrainConfig] = None

    if run_lr:
        lr_cfg = TrainConfig(
            model_type="logistic_regression",
            test_size=float(args.test_size),
            random_state=int(args.random_state),
            lr_c=float(args.lr_c),
            lr_max_iter=int(args.lr_max_iter),
            lr_solver=str(args.lr_solver),
        )

    if run_rf:
        rf_cfg = TrainConfig(
            model_type="random_forest",
            test_size=float(args.test_size),
            random_state=int(args.random_state),
            rf_n_estimators=int(args.rf_n_estimators),
            rf_max_depth=args.rf_max_depth,
            rf_min_samples_split=int(args.rf_min_samples_split),
        )

    if separate_experiments:
        if lr_cfg is not None:
            _run_configs(exp_lr, [lr_cfg])
        if rf_cfg is not None:
            _run_configs(exp_rf, [rf_cfg])
    else:
        configs: List[TrainConfig] = []
        if lr_cfg is not None:
            configs.append(lr_cfg)
        if rf_cfg is not None:
            configs.append(rf_cfg)
        _run_configs(exp_name, configs)

    best_run_id, best_metric, best_artifact_path = _search_best_run(
        client, experiment_ids, args.ranking_metric
    )

    print("=" * 70)
    print(f"Best run_id       : {best_run_id}")
    print(f"Best metric       : {args.ranking_metric}={best_metric:.6f}")
    print(f"Model artifact    : {best_artifact_path}")
    print(f"Model URI         : runs:/{best_run_id}/{best_artifact_path}")
    print("=" * 70)

    # Programmatically load the best model (requirement)
    best_model = mlflow.sklearn.load_model(f"runs:/{best_run_id}/{best_artifact_path}")
    print(f"Loaded best model type: {type(best_model)}")

    if args.register:
        _register_champion(
            client=client,
            run_id=best_run_id,
            artifact_path=best_artifact_path,
            model_name=args.model_name,
            alias=args.alias,
            description=args.description,
        )


if __name__ == "__main__":
    main()
