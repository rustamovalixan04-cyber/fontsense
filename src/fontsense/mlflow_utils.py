from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path


@contextmanager
def optional_mlflow_run(
    experiment_name: str,
    run_name: str,
    tracking_dir: str | Path = "mlruns",
):
    """Use MLflow when installed; continue cleanly when it is unavailable."""
    try:
        import mlflow

        resolved_tracking_dir = Path(tracking_dir).resolve()
        resolved_tracking_dir.mkdir(parents=True, exist_ok=True)
        database_path = (resolved_tracking_dir / "mlflow.db").as_posix()
        mlflow.set_tracking_uri(f"sqlite:///{database_path}")
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=run_name):
            yield mlflow
    except ImportError:
        yield None
