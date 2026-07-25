from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path


@contextmanager
def optional_mlflow_run(experiment_name: str, run_name: str):
    """Use MLflow when installed; continue cleanly when it is unavailable."""
    try:
        import mlflow

        tracking_dir = Path("mlruns").resolve()
        tracking_dir.mkdir(parents=True, exist_ok=True)
        database_path = (tracking_dir / "mlflow.db").as_posix()
        mlflow.set_tracking_uri(f"sqlite:///{database_path}")
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=run_name):
            yield mlflow
    except ImportError:
        yield None
