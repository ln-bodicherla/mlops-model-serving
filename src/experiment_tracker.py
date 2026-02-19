"""
MLflow experiment tracking and model registry management.

Provides a unified interface for logging training runs, registering
models, comparing experiments, and managing model lifecycle stages.
"""

import logging
import os
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Union

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Manages MLflow experiment tracking and model registry operations.

    Provides methods for logging training runs, registering models in the
    MLflow Model Registry, promoting models through stages, and comparing
    experiments to identify the best performing configurations.
    """

    def __init__(
        self,
        tracking_uri: str = "http://localhost:5000",
        registry_name: str = "production_model",
    ):
        self.tracking_uri = tracking_uri
        self.registry_name = registry_name
        self.client: Optional[MlflowClient] = None
        self.experiment_id: Optional[str] = None
        self.active_run = None

    def setup_mlflow(
        self,
        experiment_name: str = "model_serving",
        artifact_location: Optional[str] = None,
    ) -> str:
        """Configure MLflow tracking URI and experiment.

        Args:
            experiment_name: Name of the MLflow experiment.
            artifact_location: Optional S3/GCS/local path for artifacts.

        Returns:
            Experiment ID string.
        """
        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = MlflowClient(tracking_uri=self.tracking_uri)

        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            self.experiment_id = mlflow.create_experiment(
                experiment_name,
                artifact_location=artifact_location,
            )
        else:
            self.experiment_id = experiment.experiment_id

        mlflow.set_experiment(experiment_name)
        logger.info(
            f"MLflow configured: tracking_uri={self.tracking_uri}, "
            f"experiment={experiment_name} (id={self.experiment_id})"
        )
        return self.experiment_id

    @contextmanager
    def log_training_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        nested: bool = False,
    ):
        """Context manager for logging a complete training run.

        Args:
            run_name: Optional display name for the run.
            tags: Optional key-value tags for the run.
            nested: Whether this is a nested child run.

        Yields:
            Active MLflow Run object.
        """
        with mlflow.start_run(run_name=run_name, nested=nested) as run:
            if tags:
                for key, value in tags.items():
                    mlflow.set_tag(key, value)
            self.active_run = run
            logger.info(f"Started run: {run.info.run_id} ({run_name})")
            try:
                yield run
            except Exception as e:
                mlflow.set_tag("run_status", "failed")
                mlflow.log_param("error", str(e)[:250])
                logger.error(f"Run failed: {e}")
                raise
            finally:
                self.active_run = None

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log multiple parameters to the active run.

        Args:
            params: Dictionary of parameter names and values.
        """
        for key, value in params.items():
            mlflow.log_param(key, value)

    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        """Log multiple metrics to the active run.

        Args:
            metrics: Dictionary of metric names and values.
            step: Optional step number for metric history.
        """
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)

    def log_model(
        self,
        model: Any,
        artifact_path: str = "model",
        registered_name: Optional[str] = None,
        input_example: Optional[Any] = None,
    ) -> str:
        """Log a trained model as an artifact and optionally register it.

        Args:
            model: Trained sklearn model object.
            artifact_path: Artifact path within the run.
            registered_name: If provided, also registers in the Model Registry.
            input_example: Optional input example for schema inference.

        Returns:
            Model URI string.
        """
        model_info = mlflow.sklearn.log_model(
            model,
            artifact_path=artifact_path,
            registered_model_name=registered_name or self.registry_name,
            input_example=input_example,
        )

        model_uri = model_info.model_uri
        logger.info(f"Model logged: {model_uri}")
        return model_uri

    def promote_model(
        self,
        model_name: Optional[str] = None,
        version: Optional[int] = None,
        stage: str = "Production",
        archive_existing: bool = True,
    ) -> None:
        """Transition a model version to a new stage.

        Args:
            model_name: Registered model name. Defaults to registry_name.
            version: Specific version number. If None, uses latest.
            stage: Target stage ("Staging", "Production", "Archived").
            archive_existing: Whether to archive the current production model.
        """
        if self.client is None:
            self.client = MlflowClient(tracking_uri=self.tracking_uri)

        name = model_name or self.registry_name

        if version is None:
            versions = self.client.get_latest_versions(name, stages=["None", "Staging"])
            if not versions:
                raise ValueError(f"No model versions found for {name}")
            version = max(v.version for v in versions)

        if archive_existing:
            try:
                prod_versions = self.client.get_latest_versions(name, stages=["Production"])
                for pv in prod_versions:
                    self.client.transition_model_version_stage(
                        name=name,
                        version=pv.version,
                        stage="Archived",
                    )
                    logger.info(f"Archived previous production model v{pv.version}")
            except Exception:
                pass

        self.client.transition_model_version_stage(
            name=name,
            version=str(version),
            stage=stage,
        )
        logger.info(f"Model {name} v{version} promoted to {stage}")

    def compare_runs(
        self,
        metric_name: str = "f1_score",
        max_results: int = 20,
        order_by: Optional[str] = None,
    ) -> pd.DataFrame:
        """Compare metrics across experiment runs.

        Args:
            metric_name: Primary metric to compare.
            max_results: Maximum number of runs to return.
            order_by: MLflow order_by clause.

        Returns:
            DataFrame with run info and metrics.
        """
        if self.client is None:
            self.client = MlflowClient(tracking_uri=self.tracking_uri)

        order = order_by or f"metrics.{metric_name} DESC"
        runs = self.client.search_runs(
            experiment_ids=[self.experiment_id],
            order_by=[order],
            max_results=max_results,
            run_view_type=ViewType.ACTIVE_ONLY,
        )

        records = []
        for run in runs:
            record = {
                "run_id": run.info.run_id,
                "run_name": run.info.run_name,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "duration_s": (
                    (run.info.end_time - run.info.start_time) / 1000
                    if run.info.end_time else None
                ),
            }
            record.update({f"param_{k}": v for k, v in run.data.params.items()})
            record.update({f"metric_{k}": v for k, v in run.data.metrics.items()})
            records.append(record)

        df = pd.DataFrame(records)
        logger.info(f"Retrieved {len(df)} runs for comparison")
        return df

    def get_best_run(
        self,
        metric_name: str = "f1_score",
        higher_is_better: bool = True,
    ) -> Dict[str, Any]:
        """Find the best run by a specific metric.

        Args:
            metric_name: Metric to optimize.
            higher_is_better: Whether higher metric values are better.

        Returns:
            Dictionary with best run info, params, and metrics.
        """
        if self.client is None:
            self.client = MlflowClient(tracking_uri=self.tracking_uri)

        direction = "DESC" if higher_is_better else "ASC"
        runs = self.client.search_runs(
            experiment_ids=[self.experiment_id],
            order_by=[f"metrics.{metric_name} {direction}"],
            max_results=1,
            run_view_type=ViewType.ACTIVE_ONLY,
        )

        if not runs:
            raise ValueError("No completed runs found in experiment")

        best = runs[0]
        return {
            "run_id": best.info.run_id,
            "run_name": best.info.run_name,
            "params": dict(best.data.params),
            "metrics": dict(best.data.metrics),
            "artifact_uri": best.info.artifact_uri,
        }

    def load_production_model(
        self, model_name: Optional[str] = None
    ) -> Any:
        """Load the current production-stage model from the registry.

        Args:
            model_name: Registered model name. Defaults to registry_name.

        Returns:
            Loaded model object.
        """
        name = model_name or self.registry_name
        model_uri = f"models:/{name}/Production"

        model = mlflow.sklearn.load_model(model_uri)
        logger.info(f"Loaded production model: {name}")
        return model

    def load_model_version(
        self, version: int, model_name: Optional[str] = None
    ) -> Any:
        """Load a specific model version from the registry.

        Args:
            version: Model version number.
            model_name: Registered model name.

        Returns:
            Loaded model object.
        """
        name = model_name or self.registry_name
        model_uri = f"models:/{name}/{version}"

        model = mlflow.sklearn.load_model(model_uri)
        logger.info(f"Loaded model {name} v{version}")
        return model

    def get_model_history(
        self, model_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve version history with metrics for a registered model.

        Args:
            model_name: Registered model name.

        Returns:
            List of version info dictionaries with associated run metrics.
        """
        if self.client is None:
            self.client = MlflowClient(tracking_uri=self.tracking_uri)

        name = model_name or self.registry_name

        try:
            versions = self.client.search_model_versions(f"name='{name}'")
        except Exception:
            return []

        history = []
        for version in versions:
            entry = {
                "version": version.version,
                "stage": version.current_stage,
                "status": version.status,
                "creation_timestamp": version.creation_timestamp,
                "description": version.description,
                "run_id": version.run_id,
                "metrics": {},
            }

            if version.run_id:
                try:
                    run = self.client.get_run(version.run_id)
                    entry["metrics"] = dict(run.data.metrics)
                    entry["params"] = dict(run.data.params)
                except Exception:
                    pass

            history.append(entry)

        history.sort(key=lambda x: int(x["version"]), reverse=True)
        return history

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Log a local file or directory as an artifact.

        Args:
            local_path: Path to the local file or directory.
            artifact_path: Optional subdirectory in the artifact store.
        """
        mlflow.log_artifact(local_path, artifact_path)
        logger.info(f"Artifact logged: {local_path}")

    def delete_run(self, run_id: str) -> None:
        """Soft-delete a run.

        Args:
            run_id: MLflow run ID to delete.
        """
        if self.client is None:
            self.client = MlflowClient(tracking_uri=self.tracking_uri)
        self.client.delete_run(run_id)
        logger.info(f"Deleted run: {run_id}")
