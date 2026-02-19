"""
Data and prediction drift monitoring.

Implements statistical drift detection using PSI, KS-test, and
Evidently-based analysis to monitor model health in production.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class DriftMonitor:
    """Monitors data and prediction drift in production model serving.

    Maintains reference distributions from training data and compares
    incoming production data using multiple statistical tests. Supports
    both feature-level and prediction-level drift detection.
    """

    def __init__(
        self,
        psi_threshold: float = 0.2,
        ks_threshold: float = 0.05,
        reference_window: int = 10000,
        detection_window: int = 1000,
    ):
        self.psi_threshold = psi_threshold
        self.ks_threshold = ks_threshold
        self.reference_window = reference_window
        self.detection_window = detection_window

        self.reference_data: Optional[pd.DataFrame] = None
        self.reference_predictions: Optional[np.ndarray] = None
        self.feature_names: List[str] = []
        self.reference_stats: Dict[str, Dict] = {}

        self.drift_history: List[Dict[str, Any]] = []
        self.alert_callbacks: List = []

    def setup_reference_data(
        self, data: pd.DataFrame, predictions: Optional[np.ndarray] = None
    ) -> None:
        """Set the baseline distribution from training data.

        Computes and stores summary statistics for each feature to enable
        drift comparison against future production data.

        Args:
            data: Training/reference DataFrame with feature columns.
            predictions: Optional array of model predictions on reference data.
        """
        if len(data) > self.reference_window:
            self.reference_data = data.sample(n=self.reference_window, random_state=42)
        else:
            self.reference_data = data.copy()

        self.reference_predictions = predictions
        self.feature_names = list(data.select_dtypes(include=[np.number]).columns)

        for col in self.feature_names:
            values = self.reference_data[col].dropna().values
            self.reference_stats[col] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "median": float(np.median(values)),
                "q25": float(np.percentile(values, 25)),
                "q75": float(np.percentile(values, 75)),
                "num_unique": int(len(np.unique(values))),
                "histogram": np.histogram(values, bins=20),
            }

        logger.info(
            f"Reference data set: {len(self.reference_data)} samples, "
            f"{len(self.feature_names)} features"
        )

    def detect_data_drift(
        self, current_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Detect data drift across all features using multiple tests.

        Runs PSI and KS tests on each feature and aggregates results
        into an overall drift assessment.

        Args:
            current_data: Production data DataFrame.

        Returns:
            Dictionary with per-feature drift scores and overall drift flag.
        """
        if self.reference_data is None:
            raise ValueError("Reference data not set. Call setup_reference_data() first.")

        feature_results = {}
        drifted_features = []

        for col in self.feature_names:
            if col not in current_data.columns:
                continue

            ref_values = self.reference_data[col].dropna().values
            cur_values = current_data[col].dropna().values

            if len(cur_values) == 0:
                continue

            psi = self.compute_psi(ref_values, cur_values)
            ks_stat, ks_pvalue = self.compute_ks_statistic(ref_values, cur_values)

            ref_mean = np.mean(ref_values)
            cur_mean = np.mean(cur_values)
            mean_shift = abs(cur_mean - ref_mean) / (np.std(ref_values) + 1e-10)

            is_drifted = psi > self.psi_threshold or ks_pvalue < self.ks_threshold

            feature_results[col] = {
                "psi": round(float(psi), 6),
                "ks_statistic": round(float(ks_stat), 6),
                "ks_pvalue": round(float(ks_pvalue), 6),
                "mean_shift_std": round(float(mean_shift), 4),
                "reference_mean": round(float(ref_mean), 4),
                "current_mean": round(float(cur_mean), 4),
                "is_drifted": is_drifted,
            }

            if is_drifted:
                drifted_features.append(col)

        drift_fraction = len(drifted_features) / max(len(feature_results), 1)
        overall_drift = drift_fraction > 0.3 or len(drifted_features) > 0

        result = {
            "timestamp": time.time(),
            "num_samples": len(current_data),
            "features_analyzed": len(feature_results),
            "features_drifted": len(drifted_features),
            "drifted_feature_names": drifted_features,
            "drift_fraction": round(drift_fraction, 4),
            "overall_drift_detected": overall_drift,
            "feature_details": feature_results,
        }

        self.drift_history.append(result)
        return result

    def detect_prediction_drift(
        self,
        current_predictions: np.ndarray,
    ) -> Dict[str, Any]:
        """Monitor prediction distribution shifts.

        Compares the distribution of current predictions against the
        reference prediction distribution.

        Args:
            current_predictions: Array of recent model predictions.

        Returns:
            Dictionary with prediction drift metrics.
        """
        if self.reference_predictions is None:
            raise ValueError("Reference predictions not set.")

        psi = self.compute_psi(self.reference_predictions, current_predictions)
        ks_stat, ks_pvalue = self.compute_ks_statistic(
            self.reference_predictions, current_predictions
        )

        ref_mean = float(np.mean(self.reference_predictions))
        cur_mean = float(np.mean(current_predictions))
        ref_std = float(np.std(self.reference_predictions))
        cur_std = float(np.std(current_predictions))

        is_drifted = psi > self.psi_threshold or ks_pvalue < self.ks_threshold

        result = {
            "timestamp": time.time(),
            "num_predictions": len(current_predictions),
            "psi": round(float(psi), 6),
            "ks_statistic": round(float(ks_stat), 6),
            "ks_pvalue": round(float(ks_pvalue), 6),
            "reference_mean": round(ref_mean, 4),
            "current_mean": round(cur_mean, 4),
            "reference_std": round(ref_std, 4),
            "current_std": round(cur_std, 4),
            "is_drifted": is_drifted,
        }

        return result

    def compute_psi(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        num_bins: int = 20,
    ) -> float:
        """Calculate Population Stability Index.

        PSI measures the shift between two distributions. Values below 0.1
        indicate no significant change; 0.1-0.2 indicates moderate change;
        above 0.2 indicates significant change.

        Args:
            reference: Reference distribution values.
            current: Current distribution values.
            num_bins: Number of bins for histogram comparison.

        Returns:
            PSI value (float).
        """
        min_val = min(np.min(reference), np.min(current))
        max_val = max(np.max(reference), np.max(current))

        if min_val == max_val:
            return 0.0

        bin_edges = np.linspace(min_val, max_val, num_bins + 1)

        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        cur_counts, _ = np.histogram(current, bins=bin_edges)

        ref_pct = (ref_counts + 1) / (len(reference) + num_bins)
        cur_pct = (cur_counts + 1) / (len(current) + num_bins)

        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        return float(psi)

    def compute_ks_statistic(
        self, reference: np.ndarray, current: np.ndarray
    ) -> Tuple[float, float]:
        """Calculate the Kolmogorov-Smirnov statistic and p-value.

        The KS test measures the maximum distance between two empirical
        cumulative distribution functions.

        Args:
            reference: Reference distribution values.
            current: Current distribution values.

        Returns:
            Tuple of (KS statistic, p-value).
        """
        statistic, pvalue = stats.ks_2samp(reference, current)
        return float(statistic), float(pvalue)

    def generate_drift_report(
        self,
        drift_result: Dict[str, Any],
        output_path: Optional[str] = None,
    ) -> str:
        """Generate an HTML drift report.

        Args:
            drift_result: Output from detect_data_drift().
            output_path: Optional file path to save the HTML report.

        Returns:
            HTML report string.
        """
        html_parts = [
            "<!DOCTYPE html><html><head>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 40px; }",
            "table { border-collapse: collapse; width: 100%; margin: 20px 0; }",
            "th, td { border: 1px solid #ddd; padding: 10px; text-align: left; }",
            "th { background-color: #2c3e50; color: white; }",
            ".drifted { background-color: #fadbd8; }",
            ".normal { background-color: #d5f5e3; }",
            ".header { color: #2c3e50; }",
            ".summary { padding: 15px; border-radius: 5px; margin: 15px 0; }",
            ".alert { background-color: #fadbd8; border-left: 4px solid #e74c3c; }",
            ".ok { background-color: #d5f5e3; border-left: 4px solid #27ae60; }",
            "</style>",
            "</head><body>",
            "<h1 class='header'>Data Drift Report</h1>",
        ]

        is_drifted = drift_result.get("overall_drift_detected", False)
        status_class = "alert" if is_drifted else "ok"
        status_text = "DRIFT DETECTED" if is_drifted else "No Significant Drift"

        html_parts.append(
            f"<div class='summary {status_class}'>"
            f"<h2>{status_text}</h2>"
            f"<p>Samples analyzed: {drift_result.get('num_samples', 0)}</p>"
            f"<p>Features drifted: {drift_result.get('features_drifted', 0)} / "
            f"{drift_result.get('features_analyzed', 0)}</p>"
            f"</div>"
        )

        html_parts.append("<table>")
        html_parts.append(
            "<tr><th>Feature</th><th>PSI</th><th>KS Stat</th>"
            "<th>KS p-value</th><th>Mean Shift</th><th>Status</th></tr>"
        )

        for col, details in drift_result.get("feature_details", {}).items():
            row_class = "drifted" if details["is_drifted"] else "normal"
            status = "DRIFT" if details["is_drifted"] else "OK"
            html_parts.append(
                f"<tr class='{row_class}'>"
                f"<td>{col}</td>"
                f"<td>{details['psi']:.4f}</td>"
                f"<td>{details['ks_statistic']:.4f}</td>"
                f"<td>{details['ks_pvalue']:.4f}</td>"
                f"<td>{details['mean_shift_std']:.4f}</td>"
                f"<td>{status}</td></tr>"
            )

        html_parts.extend(["</table>", "</body></html>"])
        html_content = "\n".join(html_parts)

        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "w") as f:
                f.write(html_content)
            logger.info(f"Drift report saved to {output_path}")

        return html_content

    def check_and_alert(
        self,
        current_data: pd.DataFrame,
        current_predictions: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Run drift checks and trigger alerts if thresholds are exceeded.

        Args:
            current_data: Production feature data.
            current_predictions: Optional prediction values.

        Returns:
            Combined drift analysis with alert status.
        """
        data_drift = self.detect_data_drift(current_data)

        prediction_drift = None
        if current_predictions is not None and self.reference_predictions is not None:
            prediction_drift = self.detect_prediction_drift(current_predictions)

        alerts = []
        if data_drift["overall_drift_detected"]:
            alerts.append({
                "type": "data_drift",
                "severity": "high" if data_drift["drift_fraction"] > 0.5 else "medium",
                "message": (
                    f"Data drift detected in {data_drift['features_drifted']} / "
                    f"{data_drift['features_analyzed']} features"
                ),
                "drifted_features": data_drift["drifted_feature_names"],
            })

        if prediction_drift and prediction_drift["is_drifted"]:
            alerts.append({
                "type": "prediction_drift",
                "severity": "high",
                "message": (
                    f"Prediction distribution shift detected "
                    f"(PSI={prediction_drift['psi']:.4f})"
                ),
            })

        result = {
            "data_drift": data_drift,
            "prediction_drift": prediction_drift,
            "alerts": alerts,
            "has_alerts": len(alerts) > 0,
        }

        for callback in self.alert_callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

        return result

    def log_drift_metrics(
        self,
        drift_result: Dict[str, Any],
        tracker: Optional[Any] = None,
    ) -> None:
        """Send drift metrics to MLflow and/or Prometheus.

        Args:
            drift_result: Output from detect_data_drift() or check_and_alert().
            tracker: Optional ExperimentTracker instance for MLflow logging.
        """
        data_drift = drift_result.get("data_drift", drift_result)

        metrics = {
            "drift_fraction": data_drift.get("drift_fraction", 0),
            "features_drifted": data_drift.get("features_drifted", 0),
        }

        for col, details in data_drift.get("feature_details", {}).items():
            metrics[f"psi_{col}"] = details.get("psi", 0)
            metrics[f"ks_{col}"] = details.get("ks_statistic", 0)

        if tracker is not None:
            try:
                tracker.log_metrics(metrics)
            except Exception as e:
                logger.error(f"Failed to log drift metrics to MLflow: {e}")

        logger.info(
            f"Drift metrics: fraction={metrics.get('drift_fraction', 0):.4f}, "
            f"drifted_features={metrics.get('features_drifted', 0)}"
        )

    def register_alert_callback(self, callback) -> None:
        """Register a callable to be invoked when drift is detected.

        Args:
            callback: Callable that accepts a drift result dictionary.
        """
        self.alert_callbacks.append(callback)

    def get_drift_summary(self) -> Dict[str, Any]:
        """Return a summary of historical drift checks.

        Returns:
            Dictionary with drift check history statistics.
        """
        if not self.drift_history:
            return {"num_checks": 0}

        drift_fractions = [d.get("drift_fraction", 0) for d in self.drift_history]
        alerts = [d for d in self.drift_history if d.get("overall_drift_detected", False)]

        return {
            "num_checks": len(self.drift_history),
            "num_alerts": len(alerts),
            "avg_drift_fraction": round(float(np.mean(drift_fractions)), 4),
            "max_drift_fraction": round(float(np.max(drift_fractions)), 4),
            "last_check_timestamp": self.drift_history[-1].get("timestamp"),
        }
