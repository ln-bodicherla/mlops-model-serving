"""
A/B testing framework for model comparison.

Manages experiment creation, consistent traffic routing, outcome tracking,
and statistical significance testing for model comparison.
"""

import hashlib
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class ABExperiment:
    """Represents a single A/B testing experiment."""

    def __init__(
        self,
        experiment_id: str,
        champion_version: str,
        challenger_version: str,
        traffic_split: Dict[str, float],
        metric_name: str = "accuracy",
        min_sample_size: int = 1000,
        significance_level: float = 0.05,
    ):
        self.experiment_id = experiment_id
        self.champion_version = champion_version
        self.challenger_version = challenger_version
        self.traffic_split = traffic_split
        self.metric_name = metric_name
        self.min_sample_size = min_sample_size
        self.significance_level = significance_level

        self.created_at = time.time()
        self.status = "running"
        self.winner: Optional[str] = None

        self.champion_predictions: List[float] = []
        self.champion_actuals: List[float] = []
        self.challenger_predictions: List[float] = []
        self.challenger_actuals: List[float] = []

        self.champion_latencies: List[float] = []
        self.challenger_latencies: List[float] = []


class ABTestManager:
    """Manages A/B testing experiments for model comparison.

    Handles experiment creation, consistent user-based traffic routing,
    outcome logging, statistical significance testing, and automatic
    winner declaration.
    """

    def __init__(
        self,
        default_split: Optional[Dict[str, float]] = None,
        min_sample_size: int = 1000,
        significance_level: float = 0.05,
    ):
        self.default_split = default_split or {"champion": 0.9, "challenger": 0.1}
        self.min_sample_size = min_sample_size
        self.significance_level = significance_level
        self.experiments: Dict[str, ABExperiment] = {}

    def create_experiment(
        self,
        experiment_id: str,
        champion_version: str,
        challenger_version: str,
        traffic_split: Optional[Dict[str, float]] = None,
        metric_name: str = "accuracy",
    ) -> ABExperiment:
        """Create a new A/B test experiment.

        Args:
            experiment_id: Unique identifier for the experiment.
            champion_version: Current production model version.
            challenger_version: New candidate model version.
            traffic_split: Custom split (e.g., {"champion": 0.9, "challenger": 0.1}).
            metric_name: Primary metric for comparison.

        Returns:
            Created ABExperiment instance.
        """
        split = traffic_split or self.default_split

        if abs(sum(split.values()) - 1.0) > 1e-6:
            raise ValueError(f"Traffic split must sum to 1.0, got {sum(split.values())}")

        experiment = ABExperiment(
            experiment_id=experiment_id,
            champion_version=champion_version,
            challenger_version=challenger_version,
            traffic_split=split,
            metric_name=metric_name,
            min_sample_size=self.min_sample_size,
            significance_level=self.significance_level,
        )

        self.experiments[experiment_id] = experiment
        logger.info(
            f"Created A/B experiment '{experiment_id}': "
            f"{champion_version} vs {challenger_version}, split={split}"
        )
        return experiment

    def route_request(
        self, user_id: str, experiment_id: Optional[str] = None
    ) -> str:
        """Route a request to a model variant using consistent hashing.

        Uses MD5 hashing of the user ID to ensure the same user always
        sees the same variant within an experiment.

        Args:
            user_id: Unique user identifier for consistent routing.
            experiment_id: Which experiment to route for.

        Returns:
            Variant name string ("champion" or "challenger").
        """
        if experiment_id and experiment_id in self.experiments:
            experiment = self.experiments[experiment_id]
            if experiment.status != "running":
                if experiment.winner:
                    return experiment.winner
                return "champion"
            split = experiment.traffic_split
        else:
            split = self.default_split

        hash_input = f"{experiment_id or 'default'}:{user_id}"
        hash_bytes = hashlib.md5(hash_input.encode("utf-8")).digest()
        hash_value = int.from_bytes(hash_bytes[:8], byteorder="big")
        normalized = hash_value / (2**64)

        champion_threshold = split.get("champion", 0.9)
        if normalized < champion_threshold:
            return "champion"
        return "challenger"

    def log_outcome(
        self,
        experiment_id: str,
        variant: str,
        prediction: float,
        actual: Optional[float] = None,
        latency_ms: Optional[float] = None,
    ) -> None:
        """Track a prediction outcome for a specific variant.

        Args:
            experiment_id: Experiment to log against.
            variant: "champion" or "challenger".
            prediction: Model prediction value.
            actual: Ground truth label (if available).
            latency_ms: Prediction latency in milliseconds.
        """
        if experiment_id not in self.experiments:
            logger.warning(f"Unknown experiment: {experiment_id}")
            return

        experiment = self.experiments[experiment_id]

        if variant == "champion":
            experiment.champion_predictions.append(prediction)
            if actual is not None:
                experiment.champion_actuals.append(actual)
            if latency_ms is not None:
                experiment.champion_latencies.append(latency_ms)
        else:
            experiment.challenger_predictions.append(prediction)
            if actual is not None:
                experiment.challenger_actuals.append(actual)
            if latency_ms is not None:
                experiment.challenger_latencies.append(latency_ms)

    def compute_results(
        self, experiment_id: str
    ) -> Dict[str, Any]:
        """Compute statistical significance for the experiment.

        Performs chi-squared test for classification tasks and t-test
        for regression tasks to determine if the challenger significantly
        outperforms the champion.

        Args:
            experiment_id: Experiment to analyze.

        Returns:
            Dictionary with test results, p-values, and effect sizes.
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Unknown experiment: {experiment_id}")

        experiment = self.experiments[experiment_id]

        champion_n = len(experiment.champion_predictions)
        challenger_n = len(experiment.challenger_predictions)

        results = {
            "experiment_id": experiment_id,
            "champion_version": experiment.champion_version,
            "challenger_version": experiment.challenger_version,
            "champion_sample_size": champion_n,
            "challenger_sample_size": challenger_n,
            "sufficient_data": (
                champion_n >= experiment.min_sample_size
                and challenger_n >= experiment.min_sample_size
            ),
        }

        if experiment.champion_actuals and experiment.challenger_actuals:
            champ_preds = np.array(experiment.champion_predictions[:len(experiment.champion_actuals)])
            champ_actuals = np.array(experiment.champion_actuals)
            chall_preds = np.array(experiment.challenger_predictions[:len(experiment.challenger_actuals)])
            chall_actuals = np.array(experiment.challenger_actuals)

            is_classification = len(np.unique(champ_actuals)) <= 20

            if is_classification:
                champ_correct = (champ_preds == champ_actuals).astype(int)
                chall_correct = (chall_preds == chall_actuals).astype(int)

                champ_accuracy = float(np.mean(champ_correct))
                chall_accuracy = float(np.mean(chall_correct))

                contingency = np.array([
                    [np.sum(champ_correct), len(champ_correct) - np.sum(champ_correct)],
                    [np.sum(chall_correct), len(chall_correct) - np.sum(chall_correct)],
                ])

                try:
                    chi2, chi2_pvalue, _, _ = stats.chi2_contingency(contingency)
                except ValueError:
                    chi2, chi2_pvalue = 0.0, 1.0

                results.update({
                    "test_type": "chi_squared",
                    "champion_accuracy": round(champ_accuracy, 4),
                    "challenger_accuracy": round(chall_accuracy, 4),
                    "chi2_statistic": round(float(chi2), 4),
                    "p_value": round(float(chi2_pvalue), 6),
                    "is_significant": chi2_pvalue < experiment.significance_level,
                    "challenger_is_better": chall_accuracy > champ_accuracy,
                })
            else:
                champ_errors = np.abs(champ_preds - champ_actuals)
                chall_errors = np.abs(chall_preds - chall_actuals)

                champ_mae = float(np.mean(champ_errors))
                chall_mae = float(np.mean(chall_errors))

                t_stat, t_pvalue = stats.ttest_ind(champ_errors, chall_errors)

                pooled_std = np.sqrt(
                    (np.var(champ_errors) + np.var(chall_errors)) / 2
                )
                cohens_d = (champ_mae - chall_mae) / (pooled_std + 1e-10)

                results.update({
                    "test_type": "t_test",
                    "champion_mae": round(champ_mae, 4),
                    "challenger_mae": round(chall_mae, 4),
                    "t_statistic": round(float(t_stat), 4),
                    "p_value": round(float(t_pvalue), 6),
                    "cohens_d": round(float(cohens_d), 4),
                    "is_significant": t_pvalue < experiment.significance_level,
                    "challenger_is_better": chall_mae < champ_mae,
                })

        if experiment.champion_latencies and experiment.challenger_latencies:
            results["champion_avg_latency_ms"] = round(
                float(np.mean(experiment.champion_latencies)), 2
            )
            results["challenger_avg_latency_ms"] = round(
                float(np.mean(experiment.challenger_latencies)), 2
            )
            results["champion_p99_latency_ms"] = round(
                float(np.percentile(experiment.champion_latencies, 99)), 2
            )
            results["challenger_p99_latency_ms"] = round(
                float(np.percentile(experiment.challenger_latencies, 99)), 2
            )

        return results

    def declare_winner(
        self,
        experiment_id: str,
        auto_promote: bool = False,
    ) -> Dict[str, Any]:
        """Determine and declare the experiment winner.

        If the challenger is both statistically significantly better and
        has sufficient sample size, it is declared the winner.

        Args:
            experiment_id: Experiment to finalize.
            auto_promote: If True, mark experiment as completed.

        Returns:
            Dictionary with winner determination results.
        """
        results = self.compute_results(experiment_id)
        experiment = self.experiments[experiment_id]

        winner = "champion"
        reason = "default"

        if not results.get("sufficient_data", False):
            reason = "insufficient_data"
        elif results.get("is_significant", False) and results.get("challenger_is_better", False):
            winner = "challenger"
            reason = "statistically_significant_improvement"
        elif results.get("is_significant", False):
            reason = "champion_significantly_better"
        else:
            reason = "no_significant_difference"

        determination = {
            "experiment_id": experiment_id,
            "winner": winner,
            "reason": reason,
            "results": results,
        }

        if auto_promote:
            experiment.status = "completed"
            experiment.winner = winner
            logger.info(
                f"Experiment '{experiment_id}' completed: winner={winner} ({reason})"
            )

        return determination

    def get_experiment_status(
        self, experiment_id: str
    ) -> Dict[str, Any]:
        """Return current status of an A/B test experiment.

        Args:
            experiment_id: Experiment to query.

        Returns:
            Dictionary with experiment state, sample sizes, and any results.
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Unknown experiment: {experiment_id}")

        experiment = self.experiments[experiment_id]

        champion_n = len(experiment.champion_predictions)
        challenger_n = len(experiment.challenger_predictions)

        status = {
            "experiment_id": experiment_id,
            "status": experiment.status,
            "champion_version": experiment.champion_version,
            "challenger_version": experiment.challenger_version,
            "traffic_split": experiment.traffic_split,
            "champion_sample_size": champion_n,
            "challenger_sample_size": challenger_n,
            "total_requests": champion_n + challenger_n,
            "min_sample_size": experiment.min_sample_size,
            "samples_remaining": max(
                0,
                experiment.min_sample_size - min(champion_n, challenger_n),
            ),
            "created_at": experiment.created_at,
            "duration_hours": round((time.time() - experiment.created_at) / 3600, 2),
            "winner": experiment.winner,
        }

        if champion_n > 0 and challenger_n > 0:
            try:
                results = self.compute_results(experiment_id)
                status["current_results"] = {
                    "p_value": results.get("p_value"),
                    "is_significant": results.get("is_significant"),
                    "challenger_is_better": results.get("challenger_is_better"),
                }
            except Exception:
                pass

        return status

    def list_experiments(self) -> List[Dict[str, Any]]:
        """Return a summary of all experiments.

        Returns:
            List of experiment summary dictionaries.
        """
        summaries = []
        for exp_id, exp in self.experiments.items():
            summaries.append({
                "experiment_id": exp_id,
                "status": exp.status,
                "champion": exp.champion_version,
                "challenger": exp.challenger_version,
                "total_requests": (
                    len(exp.champion_predictions) + len(exp.challenger_predictions)
                ),
                "winner": exp.winner,
                "created_at": exp.created_at,
            })
        return summaries

    def stop_experiment(self, experiment_id: str) -> None:
        """Stop a running experiment without declaring a winner.

        Args:
            experiment_id: Experiment to stop.
        """
        if experiment_id in self.experiments:
            self.experiments[experiment_id].status = "stopped"
            logger.info(f"Experiment '{experiment_id}' stopped")
