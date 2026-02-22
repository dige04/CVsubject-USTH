"""Evaluation framework for hand gesture classifiers.

Provides standardized evaluation metrics including:
    - Leave-One-Person-Out cross-validation
    - Stratified train/test split evaluation
    - Confusion matrix computation and plotting
    - Classification reports (precision, recall, F1)
    - Inference latency benchmarking
    - Bootstrap confidence intervals
    - Comparison visualizations
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable, Protocol

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold, StratifiedKFold

GESTURE_CLASSES = ["open_hand", "fist", "pinch", "frame", "none"]


class Classifier(Protocol):
    """Protocol for classifiers compatible with the evaluation framework."""

    def predict(self, X: np.ndarray) -> np.ndarray | list[str]:
        ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        ...


class Evaluator:
    """Unified evaluation framework for gesture classifiers.

    Supports multiple evaluation strategies and generates comparison
    reports across different classification methods.
    """

    def __init__(self, class_names: list[str] | None = None) -> None:
        """Initialize the evaluator.

        Args:
            class_names: List of class label names. Defaults to GESTURE_CLASSES.
        """
        self.class_names = class_names or GESTURE_CLASSES

    def lopo_cv(
        self,
        train_and_predict_fn: Callable[
            [np.ndarray, np.ndarray, np.ndarray], np.ndarray
        ],
        X: np.ndarray,
        y: np.ndarray,
        person_ids: np.ndarray,
    ) -> dict[str, Any]:
        """Leave-One-Person-Out Cross-Validation.

        For each unique person, trains on all other persons and tests
        on the held-out person.

        Args:
            train_and_predict_fn: Callable that takes
                (X_train, y_train, X_test) and returns y_pred.
                This allows each method to handle its own feature
                extraction and training.
            X: Full feature array of shape (n_samples, n_features).
            y: String labels of shape (n_samples,).
            person_ids: Person identifiers of shape (n_samples,).

        Returns:
            Dictionary with:
                - per_fold_accuracy: list of accuracies per fold
                - mean_accuracy: mean across folds
                - std_accuracy: std across folds
                - all_y_true: concatenated true labels
                - all_y_pred: concatenated predictions
                - per_class_metrics: dict with precision/recall/F1 per class
        """
        unique_persons = np.unique(person_ids)
        fold_accuracies: list[float] = []
        all_y_true: list[np.ndarray] = []
        all_y_pred: list[np.ndarray] = []

        for person in unique_persons:
            test_mask = person_ids == person
            train_mask = ~test_mask

            X_train, y_train = X[train_mask], y[train_mask]
            X_test, y_test = X[test_mask], y[test_mask]

            if len(np.unique(y_train)) < 2:
                continue  # Skip folds with insufficient class diversity

            y_pred = train_and_predict_fn(X_train, y_train, X_test)
            y_pred = np.asarray(y_pred)

            fold_acc = accuracy_score(y_test, y_pred)
            fold_accuracies.append(fold_acc)
            all_y_true.append(y_test)
            all_y_pred.append(y_pred)

        all_y_true_arr = np.concatenate(all_y_true)
        all_y_pred_arr = np.concatenate(all_y_pred)

        report = classification_report(
            all_y_true_arr,
            all_y_pred_arr,
            labels=self.class_names,
            output_dict=True,
            zero_division=0,
        )

        return {
            "per_fold_accuracy": fold_accuracies,
            "mean_accuracy": float(np.mean(fold_accuracies)),
            "std_accuracy": float(np.std(fold_accuracies)),
            "all_y_true": all_y_true_arr,
            "all_y_pred": all_y_pred_arr,
            "per_class_metrics": report,
            "n_folds": len(fold_accuracies),
        }

    def group_kfold_cv(
        self,
        train_and_predict_fn: Callable[
            [np.ndarray, np.ndarray, np.ndarray], np.ndarray
        ],
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        n_splits: int = 5,
    ) -> dict[str, Any]:
        """Group K-Fold Cross-Validation (person-aware).

        Uses sklearn GroupKFold to split by person groups, ensuring no
        person appears in both train and test. Suitable when there are
        too many unique persons for full LOPO-CV.

        Args:
            train_and_predict_fn: Callable (X_train, y_train, X_test) -> y_pred.
            X: Feature array of shape (n_samples, n_features).
            y: String labels of shape (n_samples,).
            groups: Group identifiers (person IDs) of shape (n_samples,).
            n_splits: Number of folds.

        Returns:
            Same format as lopo_cv results.
        """
        gkf = GroupKFold(n_splits=n_splits)
        fold_accuracies: list[float] = []
        all_y_true: list[np.ndarray] = []
        all_y_pred: list[np.ndarray] = []

        for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            y_pred = np.asarray(train_and_predict_fn(X_train, y_train, X_test))

            fold_acc = accuracy_score(y_test, y_pred)
            fold_accuracies.append(fold_acc)
            all_y_true.append(y_test)
            all_y_pred.append(y_pred)

        all_y_true_arr = np.concatenate(all_y_true)
        all_y_pred_arr = np.concatenate(all_y_pred)

        report = classification_report(
            all_y_true_arr,
            all_y_pred_arr,
            labels=self.class_names,
            output_dict=True,
            zero_division=0,
        )

        return {
            "per_fold_accuracy": fold_accuracies,
            "mean_accuracy": float(np.mean(fold_accuracies)),
            "std_accuracy": float(np.std(fold_accuracies)),
            "all_y_true": all_y_true_arr,
            "all_y_pred": all_y_pred_arr,
            "per_class_metrics": report,
            "n_folds": len(fold_accuracies),
        }

    def stratified_split(
        self,
        train_and_predict_fn: Callable[
            [np.ndarray, np.ndarray, np.ndarray], np.ndarray
        ],
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> dict[str, Any]:
        """Evaluate with a stratified train/test split.

        Args:
            train_and_predict_fn: Callable (X_train, y_train, X_test) -> y_pred.
            X: Feature array.
            y: Labels.
            test_size: Fraction for test set.
            random_state: Random seed.

        Returns:
            Dictionary with accuracy, y_true, y_pred, classification report.
        """
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        y_pred = np.asarray(train_and_predict_fn(X_train, y_train, X_test))

        report = classification_report(
            y_test,
            y_pred,
            labels=self.class_names,
            output_dict=True,
            zero_division=0,
        )

        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "y_true": y_test,
            "y_pred": y_pred,
            "per_class_metrics": report,
        }

    def compute_confusion_matrix(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> np.ndarray:
        """Compute confusion matrix.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.

        Returns:
            Confusion matrix array of shape (n_classes, n_classes).
        """
        return confusion_matrix(y_true, y_pred, labels=self.class_names)

    def compute_classification_report(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> dict[str, Any]:
        """Compute per-class precision, recall, F1.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.

        Returns:
            Dictionary with per-class and aggregate metrics.
        """
        return classification_report(
            y_true,
            y_pred,
            labels=self.class_names,
            output_dict=True,
            zero_division=0,
        )

    def inference_latency(
        self,
        classifier: Classifier,
        X_single: np.ndarray,
        n_iterations: int = 1000,
    ) -> float:
        """Measure average inference latency for a single sample.

        Args:
            classifier: Trained classifier with predict() method.
            X_single: Single sample array of shape (1, n_features).

        Returns:
            Average inference time in milliseconds.
        """
        if X_single.ndim == 1:
            X_single = X_single.reshape(1, -1)

        # Warm up
        for _ in range(10):
            classifier.predict(X_single)

        # Measure
        start = time.perf_counter()
        for _ in range(n_iterations):
            classifier.predict(X_single)
        elapsed = time.perf_counter() - start

        return (elapsed / n_iterations) * 1000  # Convert to ms

    def bootstrap_ci(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_bootstrap: int = 1000,
        ci: float = 0.95,
        metric_fn: Callable[[np.ndarray, np.ndarray], float] | None = None,
        random_state: int = 42,
    ) -> tuple[float, float, float]:
        """Compute bootstrap confidence interval on a metric.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            n_bootstrap: Number of bootstrap samples.
            ci: Confidence level (e.g., 0.95 for 95% CI).
            metric_fn: Metric function (default: accuracy_score).
            random_state: Random seed.

        Returns:
            Tuple of (point_estimate, ci_lower, ci_upper).
        """
        if metric_fn is None:
            metric_fn = accuracy_score

        rng = np.random.RandomState(random_state)
        n = len(y_true)
        point_estimate = metric_fn(y_true, y_pred)

        scores: list[float] = []
        for _ in range(n_bootstrap):
            indices = rng.randint(0, n, size=n)
            score = metric_fn(y_true[indices], y_pred[indices])
            scores.append(score)

        alpha = (1 - ci) / 2
        ci_lower = float(np.percentile(scores, alpha * 100))
        ci_upper = float(np.percentile(scores, (1 - alpha) * 100))

        return point_estimate, ci_lower, ci_upper

    # ========================================================================
    # Plotting Methods
    # ========================================================================

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        labels: list[str] | None = None,
        save_path: str | None = None,
        title: str = "Confusion Matrix",
        normalize: bool = True,
        figsize: tuple[int, int] = (8, 6),
    ) -> None:
        """Plot a confusion matrix heatmap.

        Args:
            cm: Confusion matrix array.
            labels: Class labels for axes.
            save_path: Path to save the plot. If None, displays it.
            title: Plot title.
            normalize: Whether to show normalized (percentage) values.
            figsize: Figure size.
        """
        labels = labels or self.class_names

        fig, ax = plt.subplots(figsize=figsize)

        if normalize:
            row_sums = cm.sum(axis=1, keepdims=True)
            cm_normalized = np.divide(
                cm.astype(float),
                row_sums,
                out=np.zeros_like(cm, dtype=float),
                where=row_sums != 0,
            )
            sns.heatmap(
                cm_normalized,
                annot=True,
                fmt=".2%",
                cmap="Blues",
                xticklabels=labels,
                yticklabels=labels,
                ax=ax,
            )
        else:
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=labels,
                yticklabels=labels,
                ax=ax,
            )

        ax.set_title(title)
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def plot_feature_importance(
        self,
        importances: dict[str, float],
        save_path: str | None = None,
        title: str = "Feature Importance",
        top_n: int = 15,
        figsize: tuple[int, int] = (10, 6),
    ) -> None:
        """Plot feature importances as a horizontal bar chart.

        Args:
            importances: Dictionary mapping feature names to importance values.
            save_path: Path to save the plot.
            title: Plot title.
            top_n: Number of top features to show.
            figsize: Figure size.
        """
        sorted_items = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        top_items = sorted_items[:top_n]

        names = [item[0] for item in reversed(top_items)]
        values = [item[1] for item in reversed(top_items)]

        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(names, values, color="steelblue")
        ax.set_title(title)
        ax.set_xlabel("Importance")
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def plot_accuracy_comparison(
        self,
        results: dict[str, dict[str, Any]],
        save_path: str | None = None,
        title: str = "Method Accuracy Comparison",
        figsize: tuple[int, int] = (10, 6),
    ) -> None:
        """Plot accuracy comparison bar chart across methods.

        Args:
            results: Dictionary mapping method names to result dicts.
                Each result dict should have 'mean_accuracy' and optionally
                'std_accuracy' or 'accuracy'.
            save_path: Path to save the plot.
            title: Plot title.
            figsize: Figure size.
        """
        methods = list(results.keys())
        accuracies = []
        errors = []

        for method in methods:
            r = results[method]
            if "mean_accuracy" in r:
                accuracies.append(r["mean_accuracy"])
                errors.append(r.get("std_accuracy", 0))
            else:
                accuracies.append(r.get("accuracy", 0))
                errors.append(0)

        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.bar(
            methods, accuracies, yerr=errors, capsize=5,
            color=["#2196F3", "#4CAF50", "#FF9800"],
            edgecolor="black", linewidth=0.5,
        )

        # Add value labels on bars
        for bar, acc, err in zip(bars, accuracies, errors):
            label = f"{acc:.1%}"
            if err > 0:
                label += f" +/- {err:.1%}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + err + 0.01,
                label,
                ha="center",
                va="bottom",
                fontsize=10,
            )

        ax.set_title(title)
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.15)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def plot_per_class_f1(
        self,
        results: dict[str, dict[str, Any]],
        save_path: str | None = None,
        title: str = "Per-Class F1 Score Comparison",
        figsize: tuple[int, int] = (12, 6),
    ) -> None:
        """Plot grouped bar chart of F1 scores per class across methods.

        Args:
            results: Dictionary mapping method names to result dicts.
                Each must contain 'per_class_metrics' from classification_report.
            save_path: Path to save the plot.
            title: Plot title.
            figsize: Figure size.
        """
        methods = list(results.keys())
        n_methods = len(methods)
        n_classes = len(self.class_names)

        x = np.arange(n_classes)
        width = 0.8 / n_methods
        colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0"]

        fig, ax = plt.subplots(figsize=figsize)

        for i, method in enumerate(methods):
            report = results[method].get("per_class_metrics", {})
            f1_scores = []
            for cls in self.class_names:
                cls_metrics = report.get(cls, {})
                f1_scores.append(cls_metrics.get("f1-score", 0))

            offset = (i - n_methods / 2 + 0.5) * width
            ax.bar(
                x + offset,
                f1_scores,
                width,
                label=method,
                color=colors[i % len(colors)],
            )

        ax.set_title(title)
        ax.set_ylabel("F1 Score")
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names, rotation=45, ha="right")
        ax.set_ylim(0, 1.1)
        ax.legend()
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def generate_report(
        self,
        results_dict: dict[str, dict[str, Any]],
        save_path: str,
    ) -> None:
        """Generate a comprehensive text report comparing all methods.

        Args:
            results_dict: Dictionary mapping method names to evaluation results.
            save_path: Path to save the report file.
        """
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

        lines: list[str] = []
        lines.append("=" * 70)
        lines.append("GESTURE CLASSIFICATION - EVALUATION REPORT")
        lines.append("=" * 70)
        lines.append("")

        # Summary table
        lines.append("SUMMARY")
        lines.append("-" * 70)
        lines.append(f"{'Method':<20} {'Accuracy':<15} {'Std':<10} {'Latency (ms)':<15}")
        lines.append("-" * 70)

        for method, results in results_dict.items():
            acc = results.get("mean_accuracy", results.get("accuracy", 0))
            std = results.get("std_accuracy", 0)
            latency = results.get("latency_ms", 0)
            lines.append(
                f"{method:<20} {acc:<15.4f} {std:<10.4f} {latency:<15.3f}"
            )

        lines.append("")

        # Per-method details
        for method, results in results_dict.items():
            lines.append(f"\n{'=' * 70}")
            lines.append(f"METHOD: {method}")
            lines.append(f"{'=' * 70}")

            if "per_fold_accuracy" in results:
                folds = results["per_fold_accuracy"]
                lines.append(f"\nLOPO-CV Fold Accuracies ({len(folds)} folds):")
                for i, acc in enumerate(folds):
                    lines.append(f"  Fold {i+1}: {acc:.4f}")

            if "per_class_metrics" in results:
                report = results["per_class_metrics"]
                lines.append("\nPer-Class Metrics:")
                lines.append(
                    f"  {'Class':<15} {'Precision':<12} {'Recall':<12} "
                    f"{'F1-Score':<12} {'Support':<10}"
                )
                lines.append("  " + "-" * 60)
                for cls in self.class_names:
                    if cls in report:
                        m = report[cls]
                        lines.append(
                            f"  {cls:<15} {m['precision']:<12.4f} "
                            f"{m['recall']:<12.4f} {m['f1-score']:<12.4f} "
                            f"{m.get('support', 0):<10}"
                        )

            if "bootstrap_ci" in results:
                point, lower, upper = results["bootstrap_ci"]
                lines.append(f"\n95% Bootstrap CI: {point:.4f} [{lower:.4f}, {upper:.4f}]")

        lines.append(f"\n{'=' * 70}")
        lines.append("END OF REPORT")
        lines.append(f"{'=' * 70}")

        report_text = "\n".join(lines)
        with open(save_path, "w") as f:
            f.write(report_text)

        print(report_text)


if __name__ == "__main__":
    import tempfile

    from preprocessing import generate_sample_csv, preprocess_dataset
    from heuristic import HeuristicClassifier

    # Generate sample data
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        tmp_path = f.name

    df = generate_sample_csv(tmp_path, n_persons=4, samples_per_gesture=25)
    X, y, pids = preprocess_dataset(df)
    os.unlink(tmp_path)

    evaluator = Evaluator()
    clf = HeuristicClassifier()

    # LOPO-CV for heuristic (no training needed)
    def heuristic_train_predict(
        X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray
    ) -> np.ndarray:
        return np.array(clf.predict(X_test))

    print("Running LOPO-CV for Heuristic classifier...")
    results = evaluator.lopo_cv(heuristic_train_predict, X, y, pids)
    print(f"Mean accuracy: {results['mean_accuracy']:.3f} +/- {results['std_accuracy']:.3f}")

    # Confusion matrix
    cm = evaluator.compute_confusion_matrix(results["all_y_true"], results["all_y_pred"])
    print(f"\nConfusion Matrix:\n{cm}")

    # Bootstrap CI
    point, lower, upper = evaluator.bootstrap_ci(
        results["all_y_true"], results["all_y_pred"]
    )
    print(f"\n95% Bootstrap CI: {point:.3f} [{lower:.3f}, {upper:.3f}]")

    # Inference latency
    latency = evaluator.inference_latency(clf, X[0])
    print(f"\nInference latency: {latency:.3f} ms")
