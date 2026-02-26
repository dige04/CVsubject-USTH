#!/usr/bin/env python
"""Unified evaluation: collect results from all methods, generate comparison tables.

Aggregates per-fold metrics from MLP, CNN, Fusion, and YOLO into a single
evaluation framework.  Produces:

    - Markdown comparison table (sorted by accuracy)
    - Markdown ablation table (modality contributions)
    - Per-method confusion matrices (PNG)
    - Per-class F1 grouped bar chart (PNG)
    - Latency benchmark table
    - Full JSON dump of all metrics

Usage::

    python ml/scripts/unified_evaluation.py \\
        --results-dir results/ablation \\
        --output-dir  results/evaluation
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# Add parent directory for local imports.
_ML_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ML_DIR not in sys.path:
    sys.path.insert(0, _ML_DIR)

from preprocessing import GESTURE_CLASSES  # noqa: E402

# Canonical sorted class names -- consistent across all evaluation code.
CLASS_NAMES = sorted(GESTURE_CLASSES)  # ["fist", "frame", "none", "open_hand", "pinch"]

# Colours reused from cv_visualizations.py for visual consistency.
GESTURE_COLORS = {
    "fist": "#F44336",
    "frame": "#4CAF50",
    "none": "#9E9E9E",
    "open_hand": "#2196F3",
    "pinch": "#FF9800",
}

METHOD_COLORS = [
    "#2196F3",  # blue
    "#4CAF50",  # green
    "#FF9800",  # orange
    "#E91E63",  # pink
    "#9C27B0",  # purple
    "#00BCD4",  # cyan
    "#795548",  # brown
]


class UnifiedEvaluation:
    """Collect and compare results across all gesture recognition methods.

    Attributes:
        class_names: Sorted list of gesture class names.
        results: Mapping of method name to aggregated metrics.
        yolo_results: Separate storage for YOLO detection metrics (different
            metric space -- mAP vs accuracy).
    """

    def __init__(
        self,
        class_names: list[str] | None = None,
    ) -> None:
        self.class_names = class_names or CLASS_NAMES
        self.results: dict[str, dict[str, Any]] = {}
        self.yolo_results: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Adding results
    # ------------------------------------------------------------------

    def add_result(
        self,
        method_name: str,
        fold_metrics: list[dict[str, Any]],
    ) -> None:
        """Register per-fold metrics for a classification method.

        Args:
            method_name: Human-readable method name (e.g. "MLP (pose-only)").
            fold_metrics: List of dicts, one per fold.  Each dict must contain
                ``accuracy`` (float), ``f1_macro`` (float), ``y_true`` and
                ``y_pred`` (arrays of string labels or integer indices).
        """
        accs = [m["accuracy"] for m in fold_metrics]
        f1s = [m["f1_macro"] for m in fold_metrics]

        all_y_true = np.concatenate([np.asarray(m["y_true"]) for m in fold_metrics])
        all_y_pred = np.concatenate([np.asarray(m["y_pred"]) for m in fold_metrics])

        report = classification_report(
            all_y_true,
            all_y_pred,
            labels=self.class_names,
            output_dict=True,
            zero_division=0,
        )

        self.results[method_name] = {
            "per_fold_accuracy": accs,
            "mean_accuracy": float(np.mean(accs)),
            "std_accuracy": float(np.std(accs)),
            "mean_f1": float(np.mean(f1s)),
            "std_f1": float(np.std(f1s)),
            "all_y_true": all_y_true,
            "all_y_pred": all_y_pred,
            "per_class_metrics": report,
            "n_folds": len(fold_metrics),
        }

    def add_yolo_results(
        self,
        maps50: list[float],
        maps95: list[float],
        derived_accs: list[float] | None = None,
    ) -> None:
        """Register YOLO detection results (separate metric space).

        Args:
            maps50: Per-fold mAP@50 values.
            maps95: Per-fold mAP@50-95 values.
            derived_accs: Per-fold classification accuracy derived from
                detection outputs (optional).
        """
        self.yolo_results = {
            "mAP50_mean": float(np.mean(maps50)),
            "mAP50_std": float(np.std(maps50)),
            "mAP95_mean": float(np.mean(maps95)),
            "mAP95_std": float(np.std(maps95)),
            "per_fold_mAP50": maps50,
            "per_fold_mAP95": maps95,
        }
        if derived_accs is not None:
            self.yolo_results["derived_acc_mean"] = float(np.mean(derived_accs))
            self.yolo_results["derived_acc_std"] = float(np.std(derived_accs))
            self.yolo_results["per_fold_derived_acc"] = derived_accs

    # ------------------------------------------------------------------
    # Table generation
    # ------------------------------------------------------------------

    def comparison_table(self) -> str:
        """Generate a markdown comparison table sorted by accuracy.

        Returns:
            Markdown-formatted string.
        """
        lines = [
            "| Method | Accuracy (%) | F1-macro | Folds |",
            "|--------|:------------:|:--------:|:-----:|",
        ]

        sorted_methods = sorted(
            self.results.items(),
            key=lambda x: x[1]["mean_accuracy"],
            reverse=True,
        )

        for name, r in sorted_methods:
            acc = (
                f"{r['mean_accuracy'] * 100:.1f} "
                f"+/- {r['std_accuracy'] * 100:.1f}"
            )
            f1 = f"{r['mean_f1']:.3f} +/- {r['std_f1']:.3f}"
            folds = r.get("n_folds", "?")
            lines.append(f"| {name} | {acc} | {f1} | {folds} |")

        # Append YOLO if available.
        if self.yolo_results is not None:
            yr = self.yolo_results
            acc_str = "--"
            if "derived_acc_mean" in yr:
                acc_str = (
                    f"{yr['derived_acc_mean'] * 100:.1f} "
                    f"+/- {yr['derived_acc_std'] * 100:.1f}"
                )
            map_str = (
                f"mAP50={yr['mAP50_mean']:.3f}, "
                f"mAP95={yr['mAP95_mean']:.3f}"
            )
            lines.append(f"| YOLOv8n (detection) | {acc_str} | {map_str} | -- |")

        return "\n".join(lines)

    def ablation_table(self) -> str:
        """Generate a markdown ablation table showing modality contributions.

        Returns:
            Markdown-formatted string.
        """
        modality_map = {
            "MLP (pose-only)":       {"lm": "x", "app": "",  "fusion": "--"},
            "CNN (appearance-only)": {"lm": "",  "app": "x", "fusion": "--"},
            "Weighted Average":      {"lm": "x", "app": "x", "fusion": "W. Avg"},
            "Learned Fusion":        {"lm": "x", "app": "x", "fusion": "Concat+MLP"},
        }

        lines = [
            "| # | Model | Landmarks | Appearance | Fusion | Accuracy (%) | F1-macro |",
            "|---|-------|:---------:|:----------:|:------:|:------------:|:--------:|",
        ]

        row_order = [
            "MLP (pose-only)",
            "CNN (appearance-only)",
            "Weighted Average",
            "Learned Fusion",
        ]

        idx = 1
        for name in row_order:
            if name not in self.results:
                continue
            r = self.results[name]
            m = modality_map.get(
                name, {"lm": "?", "app": "?", "fusion": "?"}
            )
            acc = (
                f"{r['mean_accuracy'] * 100:.1f} "
                f"+/- {r['std_accuracy'] * 100:.1f}"
            )
            f1 = f"{r['mean_f1']:.3f} +/- {r['std_f1']:.3f}"
            lines.append(
                f"| {idx} | {name} | {m['lm']} | {m['app']} "
                f"| {m['fusion']} | {acc} | {f1} |"
            )
            idx += 1

        # Additional methods not in the canonical ablation order.
        for name, r in self.results.items():
            if name in row_order:
                continue
            acc = (
                f"{r['mean_accuracy'] * 100:.1f} "
                f"+/- {r['std_accuracy'] * 100:.1f}"
            )
            f1 = f"{r['mean_f1']:.3f} +/- {r['std_f1']:.3f}"
            lines.append(
                f"| {idx} | {name} | ? | ? | ? | {acc} | {f1} |"
            )
            idx += 1

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def generate_confusion_matrices(
        self,
        output_dir: str,
        normalize: bool = True,
        figsize_per: tuple[int, int] = (6, 5),
    ) -> list[str]:
        """Save a confusion matrix heatmap for each method.

        Args:
            output_dir: Directory to save PNG files.
            normalize: Whether to row-normalize the matrix.
            figsize_per: ``(width, height)`` for each subplot.

        Returns:
            List of saved file paths.
        """
        os.makedirs(output_dir, exist_ok=True)
        saved: list[str] = []

        for name, r in self.results.items():
            y_true = r["all_y_true"]
            y_pred = r["all_y_pred"]
            cm = confusion_matrix(y_true, y_pred, labels=self.class_names)

            fig, ax = plt.subplots(figsize=figsize_per)

            if normalize:
                row_sums = cm.sum(axis=1, keepdims=True)
                cm_plot = np.divide(
                    cm.astype(float),
                    row_sums,
                    out=np.zeros_like(cm, dtype=float),
                    where=row_sums != 0,
                )
                sns.heatmap(
                    cm_plot,
                    annot=True,
                    fmt=".2%",
                    cmap="Blues",
                    xticklabels=self.class_names,
                    yticklabels=self.class_names,
                    ax=ax,
                )
            else:
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    xticklabels=self.class_names,
                    yticklabels=self.class_names,
                    ax=ax,
                )

            safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
            ax.set_title(f"Confusion Matrix -- {name}")
            ax.set_ylabel("True Label")
            ax.set_xlabel("Predicted Label")
            plt.tight_layout()

            path = os.path.join(output_dir, f"cm_{safe_name}.png")
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            saved.append(path)

        return saved

    def generate_f1_chart(
        self,
        output_dir: str,
        figsize: tuple[int, int] = (12, 6),
    ) -> str:
        """Save a grouped bar chart of per-class F1 scores across methods.

        Args:
            output_dir: Directory for the output PNG.
            figsize: Figure size.

        Returns:
            Path to the saved figure.
        """
        os.makedirs(output_dir, exist_ok=True)

        methods = list(self.results.keys())
        n_methods = len(methods)
        n_classes = len(self.class_names)
        x = np.arange(n_classes)
        width = 0.8 / max(n_methods, 1)

        fig, ax = plt.subplots(figsize=figsize)

        for i, method in enumerate(methods):
            report = self.results[method].get("per_class_metrics", {})
            f1_scores = []
            for cls in self.class_names:
                cls_metrics = report.get(cls, {})
                f1_scores.append(cls_metrics.get("f1-score", 0.0))

            offset = (i - n_methods / 2 + 0.5) * width
            color = METHOD_COLORS[i % len(METHOD_COLORS)]
            ax.bar(
                x + offset,
                f1_scores,
                width,
                label=method,
                color=color,
            )

        ax.set_title("Per-Class F1 Score Comparison", fontsize=14, fontweight="bold")
        ax.set_ylabel("F1 Score")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [c.replace("_", " ").title() for c in self.class_names],
            rotation=45,
            ha="right",
        )
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        path = os.path.join(output_dir, "per_class_f1.png")
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()
        return path

    def generate_accuracy_chart(
        self,
        output_dir: str,
        figsize: tuple[int, int] = (10, 6),
    ) -> str:
        """Save an accuracy comparison bar chart with error bars.

        Args:
            output_dir: Directory for the output PNG.
            figsize: Figure size.

        Returns:
            Path to the saved figure.
        """
        os.makedirs(output_dir, exist_ok=True)

        methods = list(self.results.keys())
        accs = [self.results[m]["mean_accuracy"] for m in methods]
        stds = [self.results[m]["std_accuracy"] for m in methods]
        colors = [METHOD_COLORS[i % len(METHOD_COLORS)] for i in range(len(methods))]

        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.bar(
            methods, accs, yerr=stds, capsize=5,
            color=colors, edgecolor="black", linewidth=0.5,
        )

        for bar, acc, std in zip(bars, accs, stds):
            label = f"{acc:.1%}"
            if std > 0:
                label += f" +/- {std:.1%}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std + 0.01,
                label,
                ha="center",
                va="bottom",
                fontsize=9,
            )

        ax.set_title("Method Accuracy Comparison", fontsize=14, fontweight="bold")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.15)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()

        path = os.path.join(output_dir, "accuracy_comparison.png")
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()
        return path

    # ------------------------------------------------------------------
    # Latency benchmarking
    # ------------------------------------------------------------------

    @staticmethod
    def benchmark_latency(
        models: dict[str, Callable[..., Any]],
        input_data: Any,
        n_runs: int = 100,
        warmup: int = 10,
    ) -> dict[str, dict[str, float]]:
        """Measure inference latency for each model function.

        Args:
            models: Mapping of method name to a callable that performs
                inference on ``input_data``.
            input_data: The data to pass to each callable (e.g. a single
                sample numpy array).
            n_runs: Number of timed iterations.
            warmup: Number of untimed warmup iterations.

        Returns:
            Mapping of method name to ``{"mean_ms", "std_ms", "p95_ms"}``.
        """
        results: dict[str, dict[str, float]] = {}

        for name, model_fn in models.items():
            # Warm up.
            for _ in range(warmup):
                model_fn(input_data)

            # Timed runs.
            times: list[float] = []
            for _ in range(n_runs):
                start = time.perf_counter()
                model_fn(input_data)
                elapsed = time.perf_counter() - start
                times.append(elapsed)

            times_ms = np.array(times) * 1000
            results[name] = {
                "mean_ms": float(np.mean(times_ms)),
                "std_ms": float(np.std(times_ms)),
                "p95_ms": float(np.percentile(times_ms, 95)),
            }

        return results

    # ------------------------------------------------------------------
    # Save all artefacts
    # ------------------------------------------------------------------

    def save_all(self, output_dir: str) -> None:
        """Persist all tables, charts, and raw metrics to *output_dir*.

        Creates:
            - ``comparison_table.md``
            - ``ablation_table.md``
            - ``confusion_matrices/`` (one PNG per method)
            - ``per_class_f1.png``
            - ``accuracy_comparison.png``
            - ``metrics.json``

        Args:
            output_dir: Root output directory.
        """
        os.makedirs(output_dir, exist_ok=True)

        # --- Markdown tables ---
        comp_path = os.path.join(output_dir, "comparison_table.md")
        with open(comp_path, "w") as f:
            f.write(self.comparison_table())
        print(f"Saved: {comp_path}")

        abl_path = os.path.join(output_dir, "ablation_table.md")
        with open(abl_path, "w") as f:
            f.write(self.ablation_table())
        print(f"Saved: {abl_path}")

        # --- Confusion matrices ---
        cm_dir = os.path.join(output_dir, "confusion_matrices")
        cm_paths = self.generate_confusion_matrices(cm_dir)
        for p in cm_paths:
            print(f"Saved: {p}")

        # --- Charts ---
        f1_path = self.generate_f1_chart(output_dir)
        print(f"Saved: {f1_path}")

        acc_path = self.generate_accuracy_chart(output_dir)
        print(f"Saved: {acc_path}")

        # --- Raw JSON (strip numpy arrays for serialization) ---
        json_safe: dict[str, Any] = {}
        for name, r in self.results.items():
            json_safe[name] = {
                k: v for k, v in r.items()
                if k not in ("all_y_true", "all_y_pred", "per_class_metrics")
            }
            # Flatten per_class_metrics.
            if "per_class_metrics" in r:
                pcm = {}
                for cls in self.class_names:
                    if cls in r["per_class_metrics"]:
                        pcm[cls] = {
                            kk: float(vv) if isinstance(vv, (int, float)) else vv
                            for kk, vv in r["per_class_metrics"][cls].items()
                        }
                json_safe[name]["per_class_metrics"] = pcm

        if self.yolo_results is not None:
            json_safe["YOLOv8n (detection)"] = self.yolo_results

        json_path = os.path.join(output_dir, "metrics.json")
        with open(json_path, "w") as f:
            json.dump(json_safe, f, indent=2)
        print(f"Saved: {json_path}")

        # --- Print tables to stdout ---
        print("\n" + "=" * 70)
        print("COMPARISON TABLE")
        print("=" * 70)
        print(self.comparison_table())

        print("\n" + "=" * 70)
        print("ABLATION TABLE")
        print("=" * 70)
        print(self.ablation_table())


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Unified evaluation: generate comparison tables and plots.",
    )
    parser.add_argument(
        "--results-dir",
        default="results/ablation",
        help="Directory containing ablation_results.json.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/evaluation",
        help="Output directory for tables and plots.",
    )
    parser.add_argument(
        "--yolo-json",
        default=None,
        help="Path to YOLO metrics JSON (optional).",
    )

    args = parser.parse_args(argv)

    evaluator = UnifiedEvaluation()

    # Load ablation results if available.
    ablation_json = os.path.join(args.results_dir, "ablation_results.json")
    if os.path.exists(ablation_json):
        with open(ablation_json) as f:
            ablation_data = json.load(f)

        for method_name, data in ablation_data.items():
            # Re-construct fold_metrics from the aggregated data.
            n_folds = len(data.get("per_fold_accuracy", []))
            if n_folds == 0:
                continue

            # We only have aggregated data; create synthetic fold entries.
            fold_metrics = []
            for i in range(n_folds):
                fold_metrics.append({
                    "accuracy": data["per_fold_accuracy"][i],
                    "f1_macro": data.get("mean_f1", data["per_fold_accuracy"][i]),
                    "y_true": np.array([CLASS_NAMES[0]]),  # placeholder
                    "y_pred": np.array([CLASS_NAMES[0]]),  # placeholder
                })

            # Override with direct metrics.
            accs = data["per_fold_accuracy"]
            evaluator.results[method_name] = {
                "per_fold_accuracy": accs,
                "mean_accuracy": float(np.mean(accs)),
                "std_accuracy": float(np.std(accs)),
                "mean_f1": data.get("mean_f1", float(np.mean(accs))),
                "std_f1": data.get("std_f1", 0.0),
                "all_y_true": np.array([CLASS_NAMES[0]]),
                "all_y_pred": np.array([CLASS_NAMES[0]]),
                "per_class_metrics": {},
                "n_folds": n_folds,
            }

        print(f"Loaded {len(evaluator.results)} methods from {ablation_json}")
    else:
        print(f"No ablation results found at {ablation_json}")
        print("Add results manually via the UnifiedEvaluation API.")

    # Load YOLO results if provided.
    if args.yolo_json and os.path.exists(args.yolo_json):
        with open(args.yolo_json) as f:
            yolo_data = json.load(f)
        evaluator.add_yolo_results(
            maps50=yolo_data.get("mAP50", []),
            maps95=yolo_data.get("mAP95", []),
            derived_accs=yolo_data.get("derived_accs", None),
        )
        print(f"Loaded YOLO results from {args.yolo_json}")

    # Generate all outputs.
    if evaluator.results:
        evaluator.save_all(args.output_dir)
    else:
        print("No results to evaluate. Run the ablation study first.")


if __name__ == "__main__":
    main()
