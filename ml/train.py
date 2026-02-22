"""Training orchestrator for hand gesture classification pipeline.

Loads data, preprocesses, trains all three classifiers (Heuristic,
Random Forest, MLP), runs full evaluation (LOPO-CV + stratified split),
generates comparison plots, and saves models.

Usage:
    python train.py --data_dir ../data --output_dir ../models

If no CSV data exists, generates synthetic sample data automatically.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any

import numpy as np

# Add the ml directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing import (
    GESTURE_CLASSES,
    generate_sample_csv,
    load_data,
    preprocess_dataset,
)
from features import (
    compute_heuristic_features_batch,
    compute_mlp_features,
    compute_pairwise_distances_batch,
)
from heuristic import HeuristicClassifier
from random_forest import RFClassifier
from mlp import MLPClassifier
from evaluate import Evaluator


def find_csv_files(data_dir: str) -> list[str]:
    """Find all CSV files in the data directory.

    Args:
        data_dir: Path to data directory.

    Returns:
        List of CSV file paths.
    """
    csv_files = []
    if os.path.exists(data_dir):
        for fname in os.listdir(data_dir):
            if fname.endswith(".csv"):
                csv_files.append(os.path.join(data_dir, fname))
    return sorted(csv_files)


def train_heuristic(
    X: np.ndarray,
    y: np.ndarray,
    person_ids: np.ndarray,
    evaluator: Evaluator,
    output_dir: str,
    use_group_kfold: bool = False,
) -> dict[str, Any]:
    """Train and evaluate the heuristic classifier.

    Args:
        X: Normalized landmarks (n_samples, 60).
        y: Labels.
        person_ids: Person IDs for LOPO-CV.
        evaluator: Evaluator instance.
        output_dir: Directory for saving outputs.

    Returns:
        Evaluation results dictionary.
    """
    print("\n" + "=" * 60)
    print("METHOD 1: HEURISTIC CLASSIFIER")
    print("=" * 60)

    clf = HeuristicClassifier()

    # LOPO-CV (heuristic needs no training)
    def train_predict_fn(
        X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray
    ) -> np.ndarray:
        return np.array(clf.predict(X_test))

    print("\nRunning cross-validation...")
    if use_group_kfold:
        lopo_results = evaluator.group_kfold_cv(train_predict_fn, X, y, person_ids)
    else:
        lopo_results = evaluator.lopo_cv(train_predict_fn, X, y, person_ids)
    print(
        f"CV Accuracy: {lopo_results['mean_accuracy']:.4f} "
        f"+/- {lopo_results['std_accuracy']:.4f}"
    )

    # Stratified split
    print("Running stratified split evaluation...")
    split_results = evaluator.stratified_split(train_predict_fn, X, y)
    print(f"Stratified Split Accuracy: {split_results['accuracy']:.4f}")

    # Inference latency
    latency = evaluator.inference_latency(clf, X[0])
    print(f"Inference Latency: {latency:.3f} ms")

    # Bootstrap CI
    point, lower, upper = evaluator.bootstrap_ci(
        lopo_results["all_y_true"], lopo_results["all_y_pred"]
    )
    print(f"95% CI: {point:.4f} [{lower:.4f}, {upper:.4f}]")

    # Confusion matrix plot
    cm = evaluator.compute_confusion_matrix(
        lopo_results["all_y_true"], lopo_results["all_y_pred"]
    )
    evaluator.plot_confusion_matrix(
        cm,
        save_path=os.path.join(output_dir, "plots", "heuristic_confusion_matrix.png"),
        title="Heuristic Classifier - Confusion Matrix",
    )

    # Save model parameters
    clf.save(os.path.join(output_dir, "heuristic_params.json"))

    return {
        **lopo_results,
        "latency_ms": latency,
        "bootstrap_ci": (point, lower, upper),
        "split_accuracy": split_results["accuracy"],
    }


def train_random_forest(
    X: np.ndarray,
    y: np.ndarray,
    person_ids: np.ndarray,
    evaluator: Evaluator,
    output_dir: str,
    use_group_kfold: bool = False,
) -> dict[str, Any]:
    """Train and evaluate the Random Forest classifier.

    Args:
        X: Normalized landmarks (n_samples, 60).
        y: Labels.
        person_ids: Person IDs for LOPO-CV.
        evaluator: Evaluator instance.
        output_dir: Directory for saving outputs.

    Returns:
        Evaluation results dictionary.
    """
    print("\n" + "=" * 60)
    print("METHOD 2: RANDOM FOREST CLASSIFIER")
    print("=" * 60)

    # Extract pairwise distance features
    print("Extracting pairwise distance features...")
    X_dist, feat_names = compute_pairwise_distances_batch(X)
    print(f"Feature vector dimension: {X_dist.shape[1]}")

    # LOPO-CV
    def train_predict_fn(
        X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray
    ) -> np.ndarray:
        clf = RFClassifier()
        clf.train(X_train, y_train, feat_names)
        return clf.predict(X_test)

    print("\nRunning cross-validation...")
    if use_group_kfold:
        lopo_results = evaluator.group_kfold_cv(train_predict_fn, X_dist, y, person_ids)
    else:
        lopo_results = evaluator.lopo_cv(train_predict_fn, X_dist, y, person_ids)
    print(
        f"CV Accuracy: {lopo_results['mean_accuracy']:.4f} "
        f"+/- {lopo_results['std_accuracy']:.4f}"
    )

    # Stratified split
    print("Running stratified split evaluation...")
    split_results = evaluator.stratified_split(train_predict_fn, X_dist, y)
    print(f"Stratified Split Accuracy: {split_results['accuracy']:.4f}")

    # Train final model on all data for saving
    print("Training final model on all data...")
    final_clf = RFClassifier()
    train_metrics = final_clf.train(X_dist, y, feat_names)
    print(f"Final training accuracy: {train_metrics['train_accuracy']:.4f}")

    # Inference latency
    latency = evaluator.inference_latency(final_clf, X_dist[0])
    print(f"Inference Latency: {latency:.3f} ms")

    # Bootstrap CI
    point, lower, upper = evaluator.bootstrap_ci(
        lopo_results["all_y_true"], lopo_results["all_y_pred"]
    )
    print(f"95% CI: {point:.4f} [{lower:.4f}, {upper:.4f}]")

    # Confusion matrix plot
    cm = evaluator.compute_confusion_matrix(
        lopo_results["all_y_true"], lopo_results["all_y_pred"]
    )
    evaluator.plot_confusion_matrix(
        cm,
        save_path=os.path.join(output_dir, "plots", "rf_confusion_matrix.png"),
        title="Random Forest - Confusion Matrix",
    )

    # Feature importance plot
    importances = final_clf.get_feature_importance()
    evaluator.plot_feature_importance(
        importances,
        save_path=os.path.join(output_dir, "plots", "rf_feature_importance.png"),
        title="Random Forest - Feature Importance",
    )

    # Save model
    final_clf.save(os.path.join(output_dir, "random_forest.joblib"))

    return {
        **lopo_results,
        "latency_ms": latency,
        "bootstrap_ci": (point, lower, upper),
        "split_accuracy": split_results["accuracy"],
        "feature_importance": importances,
    }


def train_mlp(
    X: np.ndarray,
    y: np.ndarray,
    person_ids: np.ndarray,
    evaluator: Evaluator,
    output_dir: str,
    epochs: int = 100,
    verbose: int = 0,
    use_group_kfold: bool = False,
) -> dict[str, Any]:
    """Train and evaluate the MLP classifier.

    Args:
        X: Normalized landmarks (n_samples, 60).
        y: Labels.
        person_ids: Person IDs for LOPO-CV.
        evaluator: Evaluator instance.
        output_dir: Directory for saving outputs.
        epochs: Maximum training epochs.
        verbose: Keras verbosity level.

    Returns:
        Evaluation results dictionary.
    """
    print("\n" + "=" * 60)
    print("METHOD 3: MLP NEURAL NETWORK")
    print("=" * 60)

    # Suppress TF warnings
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    X_mlp = compute_mlp_features(X)  # Same as X, just for consistency
    print(f"Input dimension: {X_mlp.shape[1]}")

    # LOPO-CV
    def train_predict_fn(
        X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray
    ) -> np.ndarray:
        clf = MLPClassifier()
        clf.train(X_train, y_train, epochs=epochs, verbose=0, patience=10)
        return clf.predict(X_test)

    print("\nRunning cross-validation (this may take a while)...")
    if use_group_kfold:
        lopo_results = evaluator.group_kfold_cv(train_predict_fn, X_mlp, y, person_ids)
    else:
        lopo_results = evaluator.lopo_cv(train_predict_fn, X_mlp, y, person_ids)
    print(
        f"CV Accuracy: {lopo_results['mean_accuracy']:.4f} "
        f"+/- {lopo_results['std_accuracy']:.4f}"
    )

    # Stratified split
    print("Running stratified split evaluation...")
    split_results = evaluator.stratified_split(train_predict_fn, X_mlp, y)
    print(f"Stratified Split Accuracy: {split_results['accuracy']:.4f}")

    # Train final model on all data
    print("Training final model on all data...")
    final_clf = MLPClassifier()
    train_metrics = final_clf.train(
        X_mlp, y, epochs=epochs, verbose=verbose, patience=10
    )
    print(f"Final training metrics: {train_metrics}")

    # Inference latency
    latency = evaluator.inference_latency(final_clf, X_mlp[0])
    print(f"Inference Latency: {latency:.3f} ms")

    # Bootstrap CI
    point, lower, upper = evaluator.bootstrap_ci(
        lopo_results["all_y_true"], lopo_results["all_y_pred"]
    )
    print(f"95% CI: {point:.4f} [{lower:.4f}, {upper:.4f}]")

    # Confusion matrix plot
    cm = evaluator.compute_confusion_matrix(
        lopo_results["all_y_true"], lopo_results["all_y_pred"]
    )
    evaluator.plot_confusion_matrix(
        cm,
        save_path=os.path.join(output_dir, "plots", "mlp_confusion_matrix.png"),
        title="MLP - Confusion Matrix",
    )

    # Save model
    final_clf.save(os.path.join(output_dir, "mlp_model"))

    # Export ONNX
    try:
        final_clf.export_onnx(os.path.join(output_dir, "mlp_model.onnx"))
        print("ONNX export: OK")
    except Exception as e:
        print(f"ONNX export failed (non-critical): {e}")

    return {
        **lopo_results,
        "latency_ms": latency,
        "bootstrap_ci": (point, lower, upper),
        "split_accuracy": split_results["accuracy"],
        "train_metrics": train_metrics,
    }


def print_summary_table(results: dict[str, dict[str, Any]]) -> None:
    """Print a formatted summary comparison table.

    Args:
        results: Dictionary mapping method names to result dicts.
    """
    print("\n" + "=" * 80)
    print("SUMMARY COMPARISON")
    print("=" * 80)
    header = (
        f"{'Method':<20} {'LOPO Acc':<12} {'Std':<10} "
        f"{'Split Acc':<12} {'Latency':<12} {'95% CI':<20}"
    )
    print(header)
    print("-" * 80)

    for method, r in results.items():
        acc = r.get("mean_accuracy", r.get("accuracy", 0))
        std = r.get("std_accuracy", 0)
        split_acc = r.get("split_accuracy", 0)
        latency = r.get("latency_ms", 0)
        ci = r.get("bootstrap_ci", (0, 0, 0))

        print(
            f"{method:<20} {acc:<12.4f} {std:<10.4f} "
            f"{split_acc:<12.4f} {latency:<12.3f} "
            f"[{ci[1]:.4f}, {ci[2]:.4f}]"
        )

    print("=" * 80)


def main() -> None:
    """Main training orchestrator entry point."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate gesture classifiers"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "data"),
        help="Directory containing CSV data files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "models"),
        help="Directory for saving models and plots",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum training epochs for MLP",
    )
    parser.add_argument(
        "--generate_sample",
        action="store_true",
        help="Generate synthetic sample data if no CSV files found",
    )
    parser.add_argument(
        "--n_persons",
        type=int,
        default=5,
        help="Number of persons for sample data generation",
    )
    parser.add_argument(
        "--samples_per_gesture",
        type=int,
        default=30,
        help="Samples per gesture per person for sample data",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Verbosity level (0=silent, 1=normal, 2=detailed)",
    )

    args = parser.parse_args()
    data_dir = os.path.abspath(args.data_dir)
    output_dir = os.path.abspath(args.output_dir)

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)

    # ================================================================
    # Step 1: Load or generate data
    # ================================================================
    print("=" * 60)
    print("GESTURE CLASSIFICATION PIPELINE")
    print("=" * 60)

    csv_files = find_csv_files(data_dir)

    if not csv_files:
        if args.generate_sample:
            print("\nNo CSV files found. Generating synthetic sample data...")
            os.makedirs(data_dir, exist_ok=True)
            csv_path = os.path.join(data_dir, "sample_gestures.csv")
            generate_sample_csv(
                csv_path,
                n_persons=args.n_persons,
                samples_per_gesture=args.samples_per_gesture,
            )
            csv_files = [csv_path]
        else:
            print(f"\nNo CSV files found in {data_dir}")
            print("Use --generate_sample to create synthetic data for testing.")
            sys.exit(1)

    print(f"\nData files: {csv_files}")

    # Load and concatenate all CSV files
    import pandas as pd

    dfs = [load_data(f) for f in csv_files]
    df = pd.concat(dfs, ignore_index=True)
    print(f"Total samples: {len(df)}")
    print(f"Gesture distribution:\n{df['gesture_label'].value_counts().to_string()}")
    print(f"Unique persons: {df['person_id'].nunique()}")

    # ================================================================
    # Step 2: Preprocess
    # ================================================================
    print("\nPreprocessing landmarks...")
    X, y, person_ids = preprocess_dataset(df)
    print(f"Normalized feature shape: {X.shape}")

    # ================================================================
    # Step 3: Train and evaluate all methods
    # ================================================================
    evaluator = Evaluator()
    all_results: dict[str, dict[str, Any]] = {}

    # Choose CV strategy based on number of unique persons
    n_persons = len(np.unique(person_ids))
    MAX_LOPO_PERSONS = 20
    use_group_kfold = n_persons > MAX_LOPO_PERSONS
    if use_group_kfold:
        cv_name = f"Group 5-Fold CV ({n_persons} persons)"
        print(f"\nCV Strategy: {cv_name} (too many persons for LOPO)")
    else:
        cv_name = f"LOPO-CV ({n_persons} persons)"
        print(f"\nCV Strategy: {cv_name}")

    start_time = time.time()

    # Method 1: Heuristic
    all_results["Heuristic"] = train_heuristic(
        X, y, person_ids, evaluator, output_dir, use_group_kfold=use_group_kfold
    )

    # Method 2: Random Forest
    all_results["Random Forest"] = train_random_forest(
        X, y, person_ids, evaluator, output_dir, use_group_kfold=use_group_kfold
    )

    # Method 3: MLP
    all_results["MLP"] = train_mlp(
        X, y, person_ids, evaluator, output_dir,
        epochs=args.epochs,
        verbose=args.verbose,
        use_group_kfold=use_group_kfold,
    )

    total_time = time.time() - start_time

    # ================================================================
    # Step 4: Generate comparison plots
    # ================================================================
    print("\nGenerating comparison plots...")

    evaluator.plot_accuracy_comparison(
        all_results,
        save_path=os.path.join(output_dir, "plots", "accuracy_comparison.png"),
        title="Gesture Classification - Method Comparison",
    )

    evaluator.plot_per_class_f1(
        all_results,
        save_path=os.path.join(output_dir, "plots", "per_class_f1_comparison.png"),
        title="Gesture Classification - Per-Class F1 Comparison",
    )

    # ================================================================
    # Step 5: Generate report
    # ================================================================
    evaluator.generate_report(
        all_results,
        save_path=os.path.join(output_dir, "evaluation_report.txt"),
    )

    # ================================================================
    # Step 6: Summary
    # ================================================================
    print_summary_table(all_results)
    print(f"\nTotal pipeline time: {total_time:.1f} seconds")
    print(f"\nModels saved to: {output_dir}")
    print(f"Plots saved to: {os.path.join(output_dir, 'plots')}")


if __name__ == "__main__":
    main()
