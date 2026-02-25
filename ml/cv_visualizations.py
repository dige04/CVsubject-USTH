"""Computer Vision visualizations for the hand gesture recognition project.

Generates:
1. Landmark skeleton overlay on a sample webcam frame
2. t-SNE 2D scatter plot of normalized 60D feature space
3. Gesture class distribution bar chart
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE

# MediaPipe hand landmark connections (21 keypoints)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # Index
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17),             # Palm
]

GESTURE_COLORS = {
    "open_hand": "#2196F3",
    "fist": "#F44336",
    "pinch": "#FF9800",
    "frame": "#4CAF50",
    "none": "#9E9E9E",
}

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "plots")


def plot_landmark_skeleton(csv_path: str, output_path: str | None = None):
    """Draw the 21-point hand skeleton for one sample per gesture class.

    This visualization demonstrates the core CV representation: converting
    a raw video frame into a structured 21-point skeletal graph.
    """
    df = pd.read_csv(csv_path)
    landmark_cols = [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    fig.suptitle("MediaPipe Hand Landmark Skeleton per Gesture Class",
                 fontsize=14, fontweight="bold", y=1.02)

    for ax, gesture in zip(axes, ["open_hand", "fist", "pinch", "frame", "none"]):
        subset = df[df["gesture_label"] == gesture]
        if subset.empty:
            ax.set_title(gesture)
            ax.axis("off")
            continue

        sample = subset.iloc[0]
        coords = sample[landmark_cols].values.reshape(21, 3)
        xs, ys = coords[:, 0], coords[:, 1]

        # Invert Y for display (image coordinates have Y increasing downward)
        ys_disp = -ys

        # Draw connections
        for start, end in HAND_CONNECTIONS:
            ax.plot([xs[start], xs[end]], [ys_disp[start], ys_disp[end]],
                    color=GESTURE_COLORS[gesture], linewidth=1.5, alpha=0.7)

        # Draw joints
        ax.scatter(xs, ys_disp, c=GESTURE_COLORS[gesture], s=30, zorder=5,
                   edgecolors="white", linewidths=0.5)

        # Highlight wrist (origin) and middle finger MCP (scale reference)
        ax.scatter([xs[0]], [ys_disp[0]], c="black", s=80, zorder=6,
                   marker="D", label="Wrist (origin)")
        ax.scatter([xs[9]], [ys_disp[9]], c="gold", s=80, zorder=6,
                   marker="s", label="MCP (scale ref)")

        ax.set_title(gesture.replace("_", " ").title(),
                     fontsize=12, fontweight="bold",
                     color=GESTURE_COLORS[gesture])
        ax.set_aspect("equal")
        ax.axis("off")

    # Shared legend
    legend_elements = [
        mpatches.Patch(color="black", label="Wrist (Translation Origin)"),
        mpatches.Patch(color="gold", label="MCP Joint (Scale Reference)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2,
               fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()
    out = output_path or os.path.join(PLOTS_DIR, "landmark_skeleton_per_class.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_tsne(csv_path: str, output_path: str | None = None):
    """Generate a t-SNE 2D embedding of the normalized 60D feature space.

    This visualization proves that the normalization pipeline produces
    well-separated clusters, validating the CV preprocessing design.
    """
    sys.path.insert(0, os.path.dirname(__file__))
    from preprocessing import preprocess_dataset, load_data

    df = load_data(csv_path)
    X, y, _ = preprocess_dataset(df)

    # Subsample for performance if dataset is large
    max_samples = 3000
    if len(X) > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X), max_samples, replace=False)
        X, y = X[idx], y[idx]

    print(f"Running t-SNE on {len(X)} samples...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    X_2d = tsne.fit_transform(X)

    fig, ax = plt.subplots(figsize=(10, 8))
    for gesture in ["open_hand", "fist", "pinch", "frame", "none"]:
        mask = y == gesture
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=GESTURE_COLORS[gesture], label=gesture.replace("_", " ").title(),
                   alpha=0.6, s=15, edgecolors="none")

    ax.set_title("t-SNE Visualization of Normalized 60D Landmark Features",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("t-SNE Dimension 1", fontsize=11)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=11)
    ax.legend(fontsize=10, markerscale=3, frameon=True)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out = output_path or os.path.join(PLOTS_DIR, "tsne_feature_space.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_class_distribution(csv_path: str, output_path: str | None = None):
    """Bar chart showing class distribution across the dataset."""
    df = pd.read_csv(csv_path)
    counts = df["gesture_label"].value_counts()

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        [g.replace("_", " ").title() for g in counts.index],
        counts.values,
        color=[GESTURE_COLORS.get(g, "#999") for g in counts.index],
        edgecolor="white", linewidth=1.2,
    )

    for bar, count in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                str(count), ha="center", va="bottom", fontweight="bold", fontsize=11)

    ax.set_title("Gesture Class Distribution in Dataset",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Number of Samples", fontsize=11)
    ax.set_xlabel("Gesture Class", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    out = output_path or os.path.join(PLOTS_DIR, "class_distribution.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    csv_candidates = [
        os.path.join(data_dir, "gestures.csv"),
        os.path.join(data_dir, "sample_gestures.csv"),
    ]

    csv_path = None
    for c in csv_candidates:
        if os.path.exists(c):
            csv_path = c
            break

    if csv_path is None:
        # Generate sample data for visualization
        sys.path.insert(0, os.path.dirname(__file__))
        from preprocessing import generate_sample_csv
        os.makedirs(data_dir, exist_ok=True)
        csv_path = os.path.join(data_dir, "sample_gestures.csv")
        generate_sample_csv(csv_path, n_persons=10, samples_per_gesture=100)

    print(f"Using dataset: {csv_path}")
    plot_landmark_skeleton(csv_path)
    plot_class_distribution(csv_path)
    plot_tsne(csv_path)
    print("All CV visualizations generated!")
