"""Shared preprocessing for hand gesture classification.

Loads CSV data with MediaPipe 21-point 3D hand landmarks and normalizes
them via wrist-relative translation and palm-size scaling.

Schema: person_id, gesture_label, timestamp, x0,y0,z0, ..., x20,y20,z20
Output: 20x3 = 60-dim normalized vector per sample.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# MediaPipe landmark indices
NUM_LANDMARKS = 21
WRIST_IDX = 0
MIDDLE_FINGER_MCP_IDX = 9

# Column names for the 63 landmark coordinates (21 landmarks x 3 axes)
LANDMARK_COLS = [
    f"{axis}{i}" for i in range(NUM_LANDMARKS) for axis in ("x", "y", "z")
]

GESTURE_CLASSES = ["open_hand", "fist", "pinch", "frame", "none"]


def load_data(csv_path: str) -> pd.DataFrame:
    """Load gesture landmark CSV data.

    Args:
        csv_path: Path to CSV file with columns:
            person_id, gesture_label, timestamp, x0, y0, z0, ..., x20, y20, z20

    Returns:
        DataFrame with all columns preserved.
    """
    df = pd.read_csv(csv_path)
    expected_meta = {"person_id", "gesture_label", "timestamp"}
    missing_meta = expected_meta - set(df.columns)
    if missing_meta:
        raise ValueError(f"Missing metadata columns: {missing_meta}")

    missing_landmarks = [c for c in LANDMARK_COLS if c not in df.columns]
    if missing_landmarks:
        raise ValueError(
            f"Missing {len(missing_landmarks)} landmark columns. "
            f"First missing: {missing_landmarks[:5]}"
        )
    return df


def extract_landmarks(df: pd.DataFrame) -> np.ndarray:
    """Extract raw landmark coordinates from DataFrame.

    Args:
        df: DataFrame containing landmark columns x0,y0,z0,...,x20,y20,z20.

    Returns:
        Array of shape (n_samples, 21, 3).
    """
    raw = df[LANDMARK_COLS].values  # (n_samples, 63)
    return raw.reshape(-1, NUM_LANDMARKS, 3)


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Normalize a single sample of 21 3D landmarks.

    Steps:
        1. Wrist-relative translation: subtract landmark 0 from all landmarks.
        2. Scale normalization: divide by distance(wrist, middle_finger_MCP).
        3. Drop the wrist landmark (now all zeros) to produce 20x3 = 60 dims.

    Args:
        landmarks: Array of shape (21, 3).

    Returns:
        Flattened array of shape (60,) -- 20 landmarks x 3 coordinates.
    """
    landmarks = landmarks.astype(np.float64)

    # Step 1: wrist-relative translation
    wrist = landmarks[WRIST_IDX].copy()
    landmarks = landmarks - wrist

    # Step 2: scale by palm size (wrist to middle finger MCP distance)
    palm_size = np.linalg.norm(landmarks[MIDDLE_FINGER_MCP_IDX])
    if palm_size < 1e-8:
        # Degenerate case: all landmarks at same point
        palm_size = 1.0
    landmarks = landmarks / palm_size

    # Step 3: drop wrist (index 0) -- it is now [0, 0, 0]
    landmarks_no_wrist = landmarks[1:]  # (20, 3)
    return landmarks_no_wrist.flatten()  # (60,)


def normalize_landmarks_batch(landmarks_batch: np.ndarray) -> np.ndarray:
    """Normalize a batch of landmark arrays.

    Args:
        landmarks_batch: Array of shape (n_samples, 21, 3).

    Returns:
        Array of shape (n_samples, 60).
    """
    return np.array([normalize_landmarks(lm) for lm in landmarks_batch])


def preprocess_dataset(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full preprocessing pipeline on a DataFrame.

    Args:
        df: DataFrame loaded by ``load_data``.

    Returns:
        Tuple of:
            X_normalized: (n_samples, 60) normalized landmark vectors.
            y_labels: (n_samples,) string gesture labels.
            person_ids: (n_samples,) person identifiers.
    """
    landmarks = extract_landmarks(df)  # (n, 21, 3)
    X_normalized = normalize_landmarks_batch(landmarks)  # (n, 60)
    y_labels = df["gesture_label"].values
    person_ids = df["person_id"].values
    return X_normalized, y_labels, person_ids


def generate_sample_csv(
    csv_path: str,
    n_persons: int = 3,
    samples_per_gesture: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic CSV dataset for testing.

    Creates plausible hand landmark data for each gesture class with
    per-person variation. Useful when real data is not yet collected.

    Args:
        csv_path: Path to save the CSV file.
        n_persons: Number of simulated persons.
        samples_per_gesture: Samples per gesture per person.
        seed: Random seed for reproducibility.

    Returns:
        The generated DataFrame.
    """
    rng = np.random.RandomState(seed)
    rows: list[dict] = []
    timestamp = 0

    # Base hand shapes per gesture (21 landmarks x 3 coords)
    # These are rough approximations in a normalized coordinate space
    base_shapes = _get_base_shapes()

    for person_id in range(1, n_persons + 1):
        # Per-person scale and offset variation
        person_scale = 0.8 + rng.rand() * 0.4  # 0.8 to 1.2
        person_offset = rng.randn(3) * 0.1

        for gesture in GESTURE_CLASSES:
            shape = base_shapes[gesture]
            for _ in range(samples_per_gesture):
                # Add noise to base shape
                noisy = shape.copy() * person_scale + person_offset
                noisy += rng.randn(21, 3) * 0.02

                row: dict = {
                    "person_id": f"person_{person_id}",
                    "gesture_label": gesture,
                    "timestamp": timestamp,
                }
                for i in range(NUM_LANDMARKS):
                    row[f"x{i}"] = noisy[i, 0]
                    row[f"y{i}"] = noisy[i, 1]
                    row[f"z{i}"] = noisy[i, 2]

                rows.append(row)
                timestamp += 1

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    return df


def _get_base_shapes() -> dict[str, np.ndarray]:
    """Return approximate base landmark shapes for each gesture.

    All shapes are in an arbitrary coordinate frame; the preprocessing
    pipeline normalizes them afterward.
    """
    shapes: dict[str, np.ndarray] = {}

    # Wrist at origin, fingers extend upward (negative y in screen coords)
    # Using a simplified hand model

    # Helper: finger bases (MCP joints) spread across the palm
    palm_width = 0.08
    mcp_y = -0.10
    mcp_positions = np.array([
        [palm_width * 0.6, mcp_y * 0.5, 0.0],   # Thumb MCP (idx 1 approx)
        [palm_width * 0.3, mcp_y, 0.0],            # Index MCP
        [0.0, mcp_y * 1.05, 0.0],                  # Middle MCP
        [-palm_width * 0.3, mcp_y, 0.0],            # Ring MCP
        [-palm_width * 0.5, mcp_y * 0.9, 0.0],      # Pinky MCP
    ])

    # ---- Open Hand: all fingers extended ----
    open_hand = np.zeros((21, 3))
    # Wrist
    open_hand[0] = [0, 0, 0]
    # Thumb: landmarks 1-4
    open_hand[1] = [0.05, -0.02, 0.0]
    open_hand[2] = [0.08, -0.05, 0.0]
    open_hand[3] = [0.10, -0.09, 0.0]
    open_hand[4] = [0.11, -0.12, 0.0]  # Thumb tip
    # Index: landmarks 5-8
    open_hand[5] = [0.03, -0.10, 0.0]
    open_hand[6] = [0.03, -0.15, 0.0]
    open_hand[7] = [0.03, -0.19, 0.0]
    open_hand[8] = [0.03, -0.22, 0.0]  # Index tip
    # Middle: landmarks 9-12
    open_hand[9] = [0.0, -0.10, 0.0]
    open_hand[10] = [0.0, -0.16, 0.0]
    open_hand[11] = [0.0, -0.20, 0.0]
    open_hand[12] = [0.0, -0.24, 0.0]  # Middle tip
    # Ring: landmarks 13-16
    open_hand[13] = [-0.03, -0.10, 0.0]
    open_hand[14] = [-0.03, -0.15, 0.0]
    open_hand[15] = [-0.03, -0.19, 0.0]
    open_hand[16] = [-0.03, -0.22, 0.0]  # Ring tip
    # Pinky: landmarks 17-20
    open_hand[17] = [-0.05, -0.09, 0.0]
    open_hand[18] = [-0.05, -0.13, 0.0]
    open_hand[19] = [-0.05, -0.16, 0.0]
    open_hand[20] = [-0.05, -0.19, 0.0]  # Pinky tip
    shapes["open_hand"] = open_hand

    # ---- Fist: all fingers curled ----
    fist = np.zeros((21, 3))
    fist[0] = [0, 0, 0]
    # Thumb curled across
    fist[1] = [0.04, -0.02, 0.0]
    fist[2] = [0.06, -0.04, 0.01]
    fist[3] = [0.05, -0.06, 0.02]
    fist[4] = [0.03, -0.07, 0.03]
    # Index curled
    fist[5] = [0.03, -0.10, 0.0]
    fist[6] = [0.03, -0.12, 0.02]
    fist[7] = [0.02, -0.10, 0.04]
    fist[8] = [0.02, -0.08, 0.03]
    # Middle curled
    fist[9] = [0.0, -0.10, 0.0]
    fist[10] = [0.0, -0.12, 0.02]
    fist[11] = [-0.01, -0.10, 0.04]
    fist[12] = [-0.01, -0.08, 0.03]
    # Ring curled
    fist[13] = [-0.03, -0.10, 0.0]
    fist[14] = [-0.03, -0.12, 0.02]
    fist[15] = [-0.03, -0.10, 0.04]
    fist[16] = [-0.03, -0.08, 0.03]
    # Pinky curled
    fist[17] = [-0.05, -0.09, 0.0]
    fist[18] = [-0.05, -0.11, 0.02]
    fist[19] = [-0.05, -0.09, 0.04]
    fist[20] = [-0.05, -0.07, 0.03]
    shapes["fist"] = fist

    # ---- Pinch: thumb and index touching, others extended ----
    pinch = open_hand.copy()
    # Move thumb tip close to index tip
    pinch[3] = [0.04, -0.10, 0.02]
    pinch[4] = [0.03, -0.13, 0.02]  # Near index tip area
    # Index slightly curled toward thumb
    pinch[7] = [0.03, -0.16, 0.02]
    pinch[8] = [0.03, -0.14, 0.02]  # Index tip near thumb tip
    shapes["pinch"] = pinch

    # ---- Frame: open hand variant (single-hand component) ----
    # L-shape: thumb and index extended, others curled
    frame = fist.copy()
    # Thumb extended
    frame[1] = [0.05, -0.02, 0.0]
    frame[2] = [0.08, -0.05, 0.0]
    frame[3] = [0.10, -0.09, 0.0]
    frame[4] = [0.11, -0.12, 0.0]
    # Index extended
    frame[5] = [0.03, -0.10, 0.0]
    frame[6] = [0.03, -0.15, 0.0]
    frame[7] = [0.03, -0.19, 0.0]
    frame[8] = [0.03, -0.22, 0.0]
    shapes["frame"] = frame

    # ---- None: relaxed / ambiguous pose ----
    none_shape = open_hand.copy()
    # Slightly curled, not matching any clear pattern
    none_shape[4] = [0.09, -0.08, 0.02]   # Thumb partially curled
    none_shape[8] = [0.03, -0.18, 0.01]   # Index partially curled
    none_shape[12] = [0.0, -0.20, 0.01]   # Middle partially curled
    none_shape[16] = [-0.03, -0.18, 0.01]  # Ring partially curled
    none_shape[20] = [-0.05, -0.15, 0.01]  # Pinky partially curled
    shapes["none"] = none_shape

    return shapes


if __name__ == "__main__":
    import os

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, "sample_gestures.csv")

    print(f"Generating sample CSV at: {csv_path}")
    df = generate_sample_csv(csv_path, n_persons=5, samples_per_gesture=30)
    print(f"Generated {len(df)} samples")
    print(f"Gestures: {df['gesture_label'].value_counts().to_dict()}")
    print(f"Persons: {df['person_id'].nunique()}")

    print("\nRunning preprocessing...")
    X, y, pids = preprocess_dataset(df)
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Unique labels: {np.unique(y)}")
    print(f"Sample normalized vector (first 10 dims): {X[0, :10]}")
