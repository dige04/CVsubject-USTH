"""Feature engineering for hand gesture classification.

Three feature extraction strategies for three classification methods:
    1. Heuristic: joint angles + key distances
    2. Random Forest: pairwise Euclidean distances
    3. MLP: raw 60-dim normalized vector (no extra extraction)

All functions expect wrist-relative, palm-size-normalized landmarks
as produced by ``preprocessing.normalize_landmarks``.
"""

from __future__ import annotations

import numpy as np

# ============================================================================
# MediaPipe landmark indices (wrist removed, so original index - 1)
# Original: 0=wrist, 1-4=thumb, 5-8=index, 9-12=middle, 13-16=ring, 17-20=pinky
# After wrist removal: 0-3=thumb, 4-7=index, 8-11=middle, 12-15=ring, 16-19=pinky
# ============================================================================

# Fingertip indices (in 20-landmark space, 0-indexed)
THUMB_TIP = 3
INDEX_TIP = 7
MIDDLE_TIP = 11
RING_TIP = 15
PINKY_TIP = 19

# MCP indices (base of each finger)
THUMB_MCP = 0       # CMC actually, but closest MCP-like joint
THUMB_IP = 1        # Thumb IP
THUMB_DIP = 2       # Thumb DIP (there is no DIP on thumb, this is IP)
INDEX_MCP = 4
INDEX_PIP = 5
INDEX_DIP = 6
MIDDLE_MCP = 8
MIDDLE_PIP = 9
MIDDLE_DIP = 10
RING_MCP = 12
RING_PIP = 13
RING_DIP = 14
PINKY_MCP = 16
PINKY_PIP = 17
PINKY_DIP = 18

FINGERTIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
MCPS = [THUMB_MCP, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]
PIPS = [THUMB_IP, INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP]
DIPS = [THUMB_DIP, INDEX_DIP, MIDDLE_DIP, RING_DIP, PINKY_DIP]

# Finger definitions: (MCP, PIP, DIP, TIP)
FINGERS = [
    (THUMB_MCP, THUMB_IP, THUMB_DIP, THUMB_TIP),
    (INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
    (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
    (RING_MCP, RING_PIP, RING_DIP, RING_TIP),
    (PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
]

FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]

# Palm center approximation: average of MCP joints
# (wrist is removed, so we use the 5 MCP positions)


def _reshape_to_landmarks(normalized_vector: np.ndarray) -> np.ndarray:
    """Reshape a 60-dim vector back to (20, 3) landmarks.

    Args:
        normalized_vector: Flat array of shape (60,).

    Returns:
        Array of shape (20, 3).
    """
    return normalized_vector.reshape(20, 3)


def _angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute the angle in degrees between two 3D vectors.

    Args:
        v1: First vector.
        v2: Second vector.

    Returns:
        Angle in degrees (0 to 180).
    """
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def _palm_center(lm: np.ndarray) -> np.ndarray:
    """Compute palm center as mean of MCP joints.

    Args:
        lm: Landmarks array of shape (20, 3).

    Returns:
        3D point representing palm center.
    """
    return np.mean(lm[MCPS], axis=0)


# ============================================================================
# Method 1: Heuristic Features
# ============================================================================


def compute_angles(normalized_vector: np.ndarray) -> dict[str, float]:
    """Compute 15 joint angles for heuristic classification.

    Angles computed:
        - 5x fingertip-PIP angles (angle at PIP joint between DIP and MCP)
        - 5x finger curl (MCP-PIP-TIP angle, measuring how curled each finger is)
        - 5x abduction angles (spread between adjacent fingers at MCP level)

    Args:
        normalized_vector: Flat (60,) normalized landmark vector.

    Returns:
        Dictionary mapping angle names to degree values.
    """
    lm = _reshape_to_landmarks(normalized_vector)
    angles: dict[str, float] = {}

    for i, (mcp, pip, dip, tip) in enumerate(FINGERS):
        name = FINGER_NAMES[i]

        # PIP joint angle: angle at PIP between vectors PIP->MCP and PIP->DIP
        v_pip_mcp = lm[mcp] - lm[pip]
        v_pip_dip = lm[dip] - lm[pip]
        angles[f"{name}_pip_angle"] = _angle_between_vectors(v_pip_mcp, v_pip_dip)

        # Finger curl: angle at PIP between MCP->PIP and PIP->TIP
        # A straight finger has ~180 degrees; curled finger is much less
        v_mcp_pip = lm[pip] - lm[mcp]
        v_pip_tip = lm[tip] - lm[pip]
        angles[f"{name}_curl_angle"] = _angle_between_vectors(v_mcp_pip, v_pip_tip)

    # Abduction angles: angle between adjacent fingers at MCP level
    # Measured as angle between MCP->TIP vectors of adjacent fingers
    for i in range(4):
        name_a = FINGER_NAMES[i]
        name_b = FINGER_NAMES[i + 1]
        mcp_a, _, _, tip_a = FINGERS[i]
        mcp_b, _, _, tip_b = FINGERS[i + 1]
        v_a = lm[tip_a] - lm[mcp_a]
        v_b = lm[tip_b] - lm[mcp_b]
        angles[f"abduction_{name_a}_{name_b}"] = _angle_between_vectors(v_a, v_b)

    # One more: thumb-pinky spread for overall hand openness
    v_thumb = lm[THUMB_TIP] - lm[THUMB_MCP]
    v_pinky = lm[PINKY_TIP] - lm[PINKY_MCP]
    angles["abduction_thumb_pinky"] = _angle_between_vectors(v_thumb, v_pinky)

    return angles


def compute_key_distances(normalized_vector: np.ndarray) -> dict[str, float]:
    """Compute key distances for heuristic classification.

    Distances computed:
        - thumb-index tip distance
        - each fingertip to palm center distance (5)
        - finger spread: max distance between adjacent fingertips (4)
        - overall spread: thumb tip to pinky tip

    Args:
        normalized_vector: Flat (60,) normalized landmark vector.

    Returns:
        Dictionary mapping distance names to values (already palm-normalized).
    """
    lm = _reshape_to_landmarks(normalized_vector)
    distances: dict[str, float] = {}
    palm = _palm_center(lm)

    # Thumb-index tip distance (critical for pinch detection)
    distances["thumb_index_tip_dist"] = float(
        np.linalg.norm(lm[THUMB_TIP] - lm[INDEX_TIP])
    )

    # Fingertip to palm center distances
    for i, tip_idx in enumerate(FINGERTIPS):
        distances[f"{FINGER_NAMES[i]}_to_palm"] = float(
            np.linalg.norm(lm[tip_idx] - palm)
        )

    # Adjacent fingertip spread distances
    for i in range(4):
        tip_a = FINGERTIPS[i]
        tip_b = FINGERTIPS[i + 1]
        distances[f"spread_{FINGER_NAMES[i]}_{FINGER_NAMES[i+1]}"] = float(
            np.linalg.norm(lm[tip_a] - lm[tip_b])
        )

    # Overall spread
    distances["spread_thumb_pinky"] = float(
        np.linalg.norm(lm[THUMB_TIP] - lm[PINKY_TIP])
    )

    return distances


def compute_heuristic_features(normalized_vector: np.ndarray) -> dict[str, float]:
    """Compute all heuristic features (angles + distances) for a single sample.

    Args:
        normalized_vector: Flat (60,) normalized landmark vector.

    Returns:
        Combined dictionary of all angle and distance features.
    """
    features = {}
    features.update(compute_angles(normalized_vector))
    features.update(compute_key_distances(normalized_vector))
    return features


def compute_heuristic_features_batch(
    X_normalized: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Compute heuristic features for a batch of samples.

    Args:
        X_normalized: Array of shape (n_samples, 60).

    Returns:
        Tuple of (feature_array, feature_names).
    """
    all_features = [compute_heuristic_features(x) for x in X_normalized]
    feature_names = list(all_features[0].keys())
    feature_array = np.array([[f[k] for k in feature_names] for f in all_features])
    return feature_array, feature_names


# ============================================================================
# Method 2: Random Forest Features (Pairwise Distances)
# ============================================================================

# Pairs for pairwise distance computation (21 pairs)
_PAIRWISE_PAIRS: list[tuple[int, int, str]] = []

# Fingertip-to-fingertip (10 pairs)
for i in range(5):
    for j in range(i + 1, 5):
        _PAIRWISE_PAIRS.append(
            (FINGERTIPS[i], FINGERTIPS[j],
             f"dist_{FINGER_NAMES[i]}_tip_{FINGER_NAMES[j]}_tip")
        )

# Fingertip-to-palm (5 pairs)
# These use palm center -- handled specially in the function

# Fingertip-to-wrist (5 pairs) -- wrist is at origin after normalization
# So this is just the norm of each fingertip

# MCP-to-TIP per finger (5 pairs -- finger extension proxy)
for i in range(5):
    _PAIRWISE_PAIRS.append(
        (MCPS[i], FINGERTIPS[i],
         f"dist_{FINGER_NAMES[i]}_mcp_to_tip")
    )

# Thumb tip to each other MCP (4 pairs -- thumb opposition)
for i in range(1, 5):
    _PAIRWISE_PAIRS.append(
        (THUMB_TIP, MCPS[i],
         f"dist_thumb_tip_{FINGER_NAMES[i]}_mcp")
    )


def compute_pairwise_distances(normalized_vector: np.ndarray) -> dict[str, float]:
    """Compute pairwise Euclidean distances for Random Forest features.

    Computes ~24 distances:
        - 10 fingertip-to-fingertip
        - 5 MCP-to-TIP per finger
        - 4 thumb tip to other MCPs
        - 5 fingertip to palm center

    All distances are already palm-size normalized from preprocessing.

    Args:
        normalized_vector: Flat (60,) normalized landmark vector.

    Returns:
        Dictionary mapping distance names to float values.
    """
    lm = _reshape_to_landmarks(normalized_vector)
    distances: dict[str, float] = {}

    for idx_a, idx_b, name in _PAIRWISE_PAIRS:
        distances[name] = float(np.linalg.norm(lm[idx_a] - lm[idx_b]))

    # Fingertip to palm center
    palm = _palm_center(lm)
    for i, tip_idx in enumerate(FINGERTIPS):
        distances[f"dist_{FINGER_NAMES[i]}_tip_to_palm"] = float(
            np.linalg.norm(lm[tip_idx] - palm)
        )

    return distances


def compute_pairwise_distances_batch(
    X_normalized: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Compute pairwise distance features for a batch of samples.

    Args:
        X_normalized: Array of shape (n_samples, 60).

    Returns:
        Tuple of (feature_array of shape (n_samples, n_features), feature_names).
    """
    all_features = [compute_pairwise_distances(x) for x in X_normalized]
    feature_names = list(all_features[0].keys())
    feature_array = np.array([[f[k] for k in feature_names] for f in all_features])
    return feature_array, feature_names


# ============================================================================
# Method 3: MLP Features
# ============================================================================


def compute_mlp_features(X_normalized: np.ndarray) -> np.ndarray:
    """Return the raw 60-dim normalized vectors for MLP input.

    The MLP uses the normalized landmark coordinates directly without
    additional feature engineering.

    Args:
        X_normalized: Array of shape (n_samples, 60).

    Returns:
        Same array, unchanged.
    """
    return X_normalized


if __name__ == "__main__":
    from preprocessing import generate_sample_csv, preprocess_dataset
    import os
    import tempfile

    # Generate sample data
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        tmp_path = f.name

    df = generate_sample_csv(tmp_path, n_persons=2, samples_per_gesture=5)
    X, y, pids = preprocess_dataset(df)
    os.unlink(tmp_path)

    print("=== Heuristic Features (Method 1) ===")
    h_features, h_names = compute_heuristic_features_batch(X)
    print(f"Shape: {h_features.shape}")
    print(f"Feature names ({len(h_names)}): {h_names}")
    print(f"Sample values: {h_features[0, :5]}")

    print("\n=== Pairwise Distance Features (Method 2) ===")
    d_features, d_names = compute_pairwise_distances_batch(X)
    print(f"Shape: {d_features.shape}")
    print(f"Feature names ({len(d_names)}): {d_names}")
    print(f"Sample values: {d_features[0, :5]}")

    print("\n=== MLP Features (Method 3) ===")
    mlp_features = compute_mlp_features(X)
    print(f"Shape: {mlp_features.shape}")
    print(f"Sample values: {mlp_features[0, :5]}")
