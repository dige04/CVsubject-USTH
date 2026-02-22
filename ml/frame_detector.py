"""Two-stage frame gesture detection using two-hand spatial analysis.

The Frame gesture requires two hands forming an L-shape that together
create a rectangular "camera frame". This module checks spatial
relationships between two detected hands.
"""

from __future__ import annotations

import numpy as np

from features import (
    _reshape_to_landmarks,
    THUMB_TIP,
    THUMB_MCP,
    INDEX_TIP,
    INDEX_MCP,
    MIDDLE_TIP,
    RING_TIP,
    PINKY_TIP,
    FINGER_NAMES,
    FINGERTIPS,
    compute_angles,
)


class FrameDetector:
    """Detect the Frame gesture from two-hand landmarks.

    The frame gesture is formed by two hands making L-shapes with thumbs
    pointing toward each other and index fingers roughly parallel, creating
    a rectangular viewfinder shape.

    Detection criteria:
        1. Both hands show an L-shape (thumb + index extended, others curled)
        2. Thumbs point toward each other
        3. Index fingers are roughly parallel
        4. Bounding box aspect ratio is between 0.5 and 2.0
    """

    def __init__(
        self,
        curl_extended_threshold: float = 130.0,
        curl_curled_threshold: float = 110.0,
        thumb_alignment_threshold: float = 45.0,
        index_parallel_threshold: float = 30.0,
        aspect_ratio_min: float = 0.5,
        aspect_ratio_max: float = 2.0,
    ) -> None:
        """Initialize with tunable thresholds.

        Args:
            curl_extended_threshold: Angle above which a finger is extended.
            curl_curled_threshold: Angle below which a finger is curled.
            thumb_alignment_threshold: Max angle (degrees) between thumb
                directions for them to be considered pointing at each other.
            index_parallel_threshold: Max angle (degrees) between index
                finger directions for them to be considered parallel.
            aspect_ratio_min: Minimum acceptable aspect ratio for the frame.
            aspect_ratio_max: Maximum acceptable aspect ratio for the frame.
        """
        self.curl_extended_threshold = curl_extended_threshold
        self.curl_curled_threshold = curl_curled_threshold
        self.thumb_alignment_threshold = thumb_alignment_threshold
        self.index_parallel_threshold = index_parallel_threshold
        self.aspect_ratio_min = aspect_ratio_min
        self.aspect_ratio_max = aspect_ratio_max

    def _is_l_shape(self, normalized_vector: np.ndarray) -> tuple[bool, float]:
        """Check if a single hand forms an L-shape.

        L-shape: thumb and index extended, middle/ring/pinky curled.

        Args:
            normalized_vector: Flat (60,) normalized landmark vector.

        Returns:
            Tuple of (is_l_shape, confidence).
        """
        angles = compute_angles(normalized_vector)

        thumb_curl = angles["thumb_curl_angle"]
        index_curl = angles["index_curl_angle"]

        thumb_extended = thumb_curl > self.curl_extended_threshold
        index_extended = index_curl > self.curl_extended_threshold

        others_curled = 0
        for name in ["middle", "ring", "pinky"]:
            if angles[f"{name}_curl_angle"] < self.curl_curled_threshold:
                others_curled += 1

        is_l = thumb_extended and index_extended and others_curled >= 2

        # Compute confidence
        confidence = 0.0
        if is_l:
            confidence = 0.5
            confidence += 0.1 * (others_curled / 3.0)
            # Bonus for clear extension
            if thumb_curl > 150:
                confidence += 0.1
            if index_curl > 150:
                confidence += 0.1
            confidence = min(1.0, confidence + 0.2)

        return is_l, confidence

    def _get_finger_direction(
        self, normalized_vector: np.ndarray, base_idx: int, tip_idx: int
    ) -> np.ndarray:
        """Get the direction vector from finger base to tip.

        Args:
            normalized_vector: Flat (60,) normalized landmark vector.
            base_idx: Index of the base landmark (in 20-landmark space).
            tip_idx: Index of the tip landmark.

        Returns:
            Normalized 3D direction vector.
        """
        lm = _reshape_to_landmarks(normalized_vector)
        direction = lm[tip_idx] - lm[base_idx]
        norm = np.linalg.norm(direction)
        if norm < 1e-8:
            return np.zeros(3)
        return direction / norm

    def detect(
        self,
        hand1_normalized: np.ndarray,
        hand2_normalized: np.ndarray,
    ) -> tuple[bool, float]:
        """Detect if two hands form a frame gesture.

        Args:
            hand1_normalized: Flat (60,) normalized landmarks for hand 1.
            hand2_normalized: Flat (60,) normalized landmarks for hand 2.

        Returns:
            Tuple of (is_frame, confidence).
                is_frame: True if the frame gesture is detected.
                confidence: Float in [0, 1] indicating detection confidence.
        """
        # Stage 1: Check if both hands show L-shape
        h1_is_l, h1_conf = self._is_l_shape(hand1_normalized)
        h2_is_l, h2_conf = self._is_l_shape(hand2_normalized)

        if not (h1_is_l and h2_is_l):
            return False, 0.0

        base_confidence = (h1_conf + h2_conf) / 2.0

        # Stage 2: Check spatial relationships
        # Get thumb directions for both hands
        thumb_dir1 = self._get_finger_direction(hand1_normalized, THUMB_MCP, THUMB_TIP)
        thumb_dir2 = self._get_finger_direction(hand2_normalized, THUMB_MCP, THUMB_TIP)

        # Thumbs should point toward each other (roughly opposite directions)
        # The dot product of opposing directions should be negative
        thumb_dot = np.dot(thumb_dir1, thumb_dir2)
        thumb_angle = np.degrees(np.arccos(np.clip(abs(thumb_dot), 0, 1)))

        # For opposing thumbs, the angle between them should be close to 180
        # or equivalently, the angle between one and the negative of the other
        # should be small
        thumb_opposition_angle = np.degrees(
            np.arccos(np.clip(-thumb_dot, -1, 1))
        )
        thumbs_opposing = thumb_opposition_angle < self.thumb_alignment_threshold

        # Get index finger directions
        index_dir1 = self._get_finger_direction(hand1_normalized, INDEX_MCP, INDEX_TIP)
        index_dir2 = self._get_finger_direction(hand2_normalized, INDEX_MCP, INDEX_TIP)

        # Index fingers should be roughly parallel (same or opposite direction)
        index_dot = abs(np.dot(index_dir1, index_dir2))
        index_angle = np.degrees(np.arccos(np.clip(index_dot, 0, 1)))
        indices_parallel = index_angle < self.index_parallel_threshold

        # Check bounding box aspect ratio using fingertip positions
        lm1 = _reshape_to_landmarks(hand1_normalized)
        lm2 = _reshape_to_landmarks(hand2_normalized)

        # Combine key points from both hands
        key_points = np.vstack([
            lm1[THUMB_TIP], lm1[INDEX_TIP],
            lm2[THUMB_TIP], lm2[INDEX_TIP],
        ])

        # Compute 2D bounding box (using x, y only)
        x_range = key_points[:, 0].max() - key_points[:, 0].min()
        y_range = key_points[:, 1].max() - key_points[:, 1].min()

        if x_range < 1e-8 or y_range < 1e-8:
            return False, 0.0

        aspect_ratio = x_range / y_range
        good_aspect = self.aspect_ratio_min <= aspect_ratio <= self.aspect_ratio_max

        # Combine checks
        spatial_checks = sum([thumbs_opposing, indices_parallel, good_aspect])

        if spatial_checks < 2:
            return False, 0.0

        # Compute final confidence
        spatial_confidence = spatial_checks / 3.0
        final_confidence = base_confidence * 0.4 + spatial_confidence * 0.6

        is_frame = final_confidence > 0.4
        return is_frame, final_confidence

    def detect_batch(
        self,
        hands1_normalized: np.ndarray,
        hands2_normalized: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Detect frame gesture for batches of hand pairs.

        Args:
            hands1_normalized: Array of shape (n_samples, 60) for hand 1.
            hands2_normalized: Array of shape (n_samples, 60) for hand 2.

        Returns:
            Tuple of (detections, confidences).
                detections: Boolean array of shape (n_samples,).
                confidences: Float array of shape (n_samples,).
        """
        n = len(hands1_normalized)
        detections = np.zeros(n, dtype=bool)
        confidences = np.zeros(n, dtype=float)

        for i in range(n):
            is_frame, conf = self.detect(hands1_normalized[i], hands2_normalized[i])
            detections[i] = is_frame
            confidences[i] = conf

        return detections, confidences


if __name__ == "__main__":
    from preprocessing import generate_sample_csv, preprocess_dataset
    import tempfile
    import os

    # Generate sample data
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        tmp_path = f.name

    df = generate_sample_csv(tmp_path, n_persons=2, samples_per_gesture=10)
    X, y, _ = preprocess_dataset(df)
    os.unlink(tmp_path)

    detector = FrameDetector()

    # Test with pairs of "frame" labeled samples
    frame_mask = y == "frame"
    frame_samples = X[frame_mask]

    print("=== Frame Detection Tests ===")
    if len(frame_samples) >= 2:
        is_frame, conf = detector.detect(frame_samples[0], frame_samples[1])
        print(f"Frame pair (frame+frame): detected={is_frame}, confidence={conf:.3f}")

    # Test with non-frame pair
    open_mask = y == "open_hand"
    open_samples = X[open_mask]
    if len(open_samples) >= 2:
        is_frame, conf = detector.detect(open_samples[0], open_samples[1])
        print(f"Non-frame pair (open+open): detected={is_frame}, confidence={conf:.3f}")

    # Test with mixed pair
    fist_mask = y == "fist"
    fist_samples = X[fist_mask]
    if len(fist_samples) >= 1 and len(open_samples) >= 1:
        is_frame, conf = detector.detect(fist_samples[0], open_samples[0])
        print(f"Mixed pair (fist+open): detected={is_frame}, confidence={conf:.3f}")

    print("\nFrame detector initialized successfully.")
