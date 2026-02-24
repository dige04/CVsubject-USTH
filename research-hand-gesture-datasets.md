# Hand Gesture MediaPipe Landmark Datasets -- Research Report

**Date:** 2026-02-22
**Goal:** Find pre-extracted MediaPipe 21-point hand landmark datasets covering Open Hand, Fist, Pinch, and Frame gestures.

---

## VERDICT UPFRONT

**No single dataset provides pre-extracted MediaPipe landmark CSV coordinates for all four target gestures (Open Hand, Fist, Pinch, Frame).** The best path forward is to use the **HaGRID v2** dataset (which has images + JSON annotations with MediaPipe landmarks for all four gestures) and extract/convert the landmark coordinates yourself. Below is the full breakdown.

---

## 1. HaGRID v2 (Original Dataset) -- BEST OPTION

- **URL:** https://github.com/hukenovs/hagrid
- **Paper:** https://arxiv.org/abs/2206.11438
- **Size:** 1,086,158 FullHD RGB images, ~716 GB total
- **Format:** Images (JPEG) + JSON annotation files with bounding boxes and MediaPipe landmarks
- **License:** CC BY-NC-SA 4.0 (Non-Commercial)
- **Classes (33 + no_gesture):** call, dislike, **fist**, four, grabbing, grip, hand_heart, hand_heart2, holy, like, little_finger, middle_finger, mute, ok, one, **palm**, peace, peace_inverted, point, rock, **stop**, stop_inverted, **take_picture**, three, three2, three3, three_gun, **thumb_index**, thumb_index2, timeout, two_up, two_up_inverted, xsign, no_gesture

### Mapping to Your Target Gestures

| Your Gesture | HaGRID Class | Notes |
|---|---|---|
| Open Hand | `stop` or `palm` | `stop` = upright open hand; `palm` = flat toward camera |
| Fist | `fist` | Direct match |
| Pinch | `thumb_index` | Thumb + index finger touching/close. Also `thumb_index2` variant |
| Frame | `take_picture` | Two-hand rectangular frame gesture |

### Key Details
- JSON annotations already contain **MediaPipe landmark coordinates** (auto-annotated) for each hand
- Landmarks are stored as x,y coordinate pairs in annotation JSON, NOT as separate CSV
- You can download per-class subsets (not the full 716 GB)
- Has a `demo.py` script with `--landmarks` flag to visualize landmarks
- **Download:** Via git-lfs from GitHub or via Kaggle (https://www.kaggle.com/datasets/innominate8/hagrid)

### What You Need To Do
- Parse the JSON annotation files to extract the 21-landmark x,y coordinates
- Convert to your desired CSV format (trivial Python script)
- OR use their included scripts to process

---

## 2. HaGRID-MediaPipe-Hands (HuggingFace) -- NOT USEFUL FOR YOUR CASE

- **URL:** https://huggingface.co/datasets/Vincent-luo/hagrid-mediapipe-hands
- **Author:** Vincent-luo
- **Size:** 507,050 samples, ~112 GB (Parquet)
- **Format:** Parquet with columns: `image`, `conditioning_image` (rendered landmark skeleton image), `text`, `classes`
- **License:** Inherits CC BY-NC from HaGRID (verify)

### WHY NOT USEFUL
- **Does NOT contain raw x,y,z landmark coordinates.** Only contains rendered skeleton images (black background with drawn landmark connections).
- Built for ControlNet training (image-to-image), not gesture classification.
- The `text` column is uniformly "a photo of a hand" -- no gesture labels preserved.
- Gesture class labels exist in the underlying data but the dataset viewer shows they may not be consistently exposed.

**Skip this one.** Go straight to HaGRID v2 original.

---

## 3. ASL Gesture Dataset Using MediaPipe (Kaggle) -- USEFUL AS REFERENCE FORMAT

- **URL:** https://www.kaggle.com/datasets/jaisuryaprabu/asl-gesture-dataset-using-media-pipe
- **Format:** CSV with 63 features (21 landmarks x 3 coordinates: x, y, z)
- **Classes:** A-Z alphabet letters (26 classes)
- **Covers your gestures?** NO. ASL alphabet only. No fist, pinch, or frame.
- **Why useful:** Shows the standard CSV format for MediaPipe landmarks that you should target

---

## 4. ASL Hand Skeleton Dataset (Kaggle)

- **URL:** https://www.kaggle.com/datasets/mohamedelkassaby/american-sign-language-asl-hand-skeleton-dataset
- **Size:** 26,000 skeleton images + CSV landmark data
- **Classes:** A-Z (1,000 samples per letter)
- **License:** CC0 (Public Domain)
- **Covers your gestures?** NO. ASL alphabet only.

---

## 5. Other Notable Datasets

### Pointing Gesture Classification Dataset (Zenodo, July 2025)
- **URL:** https://zenodo.org/records/12803874
- **Format:** CSV, 13,575 instances
- **Special:** Uses Euclidean distances between landmarks (not raw coordinates)
- **Covers your gestures?** NO. Pointing gestures only.

### Number Gestures 1-5 (Kaggle)
- **URL:** https://www.kaggle.com/datasets/marnon/number-gestures-1-5-hand-landmark-dataset
- **Format:** Raw x,y coordinates for 21 MediaPipe landmarks
- **Covers your gestures?** Partially. "1" could loosely map to a fist variant, but not a direct match.

### Google ASL Signs Competition (Kaggle)
- **URL:** https://www.kaggle.com/competitions/asl-signs/data
- **Format:** Parquet (face + hand + pose landmarks)
- **Covers your gestures?** NO. ASL signs only. Massive dataset though.

---

## 6. Batch Extraction Tools (MediaPipe Images-to-CSV)

Since no ready-made dataset perfectly fits your four gestures with CSV landmarks, you will likely need to extract landmarks yourself. Here are known tools:

### a) HaGRID Built-In
- The HaGRID v2 repo (https://github.com/hukenovs/hagrid) already has **pre-computed MediaPipe landmarks in JSON annotations**. You just need a simple script to parse JSON -> CSV.

### b) Sign-Language-Recognition-System (GitHub)
- **URL:** https://github.com/JaspreetSingh-exe/Sign-Language-Recognition-System
- Contains scripts to generate `keypoints.csv` from video feeds using MediaPipe
- Can be adapted for batch image processing

### c) HandGesture2Emoji (GitHub)
- **URL:** https://github.com/Shadowfax221/HandGesture2Emoji
- Includes a pre-processed `HandLandmarks.csv` and the extraction pipeline

### d) MediaPipe Official Python API (DIY)
- **URL:** https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python
- The official Hand Landmarker v2 API (updated Jan 2025) is the canonical way to batch-extract
- Pattern: load image -> run HandLandmarker -> extract 21 landmarks (x,y,z) -> write CSV row
- Straightforward: ~30 lines of Python to process an entire image folder

---

## RECOMMENDED APPROACH

1. **Download HaGRID v2** -- only the classes you need:
   - `fist` (direct match for Fist)
   - `stop` (direct match for Open Hand)
   - `thumb_index` (direct match for Pinch)
   - `take_picture` (direct match for Frame -- note: two-hand gesture)

2. **Extract landmarks from JSON annotations** -- HaGRID already has MediaPipe landmarks auto-annotated in the JSON. Write a simple parser to convert JSON landmark arrays to CSV rows with format: `class, x0, y0, z0, x1, y1, z1, ..., x20, y20, z20`

3. **If JSON landmarks are insufficient** (e.g., need z-coordinates or higher precision), run the HaGRID images through MediaPipe Hand Landmarker v2 yourself. The official Python API makes this trivial.

4. **For the Frame gesture specifically:** This is a two-hand gesture. You will need TWO sets of 21 landmarks (42 landmarks total) per sample, or a different representation. Plan your data schema accordingly.

---

## SUMMARY TABLE

| Dataset | Format | Has Fist? | Has Open Hand? | Has Pinch? | Has Frame? | Pre-extracted CSV? |
|---|---|---|---|---|---|---|
| HaGRID v2 | Images + JSON landmarks | YES | YES (stop/palm) | YES (thumb_index) | YES (take_picture) | NO (JSON, easy to convert) |
| HaGRID-MediaPipe-Hands (HF) | Parquet (images only) | Has class but no coords | Has class but no coords | Has class but no coords | Has class but no coords | NO |
| ASL Gesture (Kaggle) | CSV landmarks | NO | NO | NO | NO | YES but wrong classes |
| ASL Skeleton (Kaggle) | Images + CSV | NO | NO | NO | NO | YES but wrong classes |
| Zenodo Pointing | CSV distances | NO | NO | NO | NO | YES but wrong classes |

---

## UNRESOLVED QUESTIONS

1. **HaGRID JSON landmark format:** Do the annotations include z-coordinates or only x,y? Need to inspect actual JSON files to confirm.
2. **take_picture (Frame) sample count:** How many samples exist for this specific class? HaGRID v2 has ~33K per class on average but two-hand gestures may have fewer.
3. **HaGRID per-class download:** Can you download individual classes without the full 716 GB? The GitHub README suggests yes (per-class archives) but exact URLs need verification.
4. **Licensing for academic use:** HaGRID is CC BY-NC-SA 4.0. If this is for a commercial product, you cannot use it directly.
