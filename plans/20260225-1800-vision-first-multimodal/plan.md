# Vision-First Multi-Modal Hand Gesture Recognition -- Implementation Plan

**Created:** 2026-02-25 | **Team:** 3 people | **Duration:** 10 days

## Goal
Transform from single-modality landmark classifier to vision-first multi-modal system: skeleton (MLP) + appearance (CNN) + detection (YOLO) + late fusion, with person-aware evaluation throughout.

## Architecture
```
Webcam -> MediaPipe -> Landmarks (60D) -> MLP -> Pose Logits ---|
  |                                                              |-> Late Fusion -> Final Prediction
  +-> Hand Crop (224x224) -> MobileNetV3-Small -> Appearance Logits -|

Separate pipeline: YOLO detection baseline (not fused; compared via derived classification accuracy)
```

## Phases

| # | Phase | Owner | Days | Status | File |
|---|-------|-------|------|--------|------|
| 01 | Data Pipeline | A+B | 1-2 | PENDING | [phase-01](./phase-01-data-pipeline.md) |
| 02 | YOLOv8n Training | A | 2-5 | PENDING | [phase-02](./phase-02-yolo-training.md) |
| 03 | CNN Training | B | 2-5 | PENDING | [phase-03](./phase-03-cnn-training.md) |
| 04 | Multi-Modal Fusion | B | 5-7 | PENDING | [phase-04](./phase-04-fusion.md) |
| 05 | Unified Evaluation | C | 3-8 | PENDING | [phase-05](./phase-05-evaluation.md) |
| 06 | Report & Deployment | C | 6-10 | PENDING | [phase-06](./phase-06-report-deployment.md) |

## Timeline (10-Day Gantt)

```
Day:   1    2    3    4    5    6    7    8    9    10
A:   [---Data---][----YOLO Training----][eval][report support]
B:   [---Data---][--CNN Train--][---Fusion---][eval][deploy]
C:   [eval framework][----eval scripts----|ablation][---report+slides---]
```

## Critical Path
1. HaGRID image download (Day 1 blocker -- ~15GB for 512px subset of 5 classes)
2. YOLO label conversion + CNN crop extraction depend on images
3. Fusion depends on trained CNN + existing MLP
4. Final evaluation depends on all trained models
5. Report depends on evaluation results

## Key Decisions
- **CNN backbone**: MobileNetV3-Small (2.5M params, real-time capable, ONNX-friendly)
- **Fusion strategy**: Separate ONNX models + JS weighted-average (simplest, preserves existing MLP export)
- **Evaluation**: Group 5-Fold CV with person-aware splits for all methods
- **YOLO comparison**: Derived classification accuracy from matched detections (IoU >= 0.5)

## Research References
- [Research 01: YOLO, HaGRID, Fusion, CNN](./research/researcher-01-yolo-fusion-cnn.md)
- [Research 02: Evaluation, Deployment, Inter-Hand](./research/researcher-02-evaluation-deployment.md)
- [Codebase Scout](./scout/scout-01-codebase.md)
