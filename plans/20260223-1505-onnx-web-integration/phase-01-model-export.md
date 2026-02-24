# Phase 1: Model Export & Setup

## Summary
The goal of this phase is to ensure the trained Multi-Layer Perceptron (MLP) model is exported to the ONNX format (`mlp_model.onnx`) and made available to the web application.

## Objective
Convert the best-performing model (MLP) from the Python training pipeline into an ONNX format that can be consumed by the ONNX Web runtime, and ensure it is located in the appropriate directory for the frontend application to access.

## Tasks

### 1.1 Export the MLP Model to ONNX

**Path**: `/Users/hieudinh/Documents/my-projects/CVsubject/ml/train.py` (or the specific file where the MLP model is trained/saved)
**Function/Class**: Where the best model is finalized.

**Action**: Add or verify the code to export the trained PyTorch/scikit-learn MLP model to the `.onnx` format. If it's a PyTorch model, use `torch.onnx.export`. If scikit-learn, use `skl2onnx`. Assume PyTorch for this example, but adapt as necessary.

**Snippet**:
```python
import torch

# Assuming `model` is the trained PyTorch MLP model
# Assuming the input shape is (1, 60) for a single hand's flattened landmarks

dummy_input = torch.randn(1, 60, dtype=torch.float32)
torch.onnx.export(
    model,
    dummy_input,
    "mlp_model.onnx",
    export_params=True,
    opset_version=14, # Ensure compatibility with ONNX Web
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)
print("Model exported to mlp_model.onnx")
```

**Justification**: The ONNX Web runtime requires models in the `.onnx` format. Exporting it with defined input/output names simplifies the JavaScript inference code.

**Risks**: Ensure the `input_names` and `output_names` exactly match what will be expected in the JavaScript side (`session.run({'input': tensor})`). Verify the opset version is supported by the ONNX Web version being used.

### 1.2 Copy Model to Web Asset Directory

**Path**: Bash/CLI execution.

**Action**: Move or copy the exported `mlp_model.onnx` file to the directory served by the frontend application (e.g., `game/`, `public/`, or `assets/`).

**Command snippet**:
```bash
# Example assuming the frontend is served from a 'game' directory
cp mlp_model.onnx ./game/
```

**Justification**: The frontend application needs to fetch the `.onnx` file via an HTTP request when initializing the `ort.InferenceSession`. It must be located in a publicly accessible path.

**Risks**: Path mismatch. The frontend code might attempt to load `./mlp_model.onnx`, but the file is located elsewhere. Ensure the path is correct relative to where `index.html` is served.

## Acceptance Criteria
- [ ] The `mlp_model.onnx` file exists.
- [ ] The file is located in the correct directory (e.g., `game/`) to be served to the frontend.
- [ ] The model's input name is confirmed to be `input` and output name is `output` (or documented otherwise).