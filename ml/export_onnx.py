"""Build ONNX model from .keras file without loading TensorFlow.

The .keras file is a ZIP archive containing:
  - config.json (model architecture)
  - model.weights.h5 (HDF5 weights)
"""
import json
import os
import sys
import zipfile

import h5py
import numpy as np
import onnx
from onnx import TensorProto, helper


def extract_weights_from_keras(keras_path: str):
    """Extract Dense layer weights from .keras ZIP file."""
    weights = []

    with zipfile.ZipFile(keras_path, "r") as zf:
        # Read config
        config = json.loads(zf.read("config.json"))
        print(f"Model type: {config['class_name']}")

        # Extract weights h5 to temp
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp.write(zf.read("model.weights.h5"))
            tmp_path = tmp.name

    # Read weights from HDF5
    with h5py.File(tmp_path, "r") as f:
        # Navigate HDF5 structure to find Dense layer weights
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  HDF5 dataset: {name} shape={obj.shape}")

        f.visititems(visit)

        # Dense layers in Sequential model are stored under their layer names
        # Find all kernel/bias pairs
        dense_weights = []

        def find_dense_weights(name, obj):
            if isinstance(obj, h5py.Dataset) and name.endswith("/0"):
                # This is a kernel or bias
                parent = name.rsplit("/", 1)[0]
                dense_weights.append((name, obj[:]))

        f.visititems(find_dense_weights)

    os.unlink(tmp_path)
    return dense_weights


def build_onnx_from_weights(keras_path: str, onnx_path: str):
    """Build ONNX model from Keras weights."""
    print(f"Reading weights from: {keras_path}")

    with zipfile.ZipFile(keras_path, "r") as zf:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp.write(zf.read("model.weights.h5"))
            tmp_path = tmp.name

    # Read all datasets from HDF5
    all_arrays = []
    with h5py.File(tmp_path, "r") as f:
        def collect(name, obj):
            if isinstance(obj, h5py.Dataset) and obj.ndim > 0:
                arr = np.array(obj)
                all_arrays.append((name, arr))
                print(f"  {name}: shape={arr.shape}, dtype={arr.dtype}")
        f.visititems(collect)

    os.unlink(tmp_path)

    # Filter to only layer weights (skip optimizer state)
    layer_arrays = [(n, a) for n, a in all_arrays if n.startswith("layers/")]
    kernels = [(n, a) for n, a in layer_arrays if a.ndim == 2]
    biases = [(n, a) for n, a in layer_arrays if a.ndim == 1]

    assert len(kernels) == 3, f"Expected 3 kernels, got {len(kernels)}"
    assert len(biases) == 3, f"Expected 3 biases, got {len(biases)}"

    print(f"\nLayer architecture:")
    for i, ((kn, k), (bn, b)) in enumerate(zip(kernels, biases)):
        act = "relu" if i < 2 else "softmax"
        print(f"  Dense {i}: ({k.shape[0]}, {k.shape[1]}) + ({b.shape[0]},) -> {act}")

    W0, b0 = kernels[0][1].astype(np.float32), biases[0][1].astype(np.float32)
    W1, b1 = kernels[1][1].astype(np.float32), biases[1][1].astype(np.float32)
    W2, b2 = kernels[2][1].astype(np.float32), biases[2][1].astype(np.float32)

    # Verify dimensions
    assert W0.shape == (60, 128), f"W0 shape mismatch: {W0.shape}"
    assert W1.shape == (128, 64), f"W1 shape mismatch: {W1.shape}"
    assert W2.shape == (64, 5), f"W2 shape mismatch: {W2.shape}"

    # Build ONNX graph
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, 60])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, 5])

    nodes = []
    initializers = []

    # Layer 0: Dense(128, relu)
    initializers.append(helper.make_tensor("W0", TensorProto.FLOAT, W0.shape, W0.flatten()))
    initializers.append(helper.make_tensor("b0", TensorProto.FLOAT, b0.shape, b0.flatten()))
    nodes.append(helper.make_node("MatMul", ["input", "W0"], ["mm0"]))
    nodes.append(helper.make_node("Add", ["mm0", "b0"], ["add0"]))
    nodes.append(helper.make_node("Relu", ["add0"], ["relu0"]))

    # Layer 1: Dense(64, relu)
    initializers.append(helper.make_tensor("W1", TensorProto.FLOAT, W1.shape, W1.flatten()))
    initializers.append(helper.make_tensor("b1", TensorProto.FLOAT, b1.shape, b1.flatten()))
    nodes.append(helper.make_node("MatMul", ["relu0", "W1"], ["mm1"]))
    nodes.append(helper.make_node("Add", ["mm1", "b1"], ["add1"]))
    nodes.append(helper.make_node("Relu", ["add1"], ["relu1"]))

    # Layer 2: Dense(5, softmax)
    initializers.append(helper.make_tensor("W2", TensorProto.FLOAT, W2.shape, W2.flatten()))
    initializers.append(helper.make_tensor("b2", TensorProto.FLOAT, b2.shape, b2.flatten()))
    nodes.append(helper.make_node("MatMul", ["relu1", "W2"], ["mm2"]))
    nodes.append(helper.make_node("Add", ["mm2", "b2"], ["logits"]))
    nodes.append(helper.make_node("Softmax", ["logits"], ["output"], axis=1))

    graph = helper.make_graph(nodes, "mlp_gesture", [X], [Y], initializer=initializers)
    model_proto = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    model_proto.ir_version = 7

    # Validate
    onnx.checker.check_model(model_proto)

    # Save
    os.makedirs(os.path.dirname(onnx_path) or ".", exist_ok=True)
    onnx.save(model_proto, onnx_path)
    print(f"\nONNX model saved: {onnx_path}")
    print(f"File size: {os.path.getsize(onnx_path)} bytes")
    print(f"Input: 'input' shape=[batch, 60]")
    print(f"Output: 'output' shape=[batch, 5]")

    # Verify with onnxruntime
    import onnxruntime as rt
    sess = rt.InferenceSession(onnx_path)
    dummy = np.random.randn(1, 60).astype(np.float32)
    result = sess.run(["output"], {"input": dummy})[0]
    print(f"\nTest inference output: {result[0]}")
    print(f"Sum of probabilities: {result[0].sum():.6f} (should be ~1.0)")
    assert abs(result[0].sum() - 1.0) < 0.01, "Softmax output doesn't sum to 1!"
    print("Verification PASSED")


if __name__ == "__main__":
    keras_path = os.path.join("..", "models", "mlp_model", "saved_model.keras")
    onnx_path = os.path.join("..", "game", "mlp_model.onnx")
    build_onnx_from_weights(keras_path, onnx_path)
