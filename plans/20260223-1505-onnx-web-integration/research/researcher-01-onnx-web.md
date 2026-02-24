# Research Report: ONNX Runtime Web for Browser Inference

## Executive Summary
This report covers how to integrate `onnxruntime-web` directly into an HTML file via CDN to run a small 60-dimensional input MLP model in the browser using JavaScript. The process involves including the ORT script, setting up the WebAssembly (WASM) paths, preparing input data as `Float32Array` objects within an `ort.Tensor`, and executing inference sessions using WebGPU or WASM.

## Research Methodology
- **Sources consulted**: Official Microsoft ONNX Runtime documentation, npm package details, and ONNX Runtime JS API docs.
- **Key Search Terms**: "onnxruntime-web cdn tutorial", "ort.Tensor from javascript array", "onnxruntime web inferencing javascript".

## Key Findings

### 1. CDN Import
To run ONNX models directly in the browser without a build system, import `onnxruntime-web` via a CDN inside your HTML file:
```html
<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js"></script>
```

### 2. Loading the Model and Running Inference
When running from a CDN, setting the WASM paths explicitly is critical since the WASM files are loaded asynchronously and must be fetched from the same CDN directory.

#### Basic Inference Skeleton:
```javascript
// Crucial for CDN usage: Set WASM path explicitly
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';

async function predict(inputArray) {
    try {
        // Load the ONNX model (fallback to 'wasm' if 'webgpu' isn't supported)
        const session = await ort.InferenceSession.create('./model.onnx', {
            executionProviders: ['webgpu', 'wasm']
        });

        // ... tensor creation and inference execution ...
    } catch (e) {
        console.error("Inference failed", e);
    }
}
```

### 3. Converting JS Arrays to `ort.Tensor` and Parsing Outputs
For a 60-dimensional input MLP model, the standard JS array needs to be converted to a `Float32Array` before being wrapped in an `ort.Tensor`.

#### Tensor Creation & Execution Example:
```javascript
// 1. Convert standard JS array (length 60) to Float32Array
const floatData = Float32Array.from(inputArray); // inputArray has 60 floats

// 2. Create the ONNX Tensor with shape [batch_size, input_features] = [1, 60]
const inputTensor = new ort.Tensor('float32', floatData, [1, 60]);

// 3. Prepare feeds dictionary.
// NOTE: 'input' must match the EXACT input node name defined in your ONNX model.
const feeds = { 'input': inputTensor }; // Replace 'input' with your model's input name

// 4. Run the session
const results = await session.run(feeds);

// 5. Parse probabilities
// NOTE: 'output' must match your ONNX model's exact output node name.
const outputData = results['output'].data;
console.log("Output Probabilities:", outputData); // Float32Array of probabilities
```

## Implementation Recommendations

### Quick Start Guide for 60-Dim MLP
1. **Host `model.onnx`**: Ensure your exported `.onnx` model is hosted alongside your HTML/JS files or served via a local web server (CORS policies block fetching `file://` URIs).
2. **Determine Node Names**: Before deploying, inspect your `.onnx` file (e.g., using Netron at https://netron.app/) to find the exact names of your input and output nodes. If your input node is named `dense_input` and output is `dense_output`, use `const feeds = { 'dense_input': inputTensor };` and `results['dense_output'].data`.
3. **Array Dimensions**: Ensure the `[1, 60]` shape strictly aligns with the expected input shape of your MLP model exported from PyTorch/TensorFlow.

### Common Pitfalls
- **WASM Loading Errors**: Forgetting to set `ort.env.wasm.wasmPaths` will result in failed fetches for `.wasm` files.
- **Node Name Mismatches**: Using generic keys like `input` or `output` in the `feeds` dictionary or `results` object instead of the actual model node names.
- **CORS Issues**: Opening the HTML file directly in the browser via `file://` protocol will block the `InferenceSession.create()` fetch request. Always serve via a local HTTP server (e.g., `python -m http.server`).

## Resources
- [ONNX Web Tutorials](https://onnxruntime.ai/docs/tutorials/web/)
- [ONNX Runtime JS API Reference](https://onnxruntime.ai/docs/api/js/)
- [Netron Model Viewer](https://netron.app/) (For inspecting node names)