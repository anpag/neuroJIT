# TensorLang: LLM Integration Roadmap

## 1. Overview
To achieve true "Self-Rewriting" capability, TensorLang requires an embedded Large Language Model (LLM) that runs within the compiler's process space. This document outlines the plan to integrate `llama.cpp` as the inference engine.

## 2. Architecture

### 2.1 The `ModelRunner` Interface
We have already defined an abstract `ModelRunner` class in `tensorlang/Runtime/ModelRunner.h`.
Currently, a `MockModelRunner` is used for testing. We will implement `LlamaCppModelRunner`.

### 2.2 Integration Strategy
We will link against `libllama` (from `llama.cpp`) dynamically or statically.
The `LlamaCppModelRunner` will manage the `llama_context` and `llama_model` objects.

## 3. Implementation Plan

### Step 1: Add Dependency
1.  Add `llama.cpp` as a git submodule in `deps/`.
2.  Build `llama.cpp` as a library (`libllama.so` or `libllama.a`).
3.  Update `tensorlang/runtime/CMakeLists.txt` to link against `libllama`.

### Step 2: Implement `LlamaCppModelRunner`
Create `tensorlang/runtime/LlamaCppModelRunner.cpp`:

```cpp
class LlamaCppModelRunner : public ModelRunner {
public:
  int load(const std::string& modelPath) override {
    // 1. Initialize backend
    llama_backend_init(false);
    
    // 2. Load model
    auto model_params = llama_model_default_params();
    model = llama_load_model_from_file(modelPath.c_str(), model_params);
    
    // 3. Create context
    auto ctx_params = llama_context_default_params();
    ctx = llama_new_context_with_model(model, ctx_params);
    return 0;
  }

  std::string query(const std::string& prompt) override {
    // 1. Tokenize prompt
    std::vector<llama_token> tokens = tokenize(prompt);
    
    // 2. Inference Loop
    // Evaluate tokens and sample new tokens until EOS or limit.
    
    // 3. Detokenize output
    return detokenize(generated_tokens);
  }

private:
  llama_model* model = nullptr;
  llama_context* ctx = nullptr;
};
```

### Step 3: Runtime Configuration
Update `tensorlang-run` to accept a `--model-path` argument and initialize the correct runner.

## 4. Performance Considerations
*   **Quantization:** Use 4-bit quantized models (GGUF format) for low memory footprint.
*   **GPU Offloading:** Ensure `llama.cpp` is built with CUDA/ROCm support if available to accelerate inference.
*   **Context Caching:** Reuse the KV cache for subsequent queries to speed up iterative optimization.

## 5. Safety
*   **Sandboxing:** The LLM's generated code must be verified before execution.
*   **Timeouts:** Inference must be bounded to prevent compiler hangs.
