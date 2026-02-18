# TensorLang: The Core

This directory contains the implementation of the TensorLang compiler infrastructure.

## Directory Structure

*   **`Dialect/`**: The heart of the language.
    *   Defines the `TensorLang` MLIR dialect.
    *   Contains the TableGen definitions (`.td`) for Types (`LinearTensor`) and Operations (`matmul`, `symbolic_dim`).
    *   Implements verifiers and semantic checks.

*   **`Conversion/`**: The "Translation Layer".
    *   Contains passes that lower TensorLang IR to other MLIR dialects (primarily `Linalg` and `Arith`).
    *   `TensorLangToLinalg`: Defines how high-level tensor operations map to loops and linear algebra primitives.

*   **`ExecutionEngine/`**: The "Engine".
    *   Implements the JIT (Just-In-Time) runner using LLVM ORC.
    *   Handles the compilation pipeline (MLIR -> LLVM IR -> Machine Code).
    *   Manages hot-swapping of functions during runtime.

*   **`Runtime/`**: The "Brain Interface".
    *   C++ hooks that the running JIT code calls into.
    *   `GeminiModelRunner.cpp`: The bridge that sends IR to Google Gemini and receives optimized code.
    *   `Runtime.cpp`: Handles assertions, self-healing triggers, and I/O.

*   **`examples/`**:
    *   `neuro_lander.mlir`: The lunar lander simulation used to demonstrate self-healing.
    *   `self_rewrite_demo.mlir`: A basic test of the reflection capabilities.

*   **`benchmarks/`**:
    *   `self_optimizing_matmul.mlir`: The matrix multiplication benchmark used to demonstrate autonomous optimization.
