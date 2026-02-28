# TensorLang: Progress Report & Roadmap
**Date:** February 8, 2026
**Status:** Building Infrastructure (LLVM/MLIR 19.x)

## 1. Project Overview
**Objective:** Build a "Tensor-Native" programming language optimized for AI/LLM workloads.
**Key Features:**
*   **First-Class Tensors:** N-dimensional arrays as primitive types.
*   **Linear Types:** Deterministic memory management (no GC) for high performance.
*   **Intrinsic Differentiation:** Enzyme-style AD on the IR.
*   **Self-Rewriting:** The ability for the language to inspect its own code (IR) and modify it at runtime (JIT) using an embedded LLM.

## 2. Infrastructure Status
We are currently setting up a robust, bare-metal development environment on Linux.

*   **Build System:** CMake + Ninja (for fast incremental builds).
*   **Dependencies:**
    *   **LLVM/MLIR 19.x:** Building from source to ensure access to the latest MLIR features (Dynamic Dialects, Python Bindings).
    *   **Location:** `deps/llvm-project`
    *   **Build Status:** COMPLETE. LLVM/MLIR 19.x built from source and integrated.

## 3. Implemented Features (The "Body")
We have established the core structural elements of the language in the `tensorlang` dialect.

### 3.1 Type System (`LinearTensorType`)
Defined in `TensorLang.td` and `TensorLangOps.cpp`.
*   **Structure:** `!tensorlang.tensor<Shape, Type, [SymbolicDims]?, [Linear]? >`
*   **Linearity:** The `linear` flag ensures a tensor is consumed exactly once, enabling safe in-place mutations (crucial for memory-constrained LLM inference).
*   **Symbolic Shapes:** Added support for `symbolicShape` (e.g., `["batch", "seq_len"]`) to handle dynamic sequence lengths in Transformers without recompilation.

### 3.2 Core Operations
*   **`tensorlang.symbolic_dim`**: Defines a runtime variable for a dimension size (e.g., `seq_len = 512`).
*   **`tensorlang.matmul`**: The compute workhorse.
    *   **Verification:** Implemented C++ logic to ensure inner dimensions match (`(M x K) * (K x N) -> (M x N)`).
*   **`tensorlang.constant`**: Creates dense tensors from compile-time attributes.
*   **`tensorlang.print`**: Basic I/O.

### 3.3 Build Integration
*   **TableGen:** Configured `mlir-tablegen` to automatically generate C++ class declarations and definitions for Types and Ops.
*   **Dialect Registration:** Updated `TensorLangDialect.cpp` to register the new `LinearTensorType` alongside operations.

## 4. Roadmap

### Phase 1: JIT Execution Engine (COMPLETE)
*   [x] **Build System:** Fix CMake/Ninja build for TensorLang.
*   [x] **JIT Infrastructure:** Implement `JitRunner` using LLVM ORC JIT.
*   [x] **Runtime Library:** Implement `TensorLangRuntime` with reflection hooks.
*   [x] **Symbol Resolution:** Connect JIT to Runtime symbols (e.g., `tensorlang_get_ir`).
*   [x] **Demo:** `self_rewrite_demo.mlir` runs and prints its own IR.

### Phase 2: Matrix Multiplication & Optimization
*   [x] **Linear Types:** Implement `VerifyLinearityPass` to enforce single-use semantics.
*   [x] **MatMul Op:** Lower `tensorlang.matmul` to Linalg.
*   [x] **Bufferization:** Verified `OneShotBufferize` integration with Linalg lowering.
*   [x] **Benchmark:** Ran `matmul_bench.mlir` (256x256). Correctness verified (Val: 512.0).

### Phase 3: LLM Integration (COMPLETE)
*   [x] **Embed LLM:** Implemented `ModelRunner` with Mock backend returning optimized IR.
*   [x] **Runtime:** Implemented `tensorlang_compile` and `tensorlang_get_symbol_address`.
*   [x] **Self-Rewrite:** `self_rewrite_demo.mlir` successfully compiles and executes generated code at runtime.
*   [x] **Full Loop:** Demonstrated `Get IR -> Query -> Compile -> Execute` cycle.

### Phase 4: Local AI Integration (COMPLETE)
*   [x] **Submodule Integration:** Added `llama.cpp` as a core dependency.
*   [x] **Local Runner:** Implemented `LlamaCppModelRunner` for on-device inference.
*   [x] **Few-Shot Prompting:** Configured specialized ChatML prompts for Qwen2.5-Coder models.
*   [x] **Hardware Acceleration:** Optimized for 64-core CPU architectures.
*   [x] **Verified Self-Healing:** Successfully ran the "NeuroLander" demo using local 7B and 32B GGUF models.

## 5. Conclusion
The compiler infrastructure is fully functional. It supports:
1.  **Tensor Math:** Efficient Linalg-based compilation.
2.  **Linear Types:** Safety checks for memory management.
3.  **Self-Optimization:** Runtime reflection and hot-swapping of code via LLM interaction (simulated).

Ready for advanced optimization research.


## 5. Files Created/Modified
*   `tensorlang/include/tensorlang/Dialect/TensorLang/IR/TensorLang.td` (Dialect/Ops/Types)
*   `tensorlang/lib/Dialect/TensorLang/IR/TensorLangOps.cpp` (Verifier Logic)
*   `tensorlang/lib/Dialect/TensorLang/IR/TensorLangDialect.cpp` (Registration)
*   `tensorlang/lib/Dialect/TensorLang/IR/CMakeLists.txt` (Build Config)
*   `build_all.sh` (Background Build Script)
