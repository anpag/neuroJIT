# NeuroJIT

**A self-optimizing "Neurosymbolic" compiler that leverages AI to evolve code at runtime.**

NeuroJIT is built on LLVM/MLIR. It combines the logical rigor of a custom tensor-native language (**TensorLang**) with the creative problem-solving of Large Language Models (LLMs) to create a system that can autonomously repair crashes and optimize its own performance during execution.

---

## 🚀 Key Features

*   **Self-Healing Runtime**: Detects imminent crashes (e.g., failed assertions) and queries an embedded LLM to rewrite the offending MLIR code on the fly.
*   **Continuous Evolution**: Proactively profiles JIT-compiled functions and triggers background AI optimization to improve latency and efficiency.
*   **Linear Type System**: Ensures deterministic memory management and safe in-place mutations for high-performance tensor operations.
*   **On-Device AI**: Deeply integrated with `llama.cpp` for local, private, and offline inference on CPU (optimized for 64-core architectures).
*   **Hot-Swap Persistence**: A persistent `StrategyCache` allows the compiler to "remember" successful optimizations across restarts.

---

## 🏗️ Architecture Summary

The core of the project is **TensorLang**, a dialect optimized for AI workloads and safety:

*   **[TensorLang Dialect](tensorlang/README.md)**: Implementation of `LinearTensor` types and core ops (`matmul`, `symbolic_dim`).
*   **[LLVM ORC JIT](tensorlang/README.md#executionengine)**: Handles live MLIR-to-Machine Code compilation and symbol hot-swapping via dynamic libraries.
*   **[AI Runtime Interface](tensorlang/README.md#runtime)**: The bridge connecting the compiler to AI models. Supports **llama.cpp (Local CPU)** and Google Gemini (Cloud).
*   **[Profiling & Evolution](tensorlang/PROGRESS_REPORT.md#phase-5)**: High-resolution timers and async loops that drive proactive code rewriting.

---

## 📊 Model Performance & Benchmarks

We have benchmarked various local models on the "NeuroLander" self-healing task (64-core CPU, 120GB RAM).

| Model | Size | Latency | Tokens/s | Result |
| :--- | :--- | :--- | :--- | :--- |
| **Qwen 2.5 Coder 7B** | 4.7 GB | ~13s | 12.2 | **SUCCESS (Stable logic)** |
| **DeepSeek V2 Lite** | 10.3 GB | ~4s | 18.6 | **FAILURE (Hallucinations)** |
| **Qwen 2.5 Coder 32B** | 19.1 GB | N/A | N/A | Process Crash (Too heavy) |

**Conclusion:** **Qwen 2.5 Coder 7B** is our recommended local backend. It provides the highest adherence to MLIR syntax and stable inference on standard CPU hardware.

---

## 🛠️ Quick Start

### 1. Build the Compiler
Ensure you have CMake, Ninja, and GCC 15+.
```bash
# Build LLVM/MLIR 19 (once)
./scripts/setup_and_build.sh

# Build NeuroJIT
mkdir build && cd build
cmake -G Ninja -S ../tensorlang -B . \
    -DMLIR_DIR=$PWD/../deps/llvm-project/build/lib/cmake/mlir \
    -DLLVM_DIR=$PWD/../deps/llvm-project/build/lib/cmake/llvm
cmake --build .
```

### 2. Run the Self-Healing Demo
Watch the compiler detect a crash and fix its own pilot logic:
```bash
./scripts/run_lander.sh
```

### 3. Generate Training Data
Run thousands of physics simulations to find optimal MLIR constants:
```bash
python3 scripts/generate_training_data.py
```

---

## 📈 Roadmap Status

*   **Phase 1-2:** JIT Infrastructure & Tensor Ops (**COMPLETE**)
*   **Phase 3:** Cloud AI Integration (Gemini) (**COMPLETE**)
*   **Phase 4:** Local AI Integration (llama.cpp) (**COMPLETE**)
*   **Phase 5:** Continuous Evolution & Profiling (**IN PROGRESS**)

---
*Created by Antonio Paulino & Gemini 3 Pro Preview / 2.5*
