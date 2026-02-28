# NeuroJIT

**A self-optimizing "Neurosymbolic" compiler that leverages AI to evolve code at runtime.**

NeuroJIT is built on LLVM/MLIR. It combines the logical rigor of a custom tensor-native language (**TensorLang**) with the creative problem-solving of Large Language Models (LLMs) to create a system that can autonomously repair crashes, optimize performance, and evolve its own "Brain" during execution.

---

## Key Features

*   **Self-Healing Runtime**: Detects imminent crashes (e.g., failed assertions) and queries an embedded LLM to rewrite the offending MLIR code on the fly.
*   **Multi-Agent Reasoning**: Employs a "Brain & Muscle" architecture using **Gemma 3 12B** for high-level logic planning and **Qwen 2.5 Coder 7B** for strict MLIR implementation.
*   **Continuous Evolution**: Proactively profiles JIT-compiled functions and triggers "Evolutionary REM Sleep" cycles to optimize for multi-objective fitness (Safety, Speed, and Resource Efficiency).
*   **On-Device AI**: Deeply integrated with `llama.cpp` for local, private, and offline inference on CPU (optimized for 64-core architectures).
*   **Fuel-Aware Physics**: Evolved from simple "not crashing" to managing fuel consumption and landing precision in lunar descent simulations.

---

## Architecture Summary

The core of the project is **TensorLang**, a dialect optimized for AI workloads and safety:

*   **[TensorLang Dialect](tensorlang/README.md)**: Implementation of `LinearTensor` types and core ops (`matmul`, `symbolic_dim`).
*   **[LLVM ORC JIT](tensorlang/README.md#executionengine)**: Handles live MLIR-to-Machine Code compilation and symbol hot-swapping via dynamic libraries.
*   **[Multi-Agent AI Runner](tensorlang/README.md#runtime)**: A dual-model inference engine that decouples reasoning from implementation.
*   **[Evolutionary Lifeform Engine](docs/PHASE_4_REPORT.md)**: Feedback loops that use real-time telemetry (Impact Velocity, Fuel, Latency) to guide code mutations.

---

## Model Performance & Benchmarks (The "Battle for the Brain")

We have benchmarked various local models on the "NeuroLander" autonomous landing task.

| Architecture | Model(s) | Success Rate | Strategy |
| :--- | :--- | :--- | :--- |
| **Multi-Agent (Elite)** | **Gemma 3 + Qwen 2.5** | **98%** | Gemma plans logic; Qwen writes the MLIR code. |
| **Single-Agent (Stable)**| **Qwen 2.5 Coder 7B** | **85%** | Highly reliable syntax; prone to "physics confusion." |
| **Single-Agent** | **Llama 3.1 8B** | **20%** | Struggled with MLIR SSA name conventions. |
| **Single-Agent** | **DeepSeek V2 Lite** | **5%** | Frequent hallucinations of non-existent MLIR dialects. |

**Final Recommendation:** The **Gemma 3 (Brain) + Qwen 2.5 (Muscle)** pair is the new state-of-the-art for local compiler evolution.

See the full **[Phase 4 Report: Multi-Agent Local AI Integration](./docs/PHASE_4_REPORT.md)** for detailed quantitative analysis.

---

## Quick Start

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

### 2. Run the Evolutionary Lifeform Demo
Watch the compiler autonomously evolve its descent strategy to optimize for soft landing and fuel consumption:
```bash
./scripts/run_lander.sh
```

---

## Roadmap Status

*   **Phase 1-2:** JIT Infrastructure & Tensor Ops (**COMPLETE**)
*   **Phase 3:** Cloud AI Integration (Gemini) (**COMPLETE**)
*   **Phase 4:** Local AI Integration (llama.cpp) (**COMPLETE**)
*   **Phase 5:** Continuous Evolution & Multi-Objective Optimization (**COMPLETE**)
*   **Phase 6:** Autonomous Lifeform Emergence (**IN PROGRESS**)

---
*Created by Antonio Paulino & Gemini 3 Pro Preview / 2.5*
