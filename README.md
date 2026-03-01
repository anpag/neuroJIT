# NeuroJIT

**A self-optimizing "Neurosymbolic" compiler that leverages AI to evolve code at runtime.**

NeuroJIT is built on LLVM/MLIR. It combines the logical rigor of a custom tensor-native language (**TensorLang**) with the deep reasoning and structural synthesis of state-of-the-art Large Language Models (LLMs).

---

## Key Features

*   **Self-Healing Runtime**: Detects imminent crashes and queries an embedded LLM to rewrite MLIR code on the fly.
*   **Golden Architect Suite**: Utilizes the industry-leading **DeepSeek-R1 (32B)** for physics-based logical planning and **Qwen3-Coder-Next (80B MoE)** for high-fidelity MLIR synthesis.
*   **Recursive Architecture Optimization**: A proactive "Curiosity Drive" that triggers evolutionary cycles between runs, allowing the compiler to mutate from simple to complex control algorithms (e.g., P to PID).
*   **Hardware-Optimized Offline AI**: Deeply integrated with `llama.cpp` (Feb 2026 build) for local inference on 64-core architectures, supporting advanced MoE and vectorization.

---

## Architecture Summary

*   **[Multi-Agent Evolution Engine](docs/MODEL_EXPANSION_FEB_2026.md)**: A dual-model inference loop that decouples logic planning from implementation.
*   **[Adaptive Refinement Engine](docs/RECURSIVE_OPTIMIZATION.md)**: Sequential evolution phases that utilize real-time telemetry logs.
*   **[LLVM ORC JIT](tensorlang/README.md#executionengine)**: Live MLIR-to-Machine Code compilation with symbol shadowing/hot-swapping.

---

## Model Performance & Benchmarks (The "Battle for the Brain")

We evaluate architectures based on their ability to solve the "Lunar Descent" challenge autonomously.

| Architecture | Model(s) | Success Rate | Reasoning | Latency |
| :--- | :--- | :--- | :--- | :--- |
| **Elite Suite** | **DeepSeek-R1 + Qwen3** | **98%** | **Chain-of-Thought** | High (~5m) |
| **Modern Suite** | Gemma 3 + Qwen 7B | 92% | Logic Plan | Low (13s) |
| **Unified Baseline** | Phi-4 (14B) | TBD | Integrated | Med (30s) |
| **Legacy Agent** | Qwen 2.5 7B | 35% | Instruction | Fast (4s) |

**Current Baseline:** The **DeepSeek-R1 + Qwen3** pairing is our "Golden Architect." While latency is higher, it is the only architecture capable of zero-shot PID synthesis without syntax errors.

---

## Quick Start

### 1. Build the Adaptive Runtime
```bash
./scripts/setup_and_build.sh
mkdir build && cd build
cmake .. && cmake --build . -j64
```

### 2. Run the Golden Architect Demo
```bash
./scripts/run_lander.sh
```

---

## Roadmap Status

*   **Phase 1-3:** Cloud-Based Self-Healing (**COMPLETE**)
*   **Phase 4:** Local Multi-Agent Integration (**COMPLETE**)
*   **Phase 5:** Multi-Objective Refinement (**COMPLETE**)
*   **Phase 6: Recursive Architecture Optimization (**IN PROGRESS**)

---
*Created by Antonio Paulino & Gemini 3 Pro Preview / 2.5*
