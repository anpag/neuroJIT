# NeuroJIT

**A self-optimizing compiler that leverages AI to evolve code at runtime.**

NeuroJIT is a "Neurosymbolic" compiler built on LLVM/MLIR. It combines the logical rigor of a custom tensor-native language (**TensorLang**) with the adaptive intuition of LLMs (**Gemini**) to create software that can autonomously optimize and heal itself while running.

---

## The Vision: Autonomous Code Evolution
In traditional software, code is static. NeuroJIT makes code dynamic and resilient:
*   **Autonomous Optimization:** The compiler identifies "hot spots" at runtime, sends the IR to Gemini for expert optimization (e.g., tiling for specific GPUs), and hot-swaps the execution pointer—all without stopping the program.
*   **Self-Healing Systems:** When a `tensorlang.assert` fails (e.g., in a flight simulation), the runtime pauses, queries the AI for a patch, applies the fix, and resumes execution.
*   **Hardware Agnosticism:** High-level math is automatically ported and tuned for new hardware architectures by the AI "ghost in the machine."

**[Read more about the Vision and Real-World Usage →](docs/vision.md)**

---

## Project Architecture

The core of the project is **TensorLang**, a dialect optimized for AI workloads and safety:

*   **[TensorLang Dialect](tensorlang/README.md)**: Implementation of `LinearTensor` types and core ops (`matmul`, `symbolic_dim`).
*   **[LLVM ORC JIT](tensorlang/README.md#executionengine)**: The engine that handles MLIR-to-Machine Code compilation and live symbol hot-swapping.
*   **[AI Runtime Interface](tensorlang/README.md#runtime)**: The bridge connecting the compiler to Gemini for code generation and repair.

---

## Quick Start

### 1. Build the Project
Requires LLVM 19 and Ninja.
```bash
./scripts/build_all.sh
```

### 2. Run the Demos
*   **Self-Healing:** Watch the "NeuroLander" simulation crash and fix itself.
    ```bash
    ./scripts/run_lander.sh
    ```
*   **Optimization:** See the compiler speed up matrix math automatically.
    ```bash
    ./scripts/run_example.sh
    ```

---

## Documentation
*   **[Concepts](docs/concepts/):** Neurosymbolic AI and Tensor Math.
*   **[Architecture](docs/architecture/):** Detailed system design and MLIR pipelines.
*   **[Progress Report](tensorlang/PROGRESS_REPORT.md):** Current development status and roadmap.

---
*Created by Antonio Paulino & Gemini 3 Pro Preview / 2.5*