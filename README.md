# NeuroJIT: A Self-Optimizing Seed AI Engine

NeuroJIT is an autonomous compiler framework that uses **Monte Carlo Tree Search (MCTS)** and **Large Language Models (LLMs)** to physically mutate and optimize its own machine code at runtime. Built on **MLIR** and **LLVM ORC JIT**, it achieves dynamic performance gains through closed-loop architectural reflection.

## 🚀 The Core Engine (Phase 7: True Self-Modification)
As of March 2026, the NeuroJIT engine has achieved stable, autonomous self-optimization. The system can:
1.  **Analyze** its own MLIR AST (Matrix Multiplication baseline).
2.  **Reason** using local 32B models (DeepSeek-R1) to propose discrete structural mutations.
3.  **Mutate** the in-memory AST using physical MLIR transformations (`unrollLoop`, `tileLoop`).
4.  **Sandbox** the candidate code in an isolated JIT environment to evaluate fitness (execution speed).
5.  **Explore** the optimization space using an MCTS tree with UCB1-guided expansion.
6.  **Hot-Swap** the active execution pointer in the main thread with superior machine code without stopping the process.

## 🏗️ Architecture
The framework is composed of four primary pillars:
*   **The AST Mutator (C++/MLIR)**: Interacts directly with the MLIR API to apply transformations.
*   **The AI Oracle (Llama.cpp)**: Embedded inference engine (DeepSeek-R1 32B) that acts as the "Brain".
*   **The MCTS Worker**: A background thread that orchestrates the explore/evaluate/expand loop.
*   **The JIT Runtime (LLVM ORC)**: Manages executable memory and atomic function pointer redirection.

## 📁 Repository Structure
*   `tensorlang/`: Source code for the MLIR dialect, JIT runner, and optimization workers.
*   `tensorlang/runtime/`: C++ implementation of the MCTS engine and AST mutator.
*   `tensorlang/benchmarks/`: MLIR baselines used for optimization targets (e.g., `matmul_pure.mlir`).
*   `docs/`: Comprehensive technical documentation (Architecture, Concepts, Archive).

## 🛠️ Getting Started

### Prerequisites
*   LLVM/MLIR 19.x (built from source)
*   CMake 3.31+
*   GCC 14+ / GCC 15+
*   Ninja

### Build and Run
The simplest way to execute the engine is using the root helper script:
```bash
bash run_me.sh
```
This will:
1.  Verify the LLVM/MLIR environment.
2.  Build the TensorLang compiler and NeuroJIT runtime.
3.  Initialize the MCTS loop with the DeepSeek 32B model.
4.  Optimize the `matmul` function and trigger a hot-swap upon finding superior logic.

## 📊 Documentation
For deeper dives into specific components:
*   [Technical Architecture Overview](docs/architecture/overview.md)
*   [AST Mutation Schema](tensorlang/docs/LLM_Integration.md)
*   [MCTS Loop & Sandbox Isolation](docs/concepts/neurosymbolic.md)

---
*NeuroJIT is an experimental framework for research in autonomous recursive optimization.*
