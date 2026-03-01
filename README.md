# NeuroJIT: The Autonomous Neurosymbolic Compiler

**An autonomous compiler system designed to perform runtime recovery and dynamic optimization via integrated Large Language Models (LLMs).**

NeuroJIT is an autonomous compiler system built on the LLVM/MLIR infrastructure. It integrates the logical structure of a custom tensor-native language (TensorLang) with the generative capabilities of Large Language Models (LLMs) to dynamically evolve its execution architecture at runtime.

---

## The Vision: Autonomous Code Evolution
In traditional software paradigms, compiled code remains static. NeuroJIT introduces dynamic, autonomous evolution:
*   **Autonomous Optimization:** The compiler identifies inefficient execution paths at runtime, queries the Reasoning Agent for an optimized intermediate representation, and hot-swaps the execution pointer without halting the process.
*   **Autonomous Runtime Recovery:** When a `tensorlang.assert` violation occurs (e.g., during a flight simulation), the runtime pauses, queries the Synthesis Engine for a resolution, applies the generated MLIR patch, and resumes execution.
*   **Architectural Introspection:** Rather than relying on static heuristic passes for diverse CPU architectures, the system introspects its own intermediate representation to evolve hardware-specific optimizations dynamically.

> Historical Note: This project was developed to investigate MLIR, LLVM, and the computational challenges associated with automated machine code generation.

---

## System Architecture
1.  **The Language:** TensorLang is defined as an MLIR dialect optimized for machine learning operations and linear type systems.
2.  **The Runtime:** Execution is managed by an LLVM ORC JIT engine.
3.  **Reflection:** The system parses its own Intermediate Representation (IR) during active execution.
4.  **The Reasoning Agent:** The extracted IR is transmitted to a local or cloud-based LLM inference engine (e.g., DeepSeek-R1, Qwen, or Gemma).
5.  **Hot-Swap Execution:** The runtime receives the newly synthesized MLIR, compiles it to machine code asynchronously, and redirects the execution pointer to the optimized function.

---

## The Progression of Autonomous Intelligence

The research and development of NeuroJIT has progressed through several distinct phases:

### Phase 1-3: Reactive Recovery (Cloud Inference)
*   **Focus:** Reactive Autonomous Runtime Recovery.
*   **Result:** Established the core JIT infrastructure and the initial retrieval-generation-compilation loop.
*   **Analysis:** Cloud latency introduced significant bottlenecks; MLIR syntax complexity reduced the efficacy of single-shot generation.

### Phase 4-5: Local Agent Architecture (64-Core Inference)
*   **Focus:** Transitioning to fully offline inference utilizing `llama.cpp`.
*   **Breakthrough:** Decoupled the Reasoning Agent (DeepSeek-R1 32B) from the Synthesis Engine (Qwen 2.5 Coder 7B).
*   **Result:** Achieved a 92% success rate in simulated lunar descent tasks.

### Phase 6-7: Modular Synthesis
*   **Focus:** Overcoming the limitations of monolithic function generation.
*   **Breakthrough:** Modular Control Synthesis. The system autonomously partitions logic into specialized, discrete functions (e.g., `@module_memory`, `@module_logic`).
*   **Result:** Achieved 98% JIT reliability for complex proportional-integral-derivative (PID) control algorithms.

### Phase 8: Swarm Intelligence
*   **Focus:** Evolving noise-resilient logic across a concurrent population of 100 simulated agents.
*   **Learning:** Identified the "Reasoning Bottleneck" where complex recursive repairs exceeded established timeouts.

### Phase 9: High-Fidelity Autonomy
*   **Focus:** Bridging the compiler diagnostic engine directly into the synthesis loop.
*   **Breakthrough 1: Scoped Diagnostics.** Implemented an `mlir::ScopedDiagnosticHandler` to capture precise `loc(line:col)` parsing errors.
*   **Breakthrough 2: Asymmetric Repair Bypass.** Decoupled strategy formulation from syntax correction, reducing Time-to-Intelligence from **30 minutes to 44 seconds**.
*   **Result:** Achieved 100% core saturation on 64-thread hardware via prompt evaluation optimization.

---

## Definitive Benchmarks

### Lunar Descent Task Evaluation

| Architecture | Model Configuration | Success Rate | Logic Depth | Latency |
| :--- | :--- | :--- | :--- | :--- |
| **Golden Architect** | DeepSeek-R1 + Qwen 7B | 98% | Advanced (PD/PID) | ~45s |
| **Elite Suite** | R1 + Qwen3 (80B MoE) | 95% | Advanced | High (~180s) |
| **Modern Suite** | Gemma 3 + Qwen 7B | 92% | Intermediate (P Control) | Low (13s) |
| **Legacy Agent** | Qwen 2.5 7B | 35% | Basic | Fast (4s) |

### Historical Performance Data
*   **Matrix Multiplication:** Automated tiling and vectorization achieved a 2.7x speedup over standard scalar implementations.
*   **Swarm Resilience:** Synthesized stabilization modules achieved a 72% survival rate in ±0.5 m/s² gravitational turbulence.

---

## Quick Start Guide (February 2026 Build)

### 1. Build the Adaptive Runtime
Requires LLVM 19, Ninja, and GCC 15+.
```bash
# Build LLVM/MLIR 19 and the latest llama.cpp inference engine
./scripts/setup_and_build.sh
mkdir build && cd build
cmake .. && cmake --build . -j$(nproc)
```

### 2. Execute the Swarm Evolution Simulation
Observe the autonomous generation of noise-resilient control logic across 100 concurrent agents:
```bash
./scripts/run_lander.sh
```

---

## Technical Documentation
*   **[Modular Control Synthesis](docs/COMPARATIVE_EVOLUTION.md)** - Methodologies for overcoming complexity barriers.
*   **[Swarm Intelligence & Autonomous Recovery](docs/SWARM_INTELLIGENCE.md)** - Population-wide evaluation logic.
*   **[Phase 4 Report: Multi-Agent Integration](docs/PHASE_4_REPORT.md)** - Hardware transition and local inference analysis.
*   **[Recursive Architecture Optimization](docs/RECURSIVE_OPTIMIZATION.md)** - Theoretical foundations of iterative generation.

---

## 🚀 Recent Performance Breakthrough (Phase 10)
Through the implementation of a Multi-Tiered Lobe Registry, we have achieved the following:
* **TtI Speedup:** 20,450x improvement in Time-to-Intelligence (from 45s to 2.2ms).
* **Memory Architecture:** L1 Cache (std::unordered_map) for nanosecond lookups during active sessions.
* **Persistent Storage:** L2 Disk Registry for cross-session cumulative learning.
* **Diagnostic Fidelity:** 100% capture of MLIR parsing errors via ScopedDiagnosticHandler.
