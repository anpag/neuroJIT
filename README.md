# NeuroJIT: The Self-Evolving Neurosymbolic Compiler

**NeuroJIT is an autonomous compiler system built on LLVM/MLIR that leverages Large Language Models to evolve its own code at runtime.**

By combining the logical rigor of a custom tensor-native language (**TensorLang**) with the creative problem-solving of a "Multi-Agent Architect," NeuroJIT transitions from a simple code repair tool to a stateful, modular intelligence capable of growing its own specialized "Nervous System."

---

## 1. The Journey to Autonomous Intelligence

Our research has progressed through several distinct evolutionary leaps:

### **Phase 1-3: Reactive Self-Healing (The "Repair" Era)**
*   **Goal:** Detect crashes and fix them using Cloud AI.
*   **Result:** Established the JIT core and hot-swapping infrastructure.

### **Phase 4-5: Local Multi-Agent Integration (The "Architect" Era)**
*   **Goal:** Move inference offline to 64-core local hardware.
*   **Breakthrough:** Decoupled the **Brain** (DeepSeek-R1) from the **Muscle** (Qwen 2.5 Coder).

### **Phase 6-7: Stateful Modular Synthesis (The "Lifeform" Era)**
*   **Goal:** Surpass the "Complexity Wall" of monolithic functions.
*   **Breakthrough:** Implementation of **Modular Lobes** and **Digital Synapses**.
*   **Result:** Achieved **98% JIT reliability** for complex control algorithms.

### **Phase 8: Swarm Intelligence (The "Population" Era)**
*   **Goal:** Evolve noise-resilient logic across a population of 100 landers.
*   **Breakthrough:** **Recursive Self-Repair** where the compiler autonomously debugs its own code using JIT diagnostics.

---

## 2. Technical Deep Dives
*   **[Phase 8: Swarm Intelligence & Self-Repair](docs/SWARM_INTELLIGENCE.md)**: Details on population evolution and autonomous debugging.
*   **[Comparative Evolution Report](docs/COMPARATIVE_EVOLUTION.md)**: How Modular Lobes solved the complexity problem.
*   **[Phase 4 Report: Multi-Agent Integration](docs/PHASE_4_REPORT.md)**: Quantitative data on our transition to local AI.

---

## 3. Definitive Benchmarks

| Architecture | Model(s) | Success Rate | Logic Depth | Syntax |
| :--- | :--- | :--- | :--- | :--- |
| **Golden Architect** | **DeepSeek-R1 + Qwen 7B** | **98%** | **Elite (PD/PID)** | **Perfect** |
| **Modern Suite** | Gemma 3 + Qwen 7B | 92% | High (P Control) | High |
| **Legacy Agent** | Qwen 2.5 7B | 35% | Basic | High |

---

## 4. Quick Start (Feb 2026 Build)

### Build the Runtime
```bash
./scripts/setup_and_build.sh
mkdir build && cd build
cmake .. && cmake --build . -j64
```

### Run the Swarm Evolution
Observe 100 parallel landers evolving and cross-breeding noise-resilient logic:
```bash
./scripts/run_lander.sh
```

---
*Created by Antonio Paulino & Gemini 3 Pro Preview / 2.5*
