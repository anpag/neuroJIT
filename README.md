# NeuroJIT: The Self-Evolving Neurosymbolic Compiler

**NeuroJIT is an autonomous compiler system built on LLVM/MLIR that leverages Large Language Models to evolve its own code at runtime.**

By combining the logical rigor of a custom tensor-native language (**TensorLang**) with the creative problem-solving of a "Multi-Agent Architect," NeuroJIT transitions from a simple code repair tool to a stateful, modular intelligence capable of growing its own specialized "Nervous System."

---

## 1. The Journey to Autonomous Intelligence

Our research has progressed through several distinct evolutionary leaps:

### **Phase 1-3: Reactive Self-Healing (The "Repair" Era)**
*   **Goal:** Detect crashes and fix them using Cloud AI (Gemini).
*   **Result:** Established the JIT core and hot-swapping infrastructure.
*   **Learning:** Cloud latency is a bottleneck for real-time evolution.

### **Phase 4-5: Local Multi-Agent Integration (The "Architect" Era)**
*   **Goal:** Move inference offline to 64-core local hardware.
*   **Breakthrough:** Decoupled the **Brain** (Gemma 3 / DeepSeek-R1) from the **Muscle** (Qwen 2.5 Coder).
*   **Success:** Achievement of a 92% landing success rate in lunar simulations.

### **Phase 6-7: Stateful Modular Synthesis (The "Lifeform" Era)**
*   **Goal:** Surpass the "Complexity Wall" of monolithic functions.
*   **Breakthrough:** Implementation of **Digital Synapses** (Memory Tensors) and **Modular Lobes** (Functional Decomposition).
*   **Result:** The compiler now autonomously breaks its own logic into specialized functions, achieving **98% JIT reliability** for complex control algorithms.

---

## 2. Definitive Benchmarks (The "Battle for the Brain")

We evaluate every model and architecture on the **Lunar Descent Stress Test**.

| Architecture | Model(s) | Success Rate | Logic Depth | Syntax |
| :--- | :--- | :--- | :--- | :--- |
| **Golden Architect** | **DeepSeek-R1 + Qwen 7B** | **98%** | **Elite (PD/PID)** | **Perfect** |
| **Elite Suite** | R1 + Qwen3 (80B MoE) | 95% | Elite | High (Type Noise) |
| **Modern Suite** | Gemma 3 + Qwen 7B | 92% | High (P Control) | High |
| **Unified Agent** | Phi-4 (14B) | 45% | High | Medium |
| **Legacy Agent** | Qwen 2.5 7B | 35% | Basic | High |

**Key Finding:** Decoupling reasoning (CoT models like R1) from implementation (specialized coders like Qwen) is the only stable path to synthesizing complex engineering logic offline.

---

## 3. Core Architectural Features

*   **Adaptive Refinement Engine**: A proactive "Curiosity Drive" that triggers "REM Sleep" cycles between runs to mutate and optimize logic.
*   **Modular Nervous System**: Decomposes control logic into linked MLIR lobes (e.g., `@lobe_memory`, `@lobe_logic`).
*   **Recursive Self-Repair**: A feedback loop that captures JIT error diagnostics and feeds them back to the AI for autonomous debugging.
*   **On-Device MoE Support**: Optimized for the latest `llama.cpp` (Feb 2026) to handle 80B+ parameter Mixture-of-Experts models on CPU.

---

## 4. Technical Deep Dives
*   **[Comparative Evolution Report](docs/COMPARATIVE_EVOLUTION.md)**: How Modular Lobes solved the complexity problem.
*   **[Phase 4 Report: Multi-Agent Integration](docs/PHASE_4_REPORT.md)**: Quantitative data on our transition to local AI.
*   **[Recursive Architecture Optimization](docs/RECURSIVE_OPTIMIZATION.md)**: Details on the "Curiosity Drive" and proactive mutation.

---

## 5. Quick Start (Feb 2026 Build)

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
