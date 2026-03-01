# NeuroJIT: The Self-Evolving Neurosymbolic Compiler

**"A computer program that can fix itself when it crashes and speed itself up when it's slow—by talking to an AI."**

NeuroJIT is an autonomous compiler system built on LLVM/MLIR. It combines the logical rigor of a custom tensor-native language (**TensorLang**) with the creative problem-solving of Large Language Models (LLMs) to create a system that evolves its own "Nervous System" during execution.

---

## ⚡️ The Vision: Autonomous Code Evolution
In traditional software, code is static. Once a programmer writes it, it never changes. NeuroJIT changes that:
*   **Autonomous Optimization:** The compiler identifies "hot spots" at runtime, asks an AI "How do I make this faster?", and hot-swaps the execution pointer to the optimized version without stopping the program.
*   **Self-Healing Systems:** When a `tensorlang.assert` fails (e.g., in a flight simulation), the runtime pauses, queries the AI for a patch, applies the fix, and resumes execution.
*   **The "What If?":** What if, instead of writing heuristic passes for every new CPU architecture, the program could just introspect its own source code and evolve?

> ⚠️ **Historical Note:** This project started as a "weekend experiment" co-developed with **Gemini 3 Pro Preview / 2.5** to learn about MLIR, LLVM, and the challenges of letting LLMs manipulate machine code.

---

## 🏗️ The "Rube Goldberg" Machine (How it Works)
1.  **The Language:** We defined `TensorLang`, a dialect optimized for AI and linear types.
2.  **The Runtime:** When code runs, it starts an **LLVM ORC JIT** engine.
3.  **Reflection:** The code reads its own Intermediate Representation (IR) at runtime.
4.  **The "Brain":** It sends this IR to a local or cloud-based AI (DeepSeek-R1, Qwen, or Gemini).
5.  **Hot-Swap:** The runtime receives new MLIR, compiles it to machine code on the fly, and redirects the program to the new version.

---

## 📈 The Journey to Autonomous Intelligence

Our research has progressed through several distinct evolutionary leaps:

### **Phase 1-3: The "Repair" Era (Cloud AI)**
*   **Focus:** Reactive Self-Healing.
*   **Result:** Established the JIT core and the first Get-Query-Compile loop using Google Gemini.
*   **Learning:** Cloud latency is a bottleneck; MLIR "syntax noise" makes single-shot repairs difficult.

### **Phase 4-5: The "Architect" Era (Local 64-Core AI)**
*   **Focus:** Transitioning to fully offline inference using `llama.cpp`.
*   **Breakthrough:** Decoupled the **Brain** (DeepSeek-R1 32B) from the **Muscle** (Qwen 2.5 Coder 7B).
*   **Success:** Achieved a 92% landing success rate in lunar simulations.

### **Phase 6-7: The "Lifeform" Era (Modular Lobes)**
*   **Focus:** Surpassing the "Complexity Wall" of monolithic functions.
*   **Breakthrough:** **Modular Lobe Synthesis**. The system now autonomously breaks its own logic into specialized functions (`@lobe_memory`, `@lobe_logic`).
*   **Result:** Achieved **98% JIT reliability** for complex control algorithms (PID).

### **Phase 8: The "Population" Era (Swarm Intelligence)**
*   **Focus:** Evolving noise-resilient logic across a population of **100 landers**.
*   **Breakthrough:** **Recursive Self-Repair**. The compiler now captures its own JIT error diagnostics and feeds them back to the AI for autonomous debugging.

---

## 📊 Definitive Benchmarks

### **The "Battle for the Brain" (Lunar Descent Task)**

| Architecture | Model(s) | Success Rate | Logic Depth | Latency |
| :--- | :--- | :--- | :--- | :--- |
| **Golden Architect** | **DeepSeek-R1 + Qwen 7B** | **98%** | **Elite (PD/PID)** | ~45s |
| **Elite Suite** | R1 + Qwen3 (80B MoE) | 95% | Elite | High (~3m) |
| **Modern Suite** | Gemma 3 + Qwen 7B | 92% | High (P Control) | Low (13s) |
| **Legacy Agent** | Qwen 2.5 7B | 35% | Basic | Fast (4s) |

### **Historical Performance Data**
*   **Matrix Multiplication:** AI-driven tiling and vectorization achieved a **2.7x speedup** over naive implementations.
*   **Swarm Resilience:** Evolved "Inner Ear" lobes achieved **72% survival** in ±0.5 m/s² gravitational turbulence.

---

## 🛠️ Quick Start (Feb 2026 Build)

### 1. Build the Adaptive Runtime
Requires LLVM 19, Ninja, and GCC 15+.
```bash
# Build LLVM/MLIR 19 and latest llama.cpp
./scripts/setup_and_build.sh
mkdir build && cd build
cmake .. && cmake --build . -j$(nproc)
```

### 2. Run the Swarm Evolution Demo
Watch 100 landers cross-breed noise-resilient logic:
```bash
./scripts/run_lander.sh
```

---

## 📚 Technical Archive & Deep Dives
*   **[Modular Lobe Synthesis & Evolution](docs/COMPARATIVE_EVOLUTION.md)** - How we solved the Complexity Wall.
*   **[Swarm Intelligence & Self-Repair](docs/SWARM_INTELLIGENCE.md)** - Population-wide selection logic.
*   **[Phase 4 Report: Multi-Agent Integration](docs/PHASE_4_REPORT.md)** - Transitioning to 64-core local hardware.
*   **[Recursive Architecture Optimization](docs/RECURSIVE_OPTIMIZATION.md)** - The "Curiosity Drive" theory.

---
*Created by Antonio Paulino & Gemini 3 Pro Preview / 2.5*
