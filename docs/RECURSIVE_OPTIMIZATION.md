# Recursive Architecture Optimization: The "Curiosity Drive"

## 1. Abstract
This document details the transition of NeuroJIT from reactive self-healing to proactive architectural evolution. By implementing a "Curiosity Drive," the compiler autonomously explores complex control theories (e.g., PID, Non-linear Damping) during idle "REM Sleep" cycles, independent of runtime failures.

## 2. Methodology: Sequential Evolution (REM Phase)
To ensure system stability on 64-core hardware, we implemented a sequential evolution loop that occurs between simulation generations. This solves race conditions observed in earlier asynchronous implementations.

### 2.1 Context Management
To prevent memory exhaustion in the `llama.cpp` KV cache during continuous evolution, the **Adaptive Refinement Engine** now utilizes a "Fresh Context" pattern:
*   Models remain resident in RAM.
*   Inference contexts are instantiated, executed, and freed for every query.
*   This ensures zero interference between logic planning (Gemma 3) and implementation (Qwen 2.5).

## 3. Results: Architectural Mutation
The system has moved beyond simple "Gain Tuning" (adjusting constants) to "Structural Synthesis" (changing algorithms).

| Generation | Architecture | Logic Type | Result |
| :--- | :--- | :--- | :--- |
| Gen 1 | Baseline | Proportional (P) | Stable descent, low accuracy. |
| Gen 2 | Mutation Alpha | PID Controller | Synthesized, JIT failure (Syntax). |
| Gen 3 | Mutation Beta | Integral-Aware (PI) | Pending refinement. |

### 3.1 Observed Phenomenon: "The PID Leap"
Gemma 3 (The Brain) autonomously decided to leap from a 1st-order Proportional controller to a 2nd-order PID system to eliminate steady-state error. This demonstrates high-level engineering reasoning within the local inference loop.

## 4. Challenges & Mitigations
1.  **Memory Slot Failure:** Resolved by clearing/recreating llama contexts per query.
2.  **Synthesis Complexity:** Qwen 2.5 occasionally struggles with the SSA-name density of complex PID math.
    *   *Mitigation:* Stricter MLIR templates and few-shot examples for math operations are being implemented.

## 5. Next Steps
*   **Robust Math Synthesis:** Refine the "Muscle" (Qwen) prompt to handle nested arithmetic without syntax errors.
*   **Environmental Stress:** Introduce variable gravity and wind noise to force the "Curiosity Drive" to evolve robust, noise-resistant architectures.
*   **State-Tensor Integration:** Transition from scalar inputs to a hidden state vector to allow the evolved "Brain" to maintain internal memory.
