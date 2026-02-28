# NeuroJIT: Phase 4 Local AI Experiments

This document tracks the performance and quality of various local LLMs when tasked with self-healing the "NeuroLander" simulation.

## Experimental Setup
*   **Hardware:** 64-Core CPU, 120GB RAM (No GPU).
*   **Prompting:** ChatML with strict "No Inline Literals" guardrails and physics guidance.
*   **Metric:** Inference latency (seconds), compilation success, and landing outcome.

## Comparison Table (7B - 9B Range)

| Model | Size | Tokens/s | Compilation | Landing Result | Logic Quality |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Qwen 2.5 Coder 7B** | 4.7 GB | 12.2 | **Success** | Improved Descent | **High**. Correct math and syntax. |
| **Llama 3.1 8B** | 4.6 GB | 9.1 | **Failure** | Crash | **Medium**. Hallucinated undeclared SSA values. |
| **Gemma 2 9B** | 5.4 GB | 6.3 | **Failure** | Crash | **Low**. Hallucinated inline literals (broken syntax). |
| **DeepSeek V2 Lite** | 10.3 GB | 18.6 | **Failure** | Crash | **Low**. Hallucinated non-existent MLIR ops. |
| **CodeGemma 7B** | 5.0 GB | 6.3 | **Success** | Crash | **Zero**. Repeated broken code exactly. |

## Conclusion & Champion
**Qwen 2.5 Coder 7B** is the undisputed champion for local MLIR generation in this size class.

### Why Qwen?
1.  **Strict Syntax Adherence**: It was the only model that consistently respected MLIR's strict SSA and typing rules without hallucinating "fake" operations or literals.
2.  **Instruction Following**: It correctly identified the zero-thrust bug and implemented a functional proportional controller based on the physics hints provided.
3.  **Efficiency**: It maintained a stable 12 tokens/sec on CPU, providing a good balance between reasoning depth and latency.

### Final Recommendation
For all future local "Continuous Evolution" and "Self-Healing" tasks, the system will default to **Qwen 2.5 Coder 7B**. For complex architectural changes exceeding local reasoning, the system should fall back to **Gemini 1.5 Pro (Cloud)**.
