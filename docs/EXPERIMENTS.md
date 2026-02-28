# NeuroJIT: Phase 4 Local AI Experiments

This document tracks the performance and quality of various local LLMs when tasked with self-healing the "NeuroLander" simulation.

## Experimental Setup
*   **Hardware:** 64-Core CPU, 120GB RAM (No GPU).
*   **Prompting:** ChatML with few-shot examples of MLIR optimization.
*   **Metric:** Inference latency (seconds), Token throughput (tokens/sec), and MLIR "healing" success.

## Trial 1: Qwen 2.5 Coder (7B Instruct)
*   **Model:** `qwen2.5-coder-7b-instruct-q4_k_m.gguf`
*   **Size:** 4.7 GB
*   **Latency:** ~70 seconds
*   **Throughput:** 11.1 tokens/sec
*   **Result:** **SUCCESS (Functional)**
    *   Correctly identified the zero-thrust issue.
    *   Implemented a simple proportional controller (`thrust = (target_v - current_v) * kp`).
    *   **MLIR Quality:** High. No syntax errors, correctly used `arith` dialect.
    *   **Landing Outcome:** Improved descent, but `kp` was slightly too low to prevent a hard landing on the first attempt.

## Trial 2: DeepSeek Coder V2 Lite (16B MoE)
*   **Model:** `deepseek-coder-v2-lite-q4_k_m.gguf`
*   **Size:** 10.3 GB
*   **Latency:** ~45 seconds
*   **Throughput:** 18.6 tokens/sec
*   **Result:** **FAILURE (Hallucination)**
    *   **MLIR Quality:** Poor. Hallucinated an invalid `arith.select` syntax and attempted to redefine the `%thrust` SSA value with a conflicting type (`i1` vs `f32`).
    *   **Error:** `use of value '%thrust' expects different type than prior uses`.
    *   **Landing Outcome:** Crash (Compilation failed).

## Trial 3: Qwen 2.5 Coder (32B Instruct)
*   **Model:** `qwen2.5-coder-32b-instruct-q4_k_m.gguf`
*   **Size:** 19.1 GB
*   **Result:** **FAILURE (Stability)**
    *   The model exceeded the stable inference threshold for the current environment's CPU/memory paging, resulting in a process crash during KV cache initialization.

## Conclusion & Recommendation
**Qwen 2.5 Coder 7B** is the current champion for local MLIR generation. Despite being the smallest model tested, it demonstrated the highest adherence to MLIR syntax rules and provided the most logically sound (compilable) patches. 

**Recommendation:** Use Qwen 7B for real-time healing; use Gemini (Cloud) for complex structural refactoring where higher-order reasoning is required.
