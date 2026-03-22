# NeuroJIT Status Report - March 22, 2026

## 1. Accomplishments & Diagnostics
*   **Variance & Diversity Solved**: Successfully broke the "zero variance" bottleneck where the model was copying few-shot examples verbatim. 
*   **Adaptive Prompting**: Implemented a "Divergent History" prompt in `LlamaCppModelRunner.cpp` that feeds the last 3 attempts (and their rewards) back into the system prompt. This forces the model to move away from previously failed or redundant solutions.
*   **Randomized Seed Constraints**: Injected 5 distinct "seed constraints" (e.g., "The base thrust must be between 0.5 and 1.0", "Use at least 3 arithmetic operations") into each query to manually force exploration across the program space.
*   **Sampling Optimization**: Re-configured the `llama.cpp` sampler chain for maximum exploration:
    *   `Temperature: 0.8f`
    *   `Top-P: 0.95f` / `Top-K: 40`
    *   `Repeat Penalty: 1.15f` (critically prevents reproducing identical constants).
*   **Grammar Alignment**: The model is now perfectly anchored to the `arith` grammar and `module { func.func @get_thrust... }` structure without requiring a concrete math example to copy from.

## 2. Technical State
*   **Metrics (22 Episodes)**: 
    *   **Unique patches**: 10 out of 22 (Previously 1).
    *   **Compile Success**: 21 out of 22 (95%+).
    *   **Reward Gradient**: 11 (1.0), 10 (-0.5), 1 (-1.0). The dataset now contains a rich mix of successes and failure modes for RLHF.
*   **Training Data**: `tensorlang_training_data.jsonl` is now structurally diverse enough for a meaningful LoRA fine-tuning run.
*   **Hot-Reload**: The C++ runtime is correctly monitoring `adapter_latest.bin` and hot-swapping weights without restart.

## 3. Key Findings
*   **The "Template Trap"**: 7B models are easily trapped by few-shot examples. Removing concrete mathematical examples in favor of **Structural Rules + History** is the only way to get true exploration in program synthesis.
*   **Repeat Penalty**: At 1.15x, the model is significantly better at varying its constant values (`1.62` -> `1.5` -> `1.8`) across attempts.

## 4. Current Configuration
*   **Model**: Qwen2.5-Coder-7B-Instruct (q4_k_m)
*   **Prompt Style**: Divergent History + Randomized Seed Constraints.
*   **Inference Chain**: Temp 0.8 -> Top-P 0.95 -> Top-K 40 -> Penalty 1.15.
