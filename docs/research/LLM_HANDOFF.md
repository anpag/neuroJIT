# NeuroJIT: Phase 4 Handoff Guide

You are an AI assistant taking over the **NeuroJIT** project. This project is a self-optimizing "Neurosymbolic" compiler built on LLVM/MLIR that rewrites its own code at runtime.

The previous engineer completed **Phases 1, 2, and 3** on the `serious-impl` branch. The core infrastructure is highly stable: it features a working LLVM ORC JIT, robust hot-swapping with proper symbol shadowing, `setjmp/longjmp` stack unwinding for crash recovery, and a persistent `StrategyCache` (`~/.neurojit/cache.json`).

Your task is to implement **Phase 4: Local AI Integration via llama.cpp**. The previous environment was blocked by a network proxy and lacked the disk space to comfortably build `llama.cpp` and host a 4GB+ GGUF model. Your environment has the CPU and memory to do this.

## Current State of the Repository
*   **Branch:** `serious-impl`
*   **Core Compiler:** Found in `tensorlang/`.
*   **AI Runner Interface:** Look at `tensorlang/include/tensorlang/Runtime/ModelRunner.h`. There is a `MockModelRunner` and a `GeminiModelRunner` currently implemented.

## Hardware Requirements
The user does **not** have a GPU. You must configure `llama.cpp` to run efficiently on a **CPU**. Ensure you use a highly quantized, fast coding model (e.g., `Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf` or `CodeLlama-7B-GGUF`).

## Your Objectives

### 1. Add `llama.cpp` as a Dependency
1.  Clone `llama.cpp` as a Git submodule into `tensorlang/deps/llama.cpp`.
2.  Update the root or runtime CMake configurations to build `llama.cpp` as a static or shared library, ensuring it links correctly with the MLIR build. (Note: The project uses `-fno-exceptions`, ensure `llama.cpp` builds cleanly within this constraint).

### 2. Download the Model
Write a script or use `wget`/`curl` to download a suitable GGUF coding model (under 5GB) into the workspace. A 7B parameter model is ideal for CPU inference.

### 3. Implement `LlamaCppModelRunner`
Create `tensorlang/runtime/LlamaCppModelRunner.cpp` inheriting from `ModelRunner`.
1.  **`load()`:** Initialize the `llama_context` and load the GGUF model from disk into CPU memory.
2.  **`query()`:** Tokenize the incoming MLIR prompt, run the inference loop using the CPU backend, detokenize the result, extract the MLIR code block, and return it.

### 4. Wire It Up
Update `tensorlang/tools/tensorlang-run/tensorlang-run.cpp` or a configuration file to load the new `LlamaCppModelRunner` by default, pointing it to the downloaded model path.

### 5. Verify the Self-Healing Loop
Run the Lunar Lander demo (`./scripts/run_lander.sh`). It should crash, query your local `llama.cpp` model, compile the response, and restart. If the model hallucinates or fails, the compiler will safely retry up to 3 times before exiting.

## Important Context
*   The system uses **LLVM/MLIR 19**.
*   The `StrategyCache` handles zero-latency repeats. Even if your CPU inference takes 5 seconds, the caching mechanism makes it acceptable for this architecture. Focus on generating *correct* MLIR.
*   Tell the local model to **avoid semicolons** and markdown backticks in its MLIR output, as this is parsed strictly.

Good luck!
