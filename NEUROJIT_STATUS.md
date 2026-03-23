# NeuroJIT Status Report - March 22, 2026

## 1. Accomplishments & Diagnostics
*   **MCTS Seed AI Engine Integration**: Successfully integrated the Monte Carlo Tree Search optimization worker. The engine compiles MLIR, spins up isolated Sandbox JIT environments, and evaluates fitness metrics in a background thread.
*   **Hardware Bypass & Container Limits Identified**: Diagnosed a persistent `SIGSEGV` during `llama.cpp` inference (with DeepSeek 32B) as a container memory limit/`mmap` allocation failure. Fixed by dynamically disabling `mmap` or swapping to a mock JSON oracle for hardware-limited testing.
*   **AST Mutator Implemented**: Capable of reading structured JSON actions (`mutateConstant`, `swapBinaryOperator`), applying them cleanly to the active MLIR module, and regenerating the LLVM IR safely.
*   **Evaluator Pipeline Completed**: Implemented the `MatmulSpeedEvaluator` leveraging `std::chrono` inside an MLIR C-Interface compliant benchmark wrapper (`matmul_pure.mlir`), dynamically scoring and punishing regressions.
*   **Concurrency Constraints Configured**: Solved severe thread-explosion issues via OpenMP by locking Llama's background workers to 16 threads and restricting the KV cache context sizes.

## 2. Technical State
*   **The Engine is LIVE**: The main daemon thread successfully kicks off the MCTS background explorer. The background thread evaluates baseline models (e.g., scoring `2.464268`), iteratively mutates the AST via JSON prompts, recompiles via LLVM OrcJIT, and benchmarks.
*   **Dataset & Hot-Reload**: Still supporting the LoRA fine-tuning infrastructure for the smaller Qwen model. Hot-reloading adapter weights remains active.

## 3. Key Findings
*   **The `mmap` Trap**: Even with 128GB of physical RAM, Docker/WSL cgroups will silently trap and kill heavily-paged virtual memory maps, resulting in `SIGSEGV` inside AVX math kernels. Forcing `use_mmap = false` correctly allocates the heap and allows heavy vector math to run.
*   **Double-Free Memory Corruption in LLVM**: `mlir::parseSourceString` returns an owning reference. Manually calling `module->erase()` inside the MCTS expansion loop results in a fatal double-free during the loop exit. Relying on scope-based destruction solved the thread crash.

## 4. Current Configuration
*   **Mode**: Hybrid MCTS Engine (Llama 32B or Mock JSON Oracle)
*   **Target**: Pure Matrix Multiplication (`tensorlang/benchmarks/matmul_pure.mlir`)
*   **Environment**: Restricted cgroup; running `use_mmap = false` for large models.
