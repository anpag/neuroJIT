# Neurosymbolic Optimization: MCTS & Seed AI

NeuroJIT implements a "Seed AI" architecture where the compiler's optimization pass is not a fixed algorithm, but a search process guided by neural reasoning.

## 1. Monte Carlo Tree Search (MCTS)

To explore the massive space of possible code transformations, NeuroJIT uses MCTS. Each node in the tree represents a state of the MLIR AST.

### The Four Phases of MCTS:
1.  **Selection**: Starting from the current baseline (root), the engine traverses the tree using the **UCB1** algorithm to balance exploitation (visiting high-performance branches) and exploration (visiting unknown branches).
2.  **Expansion**: Once a leaf node is reached, the AI Oracle (DeepSeek) is prompted to generate 2 new mutations, creating new child nodes.
3.  **Simulation (Evaluation)**: Each new mutation is compiled and executed in the **Isolated JIT Sandbox**. The resulting execution time is converted into a **Fitness Score**.
4.  **Backpropagation**: The fitness score is propagated back up the tree, updating the `totalScore` and `visitCount` of all parent nodes to inform future selections.

## 2. Seed AI & Recursive Improvement

The "Seed AI" concept refers to the system's ability to modify its own underlying logic. 

*   **Recursive Feedback**: As the system finds better mutations (e.g., tiling for cache locality), it includes these successes in the prompt history for the next iteration. This allows the LLM to "learn" which transformations are effective for the current hardware.
*   **Hot-Swapping**: The engine doesn't just suggest optimizations; it applies them. By hot-swapping the active function pointers, the system evolves while it runs, approaching an "optimal" state for its specific environment and workload.

## 3. Isolated Sandbox Execution

Execution of AI-generated code is inherently risky. NeuroJIT mitigates this by:
*   **LLVM Diagnostics**: Capturing compilation errors before they can crash the main process.
*   **Separate Dylibs**: Loading each candidate into a unique JIT dynamic library to prevent symbol collisions.
*   **Fitness Thresholds**: Only mutations that pass verification and demonstrate performance gain are ever considered for the main hot-swap.
