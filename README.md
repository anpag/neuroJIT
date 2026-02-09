# NeuroJIT 🧠⚡️

**A self-optimizing compiler that literally asks an AI "how do I make this faster?" while running.**

> ⚠️ **Disclaimer:** This is a weekend experiment co-developed with Gemini. We built this to learn about MLIR, LLVM, and the weird challenges of letting Large Language Models mess with low-level machine code. It is **not** production-ready. It is **not** optimal. It is barely held together by C++ glue and hope. But it *does* work!

## The "What If?"
Compilers are hard. Writing optimizations is hard.
What if, instead of writing heuristic passes for every new CPU architecture, the program could just pause execution, introspect its own source code, and ask an LLM:

> *"Hey, I'm doing a Matrix Multiplication on a CPU. Here's my IR. Rewrite this to use Tiling and Vectorization, please."*

And then—without restarting—compile that new code and hot-swap to it?

**NeuroJIT** does exactly that.

## How it Works (The "Rube Goldberg" Machine)

1.  **The Language:** We defined `TensorLang` (a tiny MLIR dialect) for tensor math.
2.  **The Runtime:** When you run code, it starts up an LLVM ORC JIT engine.
3.  **Reflection:** The code can read its own Intermediate Representation (IR) at runtime.
4.  **The "Brain":** It sends this IR to a "ModelRunner" (currently a Mock, but pluggable to Google Gemini).
5.  **Hot-Swap:** The Runtime receives new MLIR, compiles it to machine code on the fly, and redirects the function pointer.

## Is it fast?
**No.** Well, the *resulting* code can be fast (it's native machine code), but the process involves:
1.  Stopping execution.
2.  Generating text.
3.  Parsing text.
4.  Running a full LLVM compiler pipeline inside your process.

It's heavy. But for long-running workloads (like training a model for days), spending 5 seconds to rewrite the kernel might be worth it. Ideally. In theory.

## The Demo (`self_optimizing_matmul.mlir`)

We have a benchmark that:
1.  Runs a naive $O(n^3)$ Matrix Multiplication (~60ms).
2.  Asks the "AI" for help.
3.  The "AI" (our mock backend) returns a **Tiled implementation** using `scf.for` and `memref` subviews.
4.  The runtime compiles it.
5.  The program switches to the new kernel and runs it (~0.2ms).

## Building & Breaking It

You need LLVM 19 and Ninja. Good luck.

```bash
# Build the Frankenstein monster
./build_all.sh

# Run the self-optimizing demo
./run_example.sh
```

## Credits
*   **Architect:** Antonio Paulino & Gemini 3 Pro Preview / 2.5
*   **Backend:** LLVM / MLIR
*   **Vibe:** Experimental / Chaotic Good
