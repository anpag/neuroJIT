# NeuroJIT 🧠⚡️

**A self-optimizing compiler that literally asks an AI "how do I make this faster?" while running.**

> ⚠️ **Disclaimer:** This is a weekend experiment co-developed with **Gemini 3 Pro Preview / 2.5**. We built this to learn about MLIR, LLVM, and the weird challenges of letting Large Language Models mess with low-level machine code.
>
> **Is it practical?** No.
> **Is it optimal?** Absolutely not.
> **Is it hilarious?** Yes. It's a compiler that hallucinates optimization passes.

## The "What If?"
Compilers are hard. Writing optimizations is hard.
What if, instead of writing heuristic passes for every new CPU architecture, the program could just pause execution, introspect its own source code, and ask an LLM:

> *"Hey, I'm doing a Matrix Multiplication. Here's my IR. Rewrite this to use Tiling and Vectorization, please. And don't use semicolons."*

And then—without restarting—compile that new code and hot-swap to it?

**NeuroJIT** does exactly that.

## Two "Magic" Features

### 1. Autonomous Optimization
The runtime detects "hot" functions, sends their IR to Gemini, and asks for architecture-specific optimizations (Tiling, Vectorization, Loop Unrolling). It then hot-swaps the naive implementation with the AI's "genius" version.
*   *See `WALKTHROUGH.md` for the performance traces.*

### 2. Self-Healing Systems (NEW!) 🚑
What if the code doesn't just run slowly, but **crashes**? 
NeuroJIT can catch `tensorlang.assert` violations mid-flight. Instead of a segmentation fault, the compiler pauses, analyzes the "crash" state, and asks Gemini to rewrite the logic to prevent the failure.
*   **Demo:** We built a "Lunar Lander" simulation that is programmed to crash. The compiler catches it, writes a PID controller on the fly, and lands it safely.
*   *See `SELF_HEALING_WALKTHROUGH.md` for the terminal logs.*

## How it Works (The "Rube Goldberg" Machine)

1.  **The Language:** We defined `TensorLang` (a tiny MLIR dialect) for tensor math.
2.  **The Runtime:** When you run code, it starts up an LLVM ORC JIT engine.
3.  **Reflection:** The code can read its own Intermediate Representation (IR) at runtime.
4.  **The "Brain":** It sends this IR to **Google Gemini** (via `libcurl`).
5.  **Hot-Swap:** The Runtime receives new MLIR from the AI, compiles it to machine code on the fly, and redirects the function pointer.

## The "Real World" Benchmark (`conv2d_bench.mlir`)

We tried to make it optimize a 2D Convolution.
1.  **Baseline:** Runs a naive $O(N^4)$ convolution. Slow.
2.  **Optimization:** It asks Gemini 2.5 Pro for help.
3.  **Result:** Gemini generates a **highly advanced Tiled Matrix Multiplication** kernel... which sometimes fails to compile because it forgets a type annotation.
4.  **Success:** When it works (or when we use a cached "perfect" response), we see **~2.7x speedups**.

## Building & Breaking It

You need LLVM 19 and Ninja.

```bash
# Build the Frankenstein monster
./build_all.sh

# Run the self-optimizing demo
# (Requires GEMINI_API_KEY in .env, or defaults to Mock mode)
./run_example.sh
```

## Credits
*   **Architect:** Antonio Paulino & Gemini 3 Pro Preview / 2.5
*   **Backend:** LLVM / MLIR
*   **Vibe:** Experimental / Chaotic Good
