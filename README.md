# NeuroJIT

**A self-optimizing compiler that literally asks an AI "how do I make this faster?" while running.**

---

## ⚡️ TL;DR: What is this?
**Imagine a computer program that can fix itself when it crashes and speed itself up when it's slow—by talking to an AI.**

Usually, code is static: once a programmer writes it, it never changes. **NeuroJIT** changes that. It's a "Neurosymbolic" compiler.

*   **"Symbolic":** These are the rules. We built our own language (**TensorLang**) that follows strict, logical principles. It’s great at math but doesn't know how to optimize itself for every computer chip in the world.
*   **"Neural":** This is the intuition. We connected the language to an AI (**Gemini**). When the program runs, it "looks" at its own code, realizes it could be better, and asks the AI to rewrite it on the fly.

**In short:** It's software that evolves while it's running. It's half-robot, half-brain.

---

> ⚠️ **Disclaimer:** This is a weekend experiment co-developed with **Gemini 3 Pro Preview / 2.5**. We built this to learn about MLIR, LLVM, and the weird challenges of letting Large Language Models mess with low-level machine code.

## 📚 Documentation & Guides

We have organized the documentation to help you understand the magic (and the math) behind NeuroJIT.

### 🧠 Concepts (For Humans)
*   **[What is "Neurosymbolic" AI?](docs/concepts/neurosymbolic.md)** - Understanding why we combine Logic + AI.
*   **[Tensor Math & Optimization](docs/concepts/tensor_math.md)** - What are Convolutions, Tensors, and Tiling? (Explained simply).

### 🏗️ Architecture (For Engineers)
*   **[System Overview](docs/architecture/overview.md)** - How TensorLang, LLVM, and Gemini talk to each other.
*   **[The Vision](docs/vision.md)** - How this could work in a real-world production system.

### 🎮 Demos & Walkthroughs
*   **[Self-Healing Lunar Lander](docs/demos/self_healing_lander.md)** - Watch the compiler save a crashing simulation mid-flight.
*   **[Autonomous Optimization](docs/demos/autonomous_optimization.md)** - See the compiler speed up matrix math by 2.7x automatically.

---

## Quick Start

You need LLVM 19 and Ninja installed.

### 1. Build the Project
```bash
./build_all.sh
```

### 2. Run the Self-Healing Demo
This runs the "NeuroLander" simulation. It *will* crash, and then it *will* fix itself.
```bash
./run_lander.sh
```

### 3. Run the Optimization Demo
This runs a Matrix Multiplication benchmark.
```bash
./run_example.sh
```

## Credits
*   **Architect:** Antonio Paulino & Gemini 3 Pro Preview / 2.5
*   **Backend:** LLVM / MLIR
*   **Vibe:** Experimental / Chaotic Good
