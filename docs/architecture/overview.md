# NeuroJIT Architecture

The NeuroJIT compiler is a complex "Software-in-the-Loop" system. Here is how the pieces fit together.

```mermaid
graph TD
    User[User Code (TensorLang)] --> Runtime
    Runtime[JIT Runtime] -- 1. Detects Slowness/Crash --> Introspection
    Introspection[Reflection Layer] -- 2. Extracts IR --> Gemini
    Gemini[Google Gemini AI] -- 3. Returns Optimized IR --> Compiler
    Compiler[LLVM/MLIR Compiler] -- 4. Verifies & Compiles --> MachineCode
    MachineCode[x86_64 Binary] -- 5. Hot-Swap --> Runtime
```

## The Components

### 1. The Language: TensorLang
We built a custom Domain Specific Language (DSL) called `TensorLang`. It is built on top of **MLIR** (Multi-Level Intermediate Representation).
*   **Why MLIR?** Most languages (C++, Python) lose information when compiled. MLIR preserves the *structure* (loops, matrices) so the AI can understand the *intent*.

### 2. The Runtime (JIT)
This is the engine. It uses **LLVM ORC JIT**.
*   It doesn't just run code; it *monitors* code.
*   It has hooks (`tensorlang_assert_fail`) that trigger when things go wrong.

### 3. The "Brain" (Gemini)
We connect the runtime to **Google Gemini** via a C++ HTTP client.
*   **Prompt Engineering:** The runtime constructs a prompt like: *"Here is a crashing function. Fix the logic."* or *"Here is a slow loop. Optimize it."*
*   **Response:** Gemini returns raw MLIR code.

### 4. The Hot-Swap Mechanism
This is the magic trick.
1.  The system compiles the new AI-generated code into a shared library in memory.
2.  It uses **function pointer indirection**.
3.  It updates the pointer from the `OldFunction` to the `NewFunction`.
4.  The program continues running (or restarts the simulation) without ever exiting.
