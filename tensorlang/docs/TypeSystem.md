# TensorLang: Type System Deep Dive

## 1. Introduction
The type system of TensorLang is designed to bridge the gap between high-level mathematical abstractions (tensors) and low-level hardware constraints (memory bandwidth). Its core innovations are **Linear Types** for deterministic memory management and **Dependent Types** for shape safety.

## 2. Linear Types (Resource Safety)

### 2.1 The Problem: Memory Bandwidth
In modern AI workloads (LLMs), memory bandwidth is the primary bottleneck.
*   **Garbage Collection (GC):** Introduces unpredictable latency spikes and memory overhead.
*   **Manual Management (malloc/free):** Prone to memory leaks and use-after-free errors.
*   **Copying:** Dynamic languages often copy tensors implicitly, wasting bandwidth.

### 2.2 The Solution: Linearity
A **Linear Type** enforces the rule that a value must be **consumed exactly once**.
This allows the compiler to:
1.  **Deterministic Deallocation:** Know exactly when a tensor is no longer needed, inserting `free()` calls automatically at compile time.
2.  **In-Place Mutation:** Safely reuse memory buffers. If a tensor is consumed by an operation and never used again, the operation can write the result directly into the input buffer (In-Place Update).

### 2.3 Syntax & Rules
A tensor type is marked linear with the `linear` keyword.

```mlir
// !tensorlang.tensor<1024x1024, f32, linear>
```

**Rule 1: Use Exactly Once**
```mlir
func.func @linear_example(%x: !tensorlang.tensor<..., linear>) {
  // Correct: %x is consumed by matmul
  %y = tensorlang.matmul %x, %x : ...
  
  // Error: %x cannot be used again here!
  // tensorlang.print %x 
}
```

**Rule 2: No Implicit Duplication**
To use a linear value twice, you must explicitly `copy` it (which is expensive and visible).

## 3. Dependent Types (Shape Safety)

### 3.1 The Problem: Dynamic Shapes
LLMs process sequences of varying lengths.
*   **Static Shapes (C++ templates):** Require recompilation for every batch size.
*   **Dynamic Shapes (Python):** Shapes are checked at runtime, leading to crashes mid-training.

### 3.2 The Solution: Symbolic Dimensions
TensorLang supports symbolic dimensions in types. These are runtime variables that are tracked by the type system.

```mlir
// Define symbolic dimension
%seq_len = tensorlang.symbolic_dim "seq_len"

// A tensor whose shape depends on %seq_len
// !tensorlang.tensor<symbolic ["batch", "seq_len"], f32>
```

The compiler can verify that `matmul(A, B)` is valid even if shapes are unknown at compile time, by checking the symbolic constraints (e.g., `A.dim[1] == B.dim[0]`).

## 4. Summary
*   **Linear Types** = Zero-overhead memory safety + In-place optimizations.
*   **Dependent Types** = Flexible shapes without runtime crashes.
