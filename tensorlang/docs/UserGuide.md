# TensorLang User Guide

## 1. Introduction
TensorLang is a tensor-native programming language designed for high-performance AI workloads. It features **linear types** for deterministic memory management and **reflection capabilities** for self-optimization.

## 2. Core Concepts

### 2.1 Tensors
Tensors are the primary data structure. They are strongly typed and can have static or symbolic shapes.

```mlir
// A static 2x2 float tensor
%A = tensorlang.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
```

### 2.2 Linear Types
Use the `linear` keyword to enforce "use-once" semantics. This allows the compiler to perform in-place updates safely.

```mlir
// A linear tensor must be consumed exactly once
func.func @process(%arg0: !tensorlang.tensor<2x2, f32, linear>) {
  // Consumed here
  %0 = tensorlang.matmul %arg0, %arg0 : ...
  // Error: %arg0 cannot be used again!
}
```

### 2.3 Symbolic Shapes
For LLMs, sequence lengths vary. Use symbolic dimensions to handle dynamic shapes without recompilation.

```mlir
// Define a symbolic dimension 'seq_len'
%s = tensorlang.symbolic_dim "seq_len"

// Use it in a type (future syntax)
// !tensorlang.tensor<symbolic ["batch", "seq_len"], f32>
```

## 3. Operations

### 3.1 Matrix Multiplication
The workhorse of neural networks.

```mlir
%C = tensorlang.matmul %A, %B : tensor<2x2xf32>, tensor<2x2xf32> -> tensor<2x2xf32>
```

### 3.2 Printing
Debug your tensors easily.

```mlir
tensorlang.print %C : tensor<2x2xf32>
```

## 4. Self-Rewriting (Reflection API)
TensorLang programs can inspect and modify themselves at runtime using the embedded LLM interface.

### 4.1 Getting Current IR
Retrieve the MLIR code of the running module.

```mlir
func.func private @tensorlang_get_ir() -> !llvm.ptr
...
%ir_ptr = call @tensorlang_get_ir() : () -> !llvm.ptr
```

### 4.2 Querying the LLM
Ask the embedded model to optimize code or generate new kernels.

```mlir
func.func private @tensorlang_query_model(!llvm.ptr) -> !llvm.ptr
...
%prompt = ...
%new_code = call @tensorlang_query_model(%prompt) : (!llvm.ptr) -> !llvm.ptr
```

## 5. Running Programs
Use the `tensorlang-run` tool to execute `.mlir` files directly.

```bash
tensorlang-run my_program.mlir
```
