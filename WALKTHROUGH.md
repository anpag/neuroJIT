# Real-World Walkthrough: Self-Optimizing MatMul

This document captures a live run of **NeuroJIT** optimizing a Matrix Multiplication kernel using **Google Gemini 2.5 Pro**.

## 1. The Setup (`self_optimizing_matmul.mlir`)

We start with a naive, unoptimized Matrix Multiplication ($O(n^3)$) operating on $256 	imes 256$ tensors.

**Original Code (Abstracted):**
```mlir
    %C = tensorlang.matmul %A, %B : tensor<256x256xf32> ...
```

## 2. Execution Log

When running `./run_example.sh` with a valid `GEMINI_API_KEY`:

### Step A: Baseline Performance
The runtime executes the original unoptimized code.

```text
Running self_optimizing_matmul.mlir...
[GeminiRunner] API Key loaded.
Running Original MatMul (256x256)...
Time: 60.099820 ms
```

### Step B: The "Brain" Kicks In
The program pauses and sends a query to the Gemini API.

**Prompt Sent:**
> "You are a compiler optimization expert. Return an MLIR module containing a function @main_optimized that performs a Tiled Matrix Multiplication (64x64 tiling) using scf.for. IMPORTANT: Do NOT use memref.subview. Use explicit index arithmetic..."

**System Log:**
```text
[GeminiRunner] Sending prompt to Gemini...
[GeminiRunner] Using URL: https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent?key=...
[GeminiRunner] API Key Length: 39
```

### Step C: AI Code Generation
Gemini 2.5 Pro understands the request and generates a fully valid MLIR module implementing **Tiled Matrix Multiplication** with manual index arithmetic (to be JIT-friendly).

**Received Response (Snippet):**
```json
{
  "candidates": [
    {
      "content": {
        "parts": [
          {
            "text": "module {
  func.func @main_optimized() -> i32 {
 ... scf.for %i = %c0 to %c256 step %c1 ..."
          }
        ]
      }
    }
  ]
}
```

**The Generated Kernel (Cleaned):**
```mlir
    // Tiled Matrix Multiplication: C = A * B
    // Outer loops iterate over the tiles
    scf.for %i0 = %c0 to %c256 step %c64 {
      scf.for %j0 = %c0 to %c256 step %c64 {
        scf.for %k0 = %c0 to %c256 step %c64 {
          // Inner loops iterate within a tile
          scf.for %ii = %c0 to %c64 step %c1 {
             // ... manual index arithmetic ...
             %c_val = memref.load %C[%i, %j]
             %prod = arith.mulf %a_val, %b_val : f32
             %new_acc = arith.addf %iter_acc, %prod : f32
             memref.store %acc, %C[%i, %j]
          }
        }
      }
    }
```

### Step D: Hot-Swap & Speedup
The runtime JIT-compiles this new string into machine code (`x86_64`) and executes it immediately.

```text
[GeminiRunner] Parsed Code Length: 2418
[Runtime] Model returned 2418 bytes.
Running Optimized MatMul (Tiled)...
Time: 21.874766 ms
```

## 3. Results

| Implementation | Execution Time | Notes |
| :--- | :--- | :--- |
| **Naive MatMul** | **60.10 ms** | Standard lowering, no tiling. |
| **NeuroJIT (Gemini 2.5)** | **21.87 ms** | Tiled (64x64), explicit memory management. |

**Speedup:** **2.75x**

This proves the compiler successfully introspected its state, outsourced optimization to a Neural Network, and upgraded itself at runtime without restarting.
