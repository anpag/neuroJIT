# Real-World Usage Guide: NeuroJIT Self-Optimizing Compiler

## 1. The Vision: Code That Evolves
In a production environment, you don't manually trigger optimization. The compiler acts as an **autonomous agent** living inside your runtime.

1.  **Write Simple Code:** You write clear, mathematical logic (e.g., in Python/TensorLang).
2.  **Runtime Profiling:** The system detects "hot spots" (slow functions running frequently).
3.  **Autonomous Optimization:** The compiler extracts the slow IR, sends it to an LLM (e.g., **Google Gemini 3 Pro** or specialized coding model), and asks: *"Optimize this for an NVIDIA H100 GPU using Tensor Cores."*
4.  **Hot-Swap:** The LLM returns optimized CUDA/PTX or tiled Linalg/MLIR code. The runtime compiles it and seamlessly switches execution to the new kernel without stopping your program.

## 2. Hypothetical Workflow (Python SDK)

Imagine a Python package `neurojit`:

```python
import neurojit
import torch

# 1. Define your model with the self-optimizing JIT
@neurojit.jit(auto_optimize=True, model="gemini-3-pro-preview")
def transformer_block(x, weights):
    # You write simple, readable math
    q = x @ weights.q
    k = x @ weights.k
    v = x @ weights.v
    # ... attention logic ...
    return result

# 2. Run training loop
for batch in data:
    # Initially runs generic, safe code (~60ms)
    loss = transformer_block(batch, w)
    
    # ... after 100 iterations ...
    # The runtime notices slowness. It queries Gemini in the background.
    # Gemini rewrites the attention mechanism using FlashAttention-style tiling.
    # The runtime hot-swaps the function pointer.
    
    # Suddenly, execution speed drops to ~2ms.
```

## 3. How to "Plug in the Brain"

Currently, we use `MockModelRunner`. To make this real, you implement the Gemini integration (we provided a stub in `tensorlang/runtime/GeminiModelRunner.cpp`):

**File:** `tensorlang/runtime/GeminiModelRunner.cpp`

```cpp
std::string GeminiModelRunner::query(const std::string& prompt) {
    // 1. Prepare HTTP request to Google AI Studio API
    json body = {
        {"model", "models/gemini-3-pro-preview"}, // Or gemini-1.5-pro
        {"contents", {{
            {"parts", {{
                {"text", "You are an MLIR optimization expert. Rewrite this IR to use tiling and vectorization:\n" + prompt}
            }}}
        }}}
    };
    
    // 2. Send Request (using libcurl)
    std::string response = PostHTTP("https://generativelanguage.googleapis.com/v1beta/models/gemini-3-pro-preview:generateContent?key=" + apiKey_, body);
    
    // 3. Extract code block from JSON response
    return ExtractCodeBlock(response);
}
```

Then, in your application setup:

```cpp
JitContext::getInstance().setModelRunner(
    std::make_unique<GeminiModelRunner>()
);
```

## 4. Self-Healing Systems: Beyond Optimization

In safety-critical environments (drones, autonomous vehicles, industrial robots), a logical error isn't just a performance bug—it's a crash. 

**NeuroJIT** introduces "Software-in-the-Loop" healing:
1.  **Define Assertions:** Use `tensorlang.assert` to define physical or logical boundaries (e.g., *"Velocity must be < 5m/s at Alt 0"*).
2.  **Runtime Intervention:** If an assertion fails, the runtime pauses execution and extracts the failing logic.
3.  **Autonomous Fix:** The LLM receives the state and the failing code. It generates a patch (e.g., a PID controller) to satisfy the constraint.
4.  **Hot-Swap:** The system resumes with the corrected behavior.

See `SELF_HEALING_WALKTHROUGH.md` for a complete live-trace of this in action.

## 5. Why This Matters
*   **Hardware Agnostic:** You don't need to hand-tune kernels for every new chip (TPU v5, H100, M3). Gemini does the porting for you.
*   **Safety Net:** Our `VerifyLinearityPass` ensures the AI-generated code manages memory correctly (no leaks), even if the AI hallucinates.
*   **Resilient Systems:** Code that can recognize its own failure and "learn" to fix it mid-flight.
*   **Complexity Abstraction:** Developers write high-level logic; the "ghost in the machine" handles the bit-twiddling.
