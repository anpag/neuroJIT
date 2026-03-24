# LLM Mutation Schema & GBNF Grammar

To ensure the AI Oracle generates valid mutations that the C++ AST Mutator can parse, NeuroJIT uses a strict JSON schema and a GBNF (GGML BNF) grammar file.

## 1. JSON Mutation Schema

The model is instructed to output a single JSON object containing a target function and a list of discrete mutations.

### Example Output:
```json
{
  "target_function": "matmul",
  "mutations": [
    { "type": "unroll", "loop_depth": 1, "factor": 4 },
    { "type": "tile", "loop_depth": 0, "sizes": [32, 32, 32] }
  ]
}
```

### Mutation Types:
*   **`unroll`**: Applies loop unrolling.
    *   `factor`: Integer factor to unroll by (e.g., 4, 8).
*   **`tile`**: Applies loop tiling.
    *   `sizes`: An array of 3 integers representing tile sizes for the loop nest.

## 2. GBNF Grammar (`mutation.gbnf`)

The grammar file located at `tensorlang/runtime/models/mutation.gbnf` enforces this structure at the sampling level in `llama.cpp`.

```gbnf
root ::= "{" ws "\"target_function\"" ws ":" ws string ws "," ws "\"mutations\"" ws ":" ws "[" ws mutlist ws "]" ws "}"
ws ::= [ \t\n]*
string ::= "\"" [a-zA-Z0-9_.-]+ "\""
number ::= [0-9]+
mutlist ::= mutation | mutation ws "," ws mutation
mutation ::= "{" ws "\"type\"" ws ":" ws string ws "," ws "\"loop_depth\"" ws ":" ws number ws "," ws "\"factor\"" ws ":" ws number ws "}" | "{" ws "\"type\"" ws ":" ws string ws "," ws "\"loop_depth\"" ws ":" ws number ws "," ws "\"sizes\"" ws ":" ws "[" ws numlist ws "]" ws "}"
numlist ::= number | number ws "," ws number | number ws "," ws number ws "," ws number
```

## 3. The Prompting Strategy

The `LlamaCppModelRunner` constructs a prompt that includes:
1.  **System Prompt**: Explaining the role of the AI as a compiler optimization agent.
2.  **Output Schema**: Explicitly defining the JSON fields.
3.  **Current MLIR**: The actual `affine` dialect representation of the function to be optimized.
4.  **Performance History**: Recent fitness scores to guide the model towards improvements.
