# TensorLang: Running Instructions

## 1. Environment Setup
Before running any TensorLang programs, you must set the `LD_LIBRARY_PATH` environment variable so the runtime can find the shared libraries (`libTensorLangRuntime.so`).

```bash
export BUILD_DIR=$(pwd)/tensorlang/build
export LD_LIBRARY_PATH=$BUILD_DIR/lib:$BUILD_DIR/tensorlang/runtime:$LD_LIBRARY_PATH
```

## 2. Running Programs
Use the `tensorlang-run` tool to execute `.mlir` files directly.

### Example: Hello World (Self-Rewrite Demo)
This example demonstrates the reflection capabilities of TensorLang.

```bash
./run_example.sh
```

Or manually:
```bash
tensorlang-run tensorlang/examples/self_rewrite_demo.mlir
```

### Example: Simple MatMul
Create a file `simple.mlir`:
```mlir
func.func @main() {
  %A = tensorlang.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
  tensorlang.print %A : tensor<2x2xf32>
  return
}
```
Run it:
```bash
tensorlang-run simple.mlir
```

## 3. Running Benchmarks
We have a Python script that compares TensorLang performance against NumPy.

```bash
python3 tensorlang/benchmarks/compare_bench.py
```

**Expected Output:**
```
Running Benchmarks (Size: 1024x1024 f32)...
TensorLang: 45.20 ms
NumPy:      12.10 ms
Speedup:    0.27x (vs NumPy)
```
*(Note: NumPy uses optimized BLAS libraries like MKL/OpenBLAS. Our initial implementation uses naive LLVM code generation, so it will likely be slower at first. Optimization passes will improve this later.)*

## 4. Debugging
If execution fails with `symbol lookup error` or `cannot open shared object file`, verify your `LD_LIBRARY_PATH`:

```bash
echo $LD_LIBRARY_PATH
ls -l $BUILD_DIR/tensorlang/runtime/libTensorLangRuntime.so
```
