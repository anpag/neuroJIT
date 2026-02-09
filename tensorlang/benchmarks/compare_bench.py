import subprocess
import time
import os

def run_mlir_bench():
  start = time.time()
  cmd = [
    "build/tools/tensorlang-run/tensorlang-run",
    "tensorlang/benchmarks/matmul_bench.mlir"
  ]
  # Ensure LD_LIBRARY_PATH is set
  env = os.environ.copy()
  env["LD_LIBRARY_PATH"] = f"{env.get('LD_LIBRARY_PATH', '')}:build/lib:build/tensorlang/runtime"
  
  try:
    result = subprocess.run(cmd, env=env, check=True, capture_output=True, text=True)
    print(result.stdout)
  except subprocess.CalledProcessError as e:
    print(f"MLIR failed: {e}")
    print(e.stderr)
    return 0
  end = time.time()
  return (end - start) * 1000

if __name__ == "__main__":
  print("Running Benchmarks (Size: 256x256 f32)...")
  
  mlir_time = run_mlir_bench()
  print(f"TensorLang: {mlir_time:.2f} ms")
  print("NumPy:      Skipped (not installed)")
