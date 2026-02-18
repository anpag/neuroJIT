#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# Determine project root (one level up from scripts/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "Project Root: $PROJECT_ROOT"

echo "[1/3] Starting LLVM/MLIR Build (this will take 30+ minutes)..."
echo "Logs: $PROJECT_ROOT/llvm_build.log"
# Build LLVM
cmake --build deps/llvm-project/build > llvm_build.log 2>&1

echo "[2/3] Configuring TensorLang..."
echo "Logs: $PROJECT_ROOT/tensorlang_config.log"
# Clean and Configure TensorLang
rm -rf build && mkdir build
cmake -G Ninja -S tensorlang -B build \
   -DMLIR_DIR="$PROJECT_ROOT/deps/llvm-project/build/lib/cmake/mlir" \
   -DLLVM_DIR="$PROJECT_ROOT/deps/llvm-project/build/lib/cmake/llvm" \
   -DLLVM_EXTERNAL_LIT="$PROJECT_ROOT/deps/llvm-project/build/bin/llvm-lit" \
   > tensorlang_config.log 2>&1

echo "[3/3] Building TensorLang..."
echo "Logs: $PROJECT_ROOT/tensorlang_build.log"
# Build TensorLang
cmake --build build > tensorlang_build.log 2>&1

echo "SUCCESS: All builds completed at $(date)."
