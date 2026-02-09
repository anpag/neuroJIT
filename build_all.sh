#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

WORKING_DIR="/usr/local/google/home/antoniopaulino/dev/git/compiler"
cd "$WORKING_DIR"

echo "[1/3] Starting LLVM/MLIR Build (this will take 30+ minutes)..."
echo "Logs: $WORKING_DIR/llvm_build.log"
# Build LLVM
cmake --build deps/llvm-project/build > llvm_build.log 2>&1

echo "[2/3] Configuring TensorLang..."
echo "Logs: $WORKING_DIR/tensorlang_config.log"
# Clean and Configure TensorLang
rm -rf build && mkdir build
cmake -G Ninja -S tensorlang -B build \
   -DMLIR_DIR="$WORKING_DIR/deps/llvm-project/build/lib/cmake/mlir" \
   -DLLVM_DIR="$WORKING_DIR/deps/llvm-project/build/lib/cmake/llvm" \
   -DLLVM_EXTERNAL_LIT="$WORKING_DIR/deps/llvm-project/build/bin/llvm-lit" \
   > tensorlang_config.log 2>&1

echo "[3/3] Building TensorLang..."
echo "Logs: $WORKING_DIR/tensorlang_build.log"
# Build TensorLang
cmake --build build > tensorlang_build.log 2>&1

echo "SUCCESS: All builds completed at $(date)."
