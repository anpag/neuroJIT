#!/bin/bash
set -e

WORKING_DIR="/usr/local/google/home/antoniopaulino/dev/git/compiler"
cd "$WORKING_DIR"

# Force using GCC 14 instead of GCC 15 (which breaks LLVM build)
export CC=gcc-14
export CXX=g++-14

echo "Using Compiler: $($CC --version | head -n 1)"

echo "[1/4] Configuring LLVM/MLIR..."
# Clean up previous failed config
rm -rf deps/llvm-project/build
mkdir -p deps/llvm-project/build

cmake -G Ninja -S deps/llvm-project/llvm -B deps/llvm-project/build \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_PROJECTS="mlir;clang;lld" \
   -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU" \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DLLVM_INSTALL_UTILS=ON \
   > llvm_config.log 2>&1

echo "[2/4] Building LLVM/MLIR (this will take 30+ minutes)..."
cmake --build deps/llvm-project/build > llvm_build.log 2>&1

echo "[3/4] Configuring TensorLang..."
rm -rf build && mkdir build
cmake -G Ninja -S . -B build \
   -DMLIR_DIR="$WORKING_DIR/deps/llvm-project/build/lib/cmake/mlir" \
   -DLLVM_DIR="$WORKING_DIR/deps/llvm-project/build/lib/cmake/llvm" \
   -DLLVM_EXTERNAL_LIT="$WORKING_DIR/deps/llvm-project/build/bin/llvm-lit" \
   > tensorlang_config.log 2>&1

echo "[4/4] Building TensorLang..."
cmake --build build > tensorlang_build.log 2>&1

echo "SUCCESS: All builds completed at $(date)."
