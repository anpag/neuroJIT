#!/bin/bash
set -e

WORKING_DIR="/usr/local/google/home/antoniopaulino/dev/git/compiler"
BUILD_DIR="$WORKING_DIR/build"

# Set library path so tensorlang-run can find libTensorLangRuntime.so
export LD_LIBRARY_PATH="$BUILD_DIR/lib:$BUILD_DIR/tensorlang/runtime:$LD_LIBRARY_PATH"

echo "Running self_optimizing_matmul.mlir..."
"$BUILD_DIR/tools/tensorlang-run/tensorlang-run" "$WORKING_DIR/tensorlang/benchmarks/self_optimizing_matmul.mlir"
