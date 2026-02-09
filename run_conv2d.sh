#!/bin/bash
set -e

WORKING_DIR="/usr/local/google/home/antoniopaulino/dev/git/compiler"
BUILD_DIR="$WORKING_DIR/build"

# Set library path
export LD_LIBRARY_PATH="$BUILD_DIR/lib:$BUILD_DIR/runtime:$LD_LIBRARY_PATH"

echo "Running conv2d_bench.mlir..."
"$BUILD_DIR/tools/tensorlang-run/tensorlang-run" "$WORKING_DIR/tensorlang/benchmarks/conv2d_bench.mlir"
