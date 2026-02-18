#!/bin/bash
set -e

# Determine project root (one level up from scripts/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_ROOT/build"

# Set library path so tensorlang-run can find libTensorLangRuntime.so
export LD_LIBRARY_PATH="$BUILD_DIR/lib:$BUILD_DIR/tensorlang/runtime:$LD_LIBRARY_PATH"

echo "Running self_optimizing_matmul.mlir..."
"$BUILD_DIR/tools/tensorlang-run/tensorlang-run" "$PROJECT_ROOT/tensorlang/benchmarks/self_optimizing_matmul.mlir"
