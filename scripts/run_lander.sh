#!/bin/bash
set -e

# Determine project root (one level up from scripts/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_ROOT/build"

# Set library path so tensorlang-run can find libTensorLangRuntime.so
export LD_LIBRARY_PATH="$BUILD_DIR/lib:$BUILD_DIR/runtime:$LD_LIBRARY_PATH"

echo "Launching NeuroLander (Self-Healing Demo)..."
"$BUILD_DIR/tools/tensorlang-run/tensorlang-run" "$PROJECT_ROOT/tensorlang/examples/neuro_lander.mlir"
