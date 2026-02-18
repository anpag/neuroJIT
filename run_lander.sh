#!/bin/bash
set -e

WORKING_DIR="/usr/local/google/home/antoniopaulino/dev/git/compiler"
BUILD_DIR="$WORKING_DIR/build"

# Set library path so tensorlang-run can find libTensorLangRuntime.so
export LD_LIBRARY_PATH="$BUILD_DIR/lib:$BUILD_DIR/runtime:$LD_LIBRARY_PATH"

echo "Launching NeuroLander (Self-Healing Demo)..."
"$BUILD_DIR/tools/tensorlang-run/tensorlang-run" "$WORKING_DIR/tensorlang/examples/neuro_lander.mlir"
