#!/bin/bash
set -e

# Determine project root (one level up from scripts/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_ROOT/build"

# Set library path so tensorlang-run can find libTensorLangRuntime.so and libllama.so
export LD_LIBRARY_PATH="$BUILD_DIR/lib:$BUILD_DIR/runtime:$PROJECT_ROOT/tensorlang/deps/llama.cpp/build/bin:$LD_LIBRARY_PATH"

MODEL_PATH="${1:-$PROJECT_ROOT/tensorlang/runtime/models/qwen2.5-coder-7b-instruct-q4_k_m.gguf}"

echo "Launching NeuroLander (Self-Healing Demo)..."
timeout 1800 "$BUILD_DIR/tools/tensorlang-run/tensorlang-run" \
    --runner=llama \
    --model="$MODEL_PATH" \
    "$PROJECT_ROOT/tensorlang/examples/neuro_swarm_vector.mlir"
