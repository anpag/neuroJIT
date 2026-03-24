#!/bin/bash
./scripts/build_all.sh
export LD_LIBRARY_PATH="$(pwd)/build/lib:$(pwd)/build/runtime:$(pwd)/tensorlang/deps/llama.cpp/build/bin:$LD_LIBRARY_PATH"
./build/tools/tensorlang-run/tensorlang-run