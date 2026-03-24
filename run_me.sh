#!/bin/bash
export PATH="/usr/local/google/home/antoniopaulino/tools/cmake-3.31.5-linux-x86_64/bin:/usr/local/google/home/antoniopaulino/tools:$PATH"
export CC=gcc-14
export CXX=g++-14

./scripts/build_all.sh && \
export LD_LIBRARY_PATH="$(pwd)/build/lib:$(pwd)/build/runtime:$(pwd)/tensorlang/deps/llama.cpp/build/bin:$LD_LIBRARY_PATH" && \
./build/tools/tensorlang-run/tensorlang-run
