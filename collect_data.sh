#!/bin/bash

# Configuration
NUM_RUNS=10
MAX_RESTARTS=3
MODEL_RUNNER="llama"
TARGET_FILE="tensorlang/examples/neuro_lander_deterministic.mlir"

# Environment
export PATH=$PATH:/usr/local/google/home/antoniopaulino/tools
export LD_LIBRARY_PATH="$(pwd)/build/lib:$(pwd)/build/runtime:$(pwd)/tensorlang/deps/llama.cpp/build/bin:$LD_LIBRARY_PATH"

echo "=== NeuroJIT Data Collection ==="
echo "Targeting $NUM_RUNS runs with max $MAX_RESTARTS restarts per run."
echo "Model: $MODEL_RUNNER"
echo "--------------------------------"

for i in $(seq 1 $NUM_RUNS); do
  echo "[$(date +'%H:%M:%S')] Starting run $i/$NUM_RUNS..."
  # Run the simulation. We allow it to fail and restart, gathering patches in the background.
  ./build/tools/tensorlang-run/tensorlang-run \
    --runner=$MODEL_RUNNER \
    --max-restarts=$MAX_RESTARTS \
    $TARGET_FILE > /dev/null 2>&1
    
  # Quick count update
  if [ -f "tensorlang_training_data.jsonl" ]; then
    RECORD_COUNT=$(wc -l < tensorlang_training_data.jsonl)
    echo "  -> Current dataset size: $RECORD_COUNT records."
  fi
done

echo "--------------------------------"
echo "Data collection complete!"
if [ -f "tensorlang_training_data.jsonl" ]; then
  echo "Total records collected: $(wc -l < tensorlang_training_data.jsonl)"
fi
