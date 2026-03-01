#include "tensorlang/Runtime/OptimizationWorker.h"
#include "tensorlang/Runtime/MLIRTemplates.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdio>

// Forward declaration — defined in Runtime.cpp
extern "C" {
int tensorlang_compile(const char* ir_string);
void* tensorlang_get_symbol_address(const char* name);
}

namespace mlir {
namespace tensorlang {

OptimizationWorker::OptimizationWorker(HotSwapCallback cb)
    : hotSwapCb_(std::move(cb)) {
  thread_ = std::thread(&OptimizationWorker::workerLoop, this);
}

OptimizationWorker::~OptimizationWorker() {
  {
    std::lock_guard<std::mutex> lock(queueMutex_);
    shutdown_.store(true);
  }
  cv_.notify_all();
  if (thread_.joinable()) thread_.join();
}

void OptimizationWorker::submit(OptimizationRequest req) {
  {
    std::lock_guard<std::mutex> lock(queueMutex_);
    // Drop duplicate requests for the same function to avoid queue buildup
    if (!queue_.empty() && queue_.back().functionName == req.functionName) {
      printf("[Worker] Dropping duplicate request for '%s'\n",
             req.functionName.c_str());
      return;
    }
    queue_.push(std::move(req));
  }
  cv_.notify_one();
}

void OptimizationWorker::workerLoop() {
  while (true) {
    OptimizationRequest req;
    {
      std::unique_lock<std::mutex> lock(queueMutex_);
      cv_.wait(lock, [this] {
        return !queue_.empty() || shutdown_.load();
      });
      if (shutdown_.load() && queue_.empty()) break;
      req = std::move(queue_.front());
      queue_.pop();
    }

    busy_.store(true, std::memory_order_release);
    printf("[Worker] Processing optimization for '%s'\n",
           req.functionName.c_str());

    // The new IR comes pre-instantiated from the GA+Template pipeline.
    // We just need to compile it and hot-swap if successful.
    if (req.baselineIR.empty()) {
      printf("[Worker] Empty IR in request — skipping\n");
      busy_.store(false, std::memory_order_release);
      continue;
    }

    if (tensorlang_compile(req.baselineIR.c_str()) == 0) {
      void* fnPtr = tensorlang_get_symbol_address(req.functionName.c_str());
      if (fnPtr) {
        // ControlStrategy is embedded in req for the callback
        // For now pass a default — the GA records fitness separately
        hotSwapCb_(fnPtr, ControlStrategy{});
        printf("[Worker] Compiled and hot-swapped '%s' successfully\n",
               req.functionName.c_str());
      } else {
        printf("[Worker] Compiled but symbol '%s' not found\n",
               req.functionName.c_str());
      }
    } else {
      printf("[Worker] Compilation failed for '%s'\n",
             req.functionName.c_str());
    }

    processed_.fetch_add(1, std::memory_order_relaxed);
    busy_.store(false, std::memory_order_release);
  }
  printf("[Worker] Thread exiting cleanly\n");
}

} // namespace tensorlang
} // namespace mlir