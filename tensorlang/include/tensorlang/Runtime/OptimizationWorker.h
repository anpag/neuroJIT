#ifndef TENSORLANG_RUNTIME_OPTIMIZATIONWORKER_H
#define TENSORLANG_RUNTIME_OPTIMIZATIONWORKER_H

#include "tensorlang/Runtime/OptimizationStrategy.h"
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <functional>
#include <string>

namespace mlir {
namespace tensorlang {

/// A request submitted to the background optimization worker.
struct OptimizationRequest {
  std::string functionName;    ///< e.g. "get_thrust"
  std::string baselineIR;      ///< Current full MLIR module string
  SimulationResult baseline;   ///< Performance of the current implementation
};

/// Callback invoked on the calling thread when a better strategy is compiled.
/// Provides the new function pointer and the strategy that produced it.
using HotSwapCallback = std::function<void(void*, const ControlStrategy&)>;

/// Asynchronous single-worker queue for optimization requests.
/// The simulation loop NEVER blocks on this worker.
/// Thread safety: submit() is safe to call from any thread.
class OptimizationWorker {
public:
  explicit OptimizationWorker(HotSwapCallback cb);
  ~OptimizationWorker();

  /// Non-blocking. Drops duplicate requests for the same function name.
  void submit(OptimizationRequest req);

  /// Returns true if a request is currently in-flight.
  bool isBusy() const { return busy_.load(std::memory_order_acquire); }

  /// Returns the number of requests processed since construction.
  int processedCount() const { return processed_.load(); }

private:
  void workerLoop();

  HotSwapCallback hotSwapCb_;
  std::queue<OptimizationRequest> queue_;
  std::mutex queueMutex_;
  std::condition_variable cv_;
  std::thread thread_;
  std::atomic<bool> shutdown_{false};
  std::atomic<bool> busy_{false};
  std::atomic<int> processed_{0};
};

} // namespace tensorlang
} // namespace mlir

#endif // TENSORLANG_RUNTIME_OPTIMIZATIONWORKER_H