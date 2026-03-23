#ifndef TENSORLANG_RUNTIME_OPTIMIZATIONWORKER_H
#define TENSORLANG_RUNTIME_OPTIMIZATIONWORKER_H

#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <functional>
#include <string>
#include <vector>
#include <memory>

namespace mlir {
namespace tensorlang {

class IEvaluator; // Forward declare

struct MCTSNode {
  std::string irState;
  int visits = 0;
  float totalScore = 0.0f;
  std::vector<std::shared_ptr<MCTSNode>> children;
  std::weak_ptr<MCTSNode> parent;
};

/// A generic request submitted to the background optimization worker.
struct OptimizationRequest {
  std::string functionName;    ///< The function that needs attention.
  std::string originalIR;      ///< Current MLIR module string.
  std::string errorMessage;    ///< If empty, it's an optimization request. If populated, it's a crash/repair request.
  IEvaluator* evaluator = nullptr; ///< The evaluator interface to use during MCTS simulations.
};

/// Callback invoked on the calling thread when better/fixed MLIR is compiled.
/// Provides the new function pointer and the optimized MLIR source.
using HotSwapCallback = std::function<void(void*, const std::string&)>;

/// Asynchronous single-worker queue for AI-guided compiler requests.
/// The main execution loop NEVER blocks on this worker.
class OptimizationWorker {
public:
  explicit OptimizationWorker(HotSwapCallback cb);
  ~OptimizationWorker();

  /// Non-blocking. Returns false if the request was dropped.
  bool submit(OptimizationRequest req);

  /// Returns true if a request is currently in-flight.
  bool isBusy() const { return busy_.load(std::memory_order_acquire); }

  /// Returns the number of requests processed since construction.
  int processedCount() const { return processed_.load(); }

private:
  void workerLoop();

  HotSwapCallback hotSwapCb_;
  std::queue<std::unique_ptr<OptimizationRequest>> queue_;
  std::unique_ptr<std::mutex> queueMutex_;
  std::unique_ptr<std::condition_variable> cv_;
  std::thread thread_;
  std::atomic<bool> shutdown_{false};
  std::atomic<bool> busy_{false};
  std::atomic<int> processed_{0};
};

} // namespace tensorlang
} // namespace mlir

#endif // TENSORLANG_RUNTIME_OPTIMIZATIONWORKER_H
