#ifndef TENSORLANG_RUNTIME_JITCONTEXT_H
#define TENSORLANG_RUNTIME_JITCONTEXT_H

#include "tensorlang/Runtime/ModelRunner.h"
#include "tensorlang/Runtime/StrategyCache.h"
#include "tensorlang/Runtime/OptimizationStrategy.h"
#include "tensorlang/Runtime/OptimizationWorker.h"
#include "tensorlang/Runtime/StrategyGA.h"

#include <string>
#include <memory>
#include <atomic>
#include <mutex>
#include <vector>
#include <unordered_map>

// REMOVED: #include <csetjmp>
// REASON: longjmp is undefined behavior across C++ stack frames with
//         non-trivial destructors. Replaced with restartRequested flag.

namespace mlir {
namespace tensorlang {

class JitRunner;

/// Central context object. Singleton. Thread-safe for all public methods.
class JitContext {
public:
  static JitContext& getInstance();

  // -----------------------------------------------------------------------
  // JIT Runner registration
  // -----------------------------------------------------------------------
  void registerRunner(JitRunner* runner);
  JitRunner* getRunner() const;

  // -----------------------------------------------------------------------
  // Model runner (LLM backend)
  // -----------------------------------------------------------------------
  void setModelRunner(std::unique_ptr<ModelRunner> modelRunner);
  ModelRunner* getModelRunner() const;

  // -----------------------------------------------------------------------
  // Legacy strategy cache (kept for backward compatibility)
  // -----------------------------------------------------------------------
  StrategyCache& getStrategyCache() { return strategyCache_; }

  // -----------------------------------------------------------------------
  // Module IR reflection
  // -----------------------------------------------------------------------
  void setModuleIR(const std::string& ir);
  std::string getModuleIR() const;

  // -----------------------------------------------------------------------
  // Hot-swap: atomic function pointer
  // -----------------------------------------------------------------------
  void setOptimizedFunction(void* fnPtr);
  void* getOptimizedFunction() const;

  // -----------------------------------------------------------------------
  // Restart protocol (replaces longjmp/setjmp)
  // Set by tensorlang_assert_fail. Checked by the main run loop.
  // -----------------------------------------------------------------------
  void requestRestart(const std::string& newIR);
  bool consumeRestartRequest(std::string& outNewIR);

  // -----------------------------------------------------------------------
  // Telemetry and fitness
  // -----------------------------------------------------------------------
  void recordResult(const SimulationResult& result);
  SimulationResult getLastResult() const;
  double getAverageScore() const;
  double getBestScore() const;
  const Individual& getBestIndividual() const;

  // -----------------------------------------------------------------------
  // Async optimization worker
  // -----------------------------------------------------------------------
  OptimizationWorker& getWorker();

  // -----------------------------------------------------------------------
  // Genetic algorithm
  // -----------------------------------------------------------------------
  StrategyGA& getGA();

  // -----------------------------------------------------------------------
  // Lobe Registry: L1 (RAM) + L2 (Disk)
  // -----------------------------------------------------------------------
  void saveLobe(const std::string& name,
                const std::string& ir,
                const SimulationResult& result);
  std::string loadLobe(const std::string& name);
  bool hasLobe(const std::string& name);
  /// Returns the SimulationResult stored with the lobe, or default if absent.
  SimulationResult loadLobeResult(const std::string& name);

private:
  JitContext();
  void initWorker();

  JitRunner* runner_ = nullptr;
  std::unique_ptr<ModelRunner> modelRunner_;

  mutable std::mutex irMutex_;
  std::string currentIR_;

  std::atomic<void*> optimizedFunctionPtr_{nullptr};

  // Restart state (replaces jmp_buf)
  std::mutex restartMutex_;
  bool restartPending_ = false;
  std::string pendingIR_;

  // Telemetry
  mutable std::mutex statsMutex_;
  std::vector<SimulationResult> history_;
  Individual bestIndividual_;

  // Subsystems
  std::unique_ptr<OptimizationWorker> worker_;
  std::unique_ptr<StrategyGA> ga_;
  StrategyCache strategyCache_;

  // L1 lobe cache
  mutable std::mutex lobeMutex_;
  std::unordered_map<std::string, std::string> lobeCache_;
  std::unordered_map<std::string, SimulationResult> lobeResults_;
};

} // namespace tensorlang
} // namespace mlir

#endif // TENSORLANG_RUNTIME_JITCONTEXT_H