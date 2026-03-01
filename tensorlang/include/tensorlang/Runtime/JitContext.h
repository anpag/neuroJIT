#ifndef TENSORLANG_RUNTIME_JITCONTEXT_H
#define TENSORLANG_RUNTIME_JITCONTEXT_H

#include "tensorlang/Runtime/ModelRunner.h"
#include "tensorlang/Runtime/StrategyCache.h"
#include <string>
#include <functional>
#include <memory>
#include <atomic>
#include <csetjmp>
#include <mutex>

namespace mlir {
namespace tensorlang {

class JitRunner;

/// A context object for JIT operations exposed to the runtime.
class JitContext {
public:
  static JitContext& getInstance();

  void registerRunner(JitRunner* runner);
  JitRunner* getRunner() const;

  void setModelRunner(std::unique_ptr<ModelRunner> modelRunner);
  ModelRunner* getModelRunner() const;
  
  StrategyCache& getStrategyCache() { return strategyCache; }

  // Helpers for reflection
  void setModuleIR(const std::string& ir);
  std::string getModuleIR() const;

  // Async Hot-Swap State
  void setOptimizedFunction(void* fnPtr);
  void* getOptimizedFunction() const;
  
  /// Tries to set isOptimizing to true. Returns true if successful (lock acquired).
  bool tryStartOptimization();
  void finishOptimization();
  bool isOptimizingCurrently() const { return isOptimizing.load(); }

  // Recovery Support
  std::jmp_buf& getRecoveryPoint() { return recovery_point; }
  
  int getHealingAttempts() const { return healing_attempts.load(); }
  void incrementHealingAttempts() { healing_attempts.fetch_add(1); }
  void resetHealingAttempts() { healing_attempts.store(0); }

  // Performance Telemetry
  void recordTelemetry(double impactVel, double latency) {
    std::lock_guard<std::mutex> lock(statsMutex);
    lastLatency = latency;
    totalLatency += latency;
    executionCount++;
    
    // Impact velocity (smaller is better/safer)
    if (std::abs(impactVel) < std::abs(bestImpactVelocity)) {
      bestImpactVelocity = impactVel;
    }
  }

  double getBestImpactVelocity() const {
    std::lock_guard<std::mutex> lock(statsMutex);
    return bestImpactVelocity;
  }

  double getLastLatency() const { 
    std::lock_guard<std::mutex> lock(statsMutex);
    return lastLatency; 
  }
  double getAverageLatency() const {
    std::lock_guard<std::mutex> lock(statsMutex);
    return executionCount > 0 ? totalLatency / executionCount : 0.0;
  }

  // --- LOBE REGISTRY INTERFACE ---
  void saveLobe(const std::string& name, const std::string& ir);
  std::string loadLobe(const std::string& name);
  bool hasLobe(const std::string& name);

private:
  JitContext() = default;
  JitRunner* runner = nullptr;
  std::unique_ptr<ModelRunner> modelRunner;
  std::string currentIR;
  
  std::atomic<void*> optimizedFunctionPtr{nullptr};
  std::atomic<bool> isOptimizing{false};
  std::atomic<int> healing_attempts{0};

  // Performance stats
  mutable std::mutex statsMutex;
  double lastLatency = 0.0;
  double totalLatency = 0.0;
  double bestImpactVelocity = 100.0; // Starting with high value (bad landing)
  uint64_t executionCount = 0;

  std::jmp_buf recovery_point;
  StrategyCache strategyCache;
};

} // namespace tensorlang
} // namespace mlir

#endif // TENSORLANG_RUNTIME_JITCONTEXT_H
