#ifndef TENSORLANG_RUNTIME_JITCONTEXT_H
#define TENSORLANG_RUNTIME_JITCONTEXT_H

#include "tensorlang/Runtime/ModelRunner.h"
#include "tensorlang/Runtime/StrategyCache.h"
#include <string>
#include <functional>
#include <memory>
#include <atomic>
#include <csetjmp>

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

  // Recovery Support
  std::jmp_buf& getRecoveryPoint() { return recovery_point; }
  
  int getHealingAttempts() const { return healing_attempts.load(); }
  void incrementHealingAttempts() { healing_attempts.fetch_add(1); }
  void resetHealingAttempts() { healing_attempts.store(0); }

private:
  JitContext() = default;
  JitRunner* runner = nullptr;
  std::unique_ptr<ModelRunner> modelRunner;
  std::string currentIR;
  
  std::atomic<void*> optimizedFunctionPtr{nullptr};
  std::atomic<bool> isOptimizing{false};
  std::atomic<int> healing_attempts{0};

  std::jmp_buf recovery_point;
  StrategyCache strategyCache;
};

} // namespace tensorlang
} // namespace mlir

#endif // TENSORLANG_RUNTIME_JITCONTEXT_H
