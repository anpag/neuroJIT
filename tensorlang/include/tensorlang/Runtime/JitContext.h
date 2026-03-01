#ifndef TENSORLANG_RUNTIME_JITCONTEXT_H
#define TENSORLANG_RUNTIME_JITCONTEXT_H

#include "tensorlang/Runtime/ModelRunner.h"
#include "tensorlang/Runtime/StrategyCache.h"
#include "tensorlang/Runtime/OptimizationWorker.h"

#include <string>
#include <memory>
#include <atomic>
#include <mutex>
#include <unordered_map>

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
  // Restart protocol (Safe crash recovery without longjmp)
  // -----------------------------------------------------------------------
  void requestRestart(const std::string& newIR);
  bool consumeRestartRequest(std::string& outNewIR);

  // -----------------------------------------------------------------------
  // Async optimization worker
  // -----------------------------------------------------------------------
  OptimizationWorker& getWorker();

  void setOnlineOptimization(bool enabled) { onlineOptimizationEnabled_ = enabled; }
  bool isOnlineOptimizationEnabled() const { return onlineOptimizationEnabled_; }

  void shutdown();

  // -----------------------------------------------------------------------
  // Lobe Registry: L1 (RAM) + L2 (Disk) for caching successful LLM outputs
  // -----------------------------------------------------------------------
  void saveLobe(const std::string& name, const std::string& ir);
  std::string loadLobe(const std::string& name);
  bool hasLobe(const std::string& name);

private:
  JitContext();
  void initWorker();

  JitRunner* runner_ = nullptr;
  std::unique_ptr<ModelRunner> modelRunner_;

  mutable std::mutex irMutex_;
  std::string currentIR_;

  bool onlineOptimizationEnabled_ = true;

  std::atomic<void*> optimizedFunctionPtr_{nullptr};

  std::mutex restartMutex_;
  bool restartPending_ = false;
  std::string pendingIR_;

  std::unique_ptr<OptimizationWorker> worker_;
  StrategyCache strategyCache_;

  mutable std::mutex lobeMutex_;
  std::unordered_map<std::string, std::string> lobeCache_;
};

} // namespace tensorlang
} // namespace mlir

#endif // TENSORLANG_RUNTIME_JITCONTEXT_H