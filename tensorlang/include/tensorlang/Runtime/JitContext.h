#ifndef TENSORLANG_RUNTIME_JITCONTEXT_H
#define TENSORLANG_RUNTIME_JITCONTEXT_H

#include "tensorlang/Runtime/ModelRunner.h"
#include <string>
#include <functional>
#include <memory>
#include <atomic>

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

  // Helpers for reflection
  void setModuleIR(const std::string& ir);
  std::string getModuleIR() const;

  // Async Hot-Swap State
  void setOptimizedFunction(void* fnPtr);
  void* getOptimizedFunction() const;
  
  /// Tries to set isOptimizing to true. Returns true if successful (lock acquired).
  bool tryStartOptimization();
  void finishOptimization();

private:
  JitContext() = default;
  JitRunner* runner = nullptr;
  std::unique_ptr<ModelRunner> modelRunner;
  std::string currentIR;
  
  std::atomic<void*> optimizedFunctionPtr{nullptr};
  std::atomic<bool> isOptimizing{false};
};

} // namespace tensorlang
} // namespace mlir

#endif // TENSORLANG_RUNTIME_JITCONTEXT_H
