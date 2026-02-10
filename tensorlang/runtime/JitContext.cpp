#include "tensorlang/Runtime/JitContext.h"
#include <mutex>

namespace mlir {
namespace tensorlang {

JitContext& JitContext::getInstance() {
  static JitContext instance;
  return instance;
}

void JitContext::registerRunner(JitRunner* runner) {
  this->runner = runner;
}

JitRunner* JitContext::getRunner() const {
  return runner;
}

void JitContext::setModelRunner(std::unique_ptr<ModelRunner> mr) {
  modelRunner = std::move(mr);
}

ModelRunner* JitContext::getModelRunner() const {
  return modelRunner.get();
}

void JitContext::setModuleIR(const std::string& ir) {
  currentIR = ir;
}

std::string JitContext::getModuleIR() const {
  return currentIR;
}

void JitContext::setOptimizedFunction(void* fnPtr) {
  // Relaxed ordering is sufficient because the function pointer itself is the synchronization point
  // for the consumer (if they check null). However, Release is safer to ensure code memory is visible.
  optimizedFunctionPtr.store(fnPtr, std::memory_order_release);
}

void* JitContext::getOptimizedFunction() const {
  return optimizedFunctionPtr.load(std::memory_order_acquire);
}

bool JitContext::tryStartOptimization() {
  bool expected = false;
  // Compare-and-Swap (CAS)
  return isOptimizing.compare_exchange_strong(expected, true);
}

void JitContext::finishOptimization() {
  isOptimizing.store(false);
}

} // namespace tensorlang
} // namespace mlir
