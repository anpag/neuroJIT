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

} // namespace tensorlang
} // namespace mlir
