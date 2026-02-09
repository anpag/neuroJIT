#ifndef TENSORLANG_RUNTIME_JITCONTEXT_H
#define TENSORLANG_RUNTIME_JITCONTEXT_H

#include "tensorlang/Runtime/ModelRunner.h"
#include <string>
#include <functional>
#include <memory>

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

private:
  JitContext() = default;
  JitRunner* runner = nullptr;
  std::unique_ptr<ModelRunner> modelRunner;
  std::string currentIR;
};

} // namespace tensorlang
} // namespace mlir

#endif // TENSORLANG_RUNTIME_JITCONTEXT_H
