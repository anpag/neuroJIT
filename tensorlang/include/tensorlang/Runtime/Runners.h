#ifndef TENSORLANG_RUNTIME_RUNNERS_H
#define TENSORLANG_RUNTIME_RUNNERS_H

#include "tensorlang/Runtime/ModelRunner.h"
#include <memory>

namespace mlir {
namespace tensorlang {

std::unique_ptr<ModelRunner> createGeminiModelRunner();
std::unique_ptr<ModelRunner> createLlamaCppModelRunner();

} // namespace tensorlang
} // namespace mlir

#endif // TENSORLANG_RUNTIME_RUNNERS_H
