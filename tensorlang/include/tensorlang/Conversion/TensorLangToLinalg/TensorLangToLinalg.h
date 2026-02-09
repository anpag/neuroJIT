#ifndef TENSORLANG_CONVERSION_TENSORLANGTOLINALG_H
#define TENSORLANG_CONVERSION_TENSORLANGTOLINALG_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace tensorlang {

/// Creates a pass that lowers TensorLang operations to the Linalg dialect.
/// This transformation is a prerequisite for executing TensorLang code via LLVM.
std::unique_ptr<Pass> createConvertTensorLangToLinalgPass();

/// Registers the pass with the pass manager.
void registerConvertTensorLangToLinalgPass();

} // namespace tensorlang
} // namespace mlir

#endif // TENSORLANG_CONVERSION_TENSORLANGTOLINALG_H
