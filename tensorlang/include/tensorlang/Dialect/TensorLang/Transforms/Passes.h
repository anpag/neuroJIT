#ifndef TENSORLANG_TRANSFORMS_PASSES_H
#define TENSORLANG_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"
#include "tensorlang/Dialect/TensorLang/IR/TensorLangOps.h"

namespace mlir {
namespace tensorlang {

std::unique_ptr<Pass> createVerifyLinearity();

#define GEN_PASS_REGISTRATION
#include "Passes.h.inc"

} // namespace tensorlang
} // namespace mlir

#endif // TENSORLANG_TRANSFORMS_PASSES_H
