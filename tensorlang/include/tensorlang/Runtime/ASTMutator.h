#ifndef TENSORLANG_RUNTIME_ASTMUTATOR_H
#define TENSORLANG_RUNTIME_ASTMUTATOR_H

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringRef.h"
#include <optional>
#include <string>

namespace mlir {
namespace tensorlang {

class ASTMutator {
public:
  static std::optional<mlir::ModuleOp> applyMutations(mlir::ModuleOp baseModule, llvm::StringRef jsonCommand);
  static std::optional<mlir::ModuleOp> mutateConstant(mlir::ModuleOp baseModule, llvm::StringRef funcName, int constantIndex, float newValue);
  static std::optional<mlir::ModuleOp> swapBinaryOperator(mlir::ModuleOp baseModule, llvm::StringRef funcName, int opIndex, llvm::StringRef newOpType);
};

} // namespace tensorlang
} // namespace mlir

#endif // TENSORLANG_RUNTIME_ASTMUTATOR_H
