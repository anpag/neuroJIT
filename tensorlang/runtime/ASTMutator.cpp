#include "tensorlang/Runtime/ASTMutator.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OperationSupport.h"

namespace mlir {
namespace tensorlang {

std::optional<mlir::ModuleOp> ASTMutator::mutateConstant(mlir::ModuleOp baseModule, llvm::StringRef funcName, int constantIndex, float newValue) {
  mlir::ModuleOp clonedModule = baseModule.clone();
  
  auto funcOp = clonedModule.lookupSymbol<mlir::func::FuncOp>(funcName);
  if (!funcOp) {
    clonedModule.erase();
    return std::nullopt;
  }

  int currentIndex = 0;
  mlir::arith::ConstantOp targetOp;
  
  funcOp.walk([&](mlir::arith::ConstantOp op) {
    // Only count float constants for simplicity, based on newValue being float
    if (mlir::isa<mlir::FloatType>(op.getType())) {
      if (currentIndex == constantIndex) {
        targetOp = op;
      }
      currentIndex++;
    }
  });

  if (!targetOp) {
    clonedModule.erase();
    return std::nullopt;
  }

  mlir::OpBuilder builder(targetOp);
  auto newConst = builder.create<mlir::arith::ConstantOp>(
      targetOp.getLoc(), targetOp.getType(), builder.getF32FloatAttr(newValue));
  
  targetOp.replaceAllUsesWith(newConst.getResult());
  targetOp.erase();

  return clonedModule;
}

std::optional<mlir::ModuleOp> ASTMutator::swapBinaryOperator(mlir::ModuleOp baseModule, llvm::StringRef funcName, int opIndex, llvm::StringRef newOpType) {
  mlir::ModuleOp clonedModule = baseModule.clone();
  
  auto funcOp = clonedModule.lookupSymbol<mlir::func::FuncOp>(funcName);
  if (!funcOp) {
    clonedModule.erase();
    return std::nullopt;
  }

  int currentIndex = 0;
  mlir::Operation* targetOp = nullptr;
  
  funcOp.walk([&](mlir::Operation* op) {
    // Only target ops that have 2 operands and 1 result (likely binary ops)
    if (op->getNumOperands() == 2 && op->getNumResults() == 1 &&
        op->getName().getStringRef().starts_with("arith.") &&
        !llvm::isa<mlir::arith::CmpFOp, mlir::arith::CmpIOp>(op)) {
      if (currentIndex == opIndex) {
        targetOp = op;
      }
      currentIndex++;
    }
  });

  if (!targetOp) {
    clonedModule.erase();
    return std::nullopt;
  }

  mlir::OpBuilder builder(targetOp);
  mlir::OperationState state(targetOp->getLoc(), newOpType);
  state.addOperands(targetOp->getOperands());
  state.addTypes(targetOp->getResultTypes());
  
  // Ensure the new operation name is registered/valid
  mlir::OperationName opName(newOpType, clonedModule.getContext());
  if (!opName.isRegistered()) {
    clonedModule.erase();
    return std::nullopt;
  }

  mlir::Operation* newOp = builder.create(state);
  
  targetOp->replaceAllUsesWith(newOp);
  targetOp->erase();

  return clonedModule;
}

} // namespace tensorlang
} // namespace mlir
