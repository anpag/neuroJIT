#include "tensorlang/Dialect/TensorLang/Transforms/Passes.h"
#include "tensorlang/Dialect/TensorLang/IR/TensorLangOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;
using namespace mlir::tensorlang;

namespace {

struct VerifyLinearity : public PassWrapper<VerifyLinearity, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VerifyLinearity)

  StringRef getArgument() const override { return "verify-linearity"; }
  StringRef getDescription() const override { return "Verifies linear type semantics"; }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    bool hasError = false;

    auto checkLinearValue = [&](Value value, Operation *sourceOp) {
      if (!isa<LinearTensorType>(value.getType()))
        return;

      if (!value.hasOneUse()) {
        sourceOp->emitError("linear value must be used exactly once");
        hasError = true;
        return;
      }

      // Check if the single use is inside a loop
      OpOperand &use = *value.getUses().begin();
      Operation *user = use.getOwner();
      
      // Walk up the parent operations to see if we are inside a loop
      Operation *parent = user->getParentOp();
      while (parent && !isa<func::FuncOp>(parent)) {
        if (isa<scf::ForOp, scf::WhileOp, scf::ParallelOp>(parent)) {
          user->emitError("linear value cannot be used inside a loop body");
          hasError = true;
          break;
        }
        parent = parent->getParentOp();
      }
    };

    module.walk([&](Operation *op) {
      // Check results
      for (Value result : op->getResults()) {
        checkLinearValue(result, op);
      }
      
      // Check block arguments
      for (Region &region : op->getRegions()) {
        for (Block &block : region.getBlocks()) {
          for (BlockArgument arg : block.getArguments()) {
            checkLinearValue(arg, op);
          }
        }
      }
    });

    if (hasError)
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::tensorlang::createVerifyLinearity() {
  return std::make_unique<VerifyLinearity>();
}
