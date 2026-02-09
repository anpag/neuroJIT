#include "tensorlang/Dialect/TensorLang/Transforms/Passes.h"
#include "tensorlang/Dialect/TensorLang/IR/TensorLangOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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

    module.walk([&](Operation *op) {
      // Check results
      for (Value result : op->getResults()) {
        if (isa<LinearTensorType>(result.getType())) {
          if (!result.hasOneUse()) {
            op->emitOpError("result with linear type must be used exactly once");
            hasError = true;
          }
        }
      }
      
      // Check block arguments
      for (Region &region : op->getRegions()) {
        for (Block &block : region.getBlocks()) {
          for (BlockArgument arg : block.getArguments()) {
             if (isa<LinearTensorType>(arg.getType())) {
               if (!arg.hasOneUse()) {
                 // Use operation location
                 op->emitError("block argument with linear type must be used exactly once");
                 hasError = true;
               }
             }
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
