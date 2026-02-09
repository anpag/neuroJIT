#ifndef TENSORLANG_TRANSFORMS_PASSES_TD
#define TENSORLANG_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def VerifyLinearity : Pass<"verify-linearity", "ModuleOp"> {
  let summary = "Verifies linear type semantics";
  let description = [{
    Checks that values with LinearTensorType are consumed exactly once.
  }];
}

#endif // TENSORLANG_TRANSFORMS_PASSES_TD
