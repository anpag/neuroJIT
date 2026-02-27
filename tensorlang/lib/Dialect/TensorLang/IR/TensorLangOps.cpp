#include "tensorlang/Dialect/TensorLang/IR/TensorLangOps.h"
#include "tensorlang/Dialect/TensorLang/IR/TensorLangDialect.h"
#include "tensorlang/Dialect/TensorLang/IR/TensorLangTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;
using namespace mlir::tensorlang;

//===----------------------------------------------------------------------===//
// LinearTensorType
//===----------------------------------------------------------------------===//

LogicalResult LinearTensorType::verify(function_ref<InFlightDiagnostic()> emitError,
                                       Type elementType,
                                       ArrayRef<int64_t> shape) {
  if (!elementType.isIntOrIndexOrFloat()) {
    return emitError() << "linear tensor requires int, index, or float element type";
  }

  for (int64_t dim : shape) {
    if (dim < 0 && dim != -1) {
      return emitError() << "tensor dimension must be non-negative or -1";
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// MatMulOp
//===----------------------------------------------------------------------===//

LogicalResult MatMulOp::verify() {
  auto lhsType = getLhs().getType();
  auto rhsType = getRhs().getType();

  // Helper to get shape and rank
  auto getShape = [](Type t) -> ArrayRef<int64_t> {
    if (auto tensor = dyn_cast<RankedTensorType>(t)) return tensor.getShape();
    if (auto linear = dyn_cast<LinearTensorType>(t)) return linear.getShape();
    return {};
  };

  ArrayRef<int64_t> lhsShape = getShape(lhsType);
  ArrayRef<int64_t> rhsShape = getShape(rhsType);
  
  if (lhsShape.empty() || rhsShape.empty()) {
     return emitOpError("operands must be ranked tensors");
  }

  if (lhsShape.size() != 2 || rhsShape.size() != 2) {
      // For simplicity, enforce 2D for now.
      return emitOpError("operands must be 2D tensors for matmul");
  }

  // Check inner dimensions: (M x K) * (K x N) -> (M x N)
  int64_t K_lhs = lhsShape[1];
  int64_t K_rhs = rhsShape[0];

  if (K_lhs != -1 && K_rhs != -1 && K_lhs != K_rhs) {
    return emitOpError("inner dimensions mismatch: ") << K_lhs << " != " << K_rhs;
  }
  
  // TODO: Check symbolic dimensions if available
  
  return success();
}

//===----------------------------------------------------------------------===//
// TableGen Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "TensorLang.cpp.inc"
