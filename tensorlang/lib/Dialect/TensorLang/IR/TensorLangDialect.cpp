#include "tensorlang/Dialect/TensorLang/IR/TensorLangDialect.h"
#include "tensorlang/Dialect/TensorLang/IR/TensorLangOps.h"
#include "tensorlang/Dialect/TensorLang/IR/TensorLangTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "TensorLangDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "TensorLangTypes.cpp.inc"

using namespace mlir::tensorlang;

void TensorLangDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "TensorLang.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "TensorLangTypes.cpp.inc"
      >();
}
