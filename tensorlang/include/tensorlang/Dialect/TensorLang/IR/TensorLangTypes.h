#ifndef TENSORLANG_DIALECT_TENSORLANG_IR_TENSORLANGTYPES_H_
#define TENSORLANG_DIALECT_TENSORLANG_IR_TENSORLANGTYPES_H_

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/TypeSwitch.h"

#define GET_TYPEDEF_CLASSES
#include "TensorLangTypes.h.inc"

#endif // TENSORLANG_DIALECT_TENSORLANG_IR_TENSORLANGTYPES_H_
