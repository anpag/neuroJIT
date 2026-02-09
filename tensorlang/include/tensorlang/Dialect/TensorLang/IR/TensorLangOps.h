#ifndef TENSORLANG_DIALECT_TENSORLANG_IR_TENSORLANGOPS_H_
#define TENSORLANG_DIALECT_TENSORLANG_IR_TENSORLANGOPS_H_

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"

#include "tensorlang/Dialect/TensorLang/IR/TensorLangTypes.h"

#define GET_OP_CLASSES
#include "TensorLang.h.inc"

#endif // TENSORLANG_DIALECT_TENSORLANG_IR_TENSORLANGOPS_H_
