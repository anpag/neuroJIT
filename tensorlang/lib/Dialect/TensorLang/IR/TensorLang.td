#ifndef TENSORLANG_DIALECT_TD
#define TENSORLANG_DIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Dialect Definition
//===----------------------------------------------------------------------===//

def TensorLang_Dialect : Dialect {
  let name = "tensorlang";
  let summary = "A tensor-native dialect with linear and dependent types";
  let description = [{
    The TensorLang dialect is designed for high-performance AI workloads.
    It features first-class support for linear types (for deterministic memory management)
    and symbolic shapes (for dynamic LLM sequence lengths).
  }];
  let cppNamespace = "::mlir::tensorlang";
  let useDefaultTypePrinterParser = 1;
}

// Base class for TensorLang operations.
class TensorLang_Op<string mnemonic, list<Trait> traits = []>
    : Op<TensorLang_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Type Definitions
//===----------------------------------------------------------------------===//

class TensorLang_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<TensorLang_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def LinearTensorType : TensorLang_Type<"LinearTensor", "tensor"> {
  let summary = "A tensor type with linear usage semantics";
  let description = [{
    A tensor type that carries linearity information.
  }];
  
  let parameters = (ins
    "Type":$elementType,
    ArrayRefParameter<"int64_t">:$shape
  );

  let assemblyFormat = "`<` $elementType `,` $shape `>`";
  
  let genVerifyDecl = 1;
}

// Constraint to accept either built-in tensors or our LinearTensor
def TL_AnyTensor : TypeConstraint<Or<[
  IsTensorTypePred, 
  CPred<"isa<::mlir::tensorlang::LinearTensorType>($_self)">
]>, "any tensor type">;

//===----------------------------------------------------------------------===//
// Operations
//===----------------------------------------------------------------------===//

def ConstantOp : TensorLang_Op<"constant", [Pure]> {
  let summary = "Creates a constant tensor.";
  let description = [{
    Creates a constant tensor from a dense attribute.
  }];

  let arguments = (ins ElementsAttr:$value);
  let results = (outs TL_AnyTensor:$output);

  let assemblyFormat = "attr-dict $value `:` type($output)";
}

def SymbolicDimOp : TensorLang_Op<"symbolic_dim", [Pure]> {
  let summary = "Defines a symbolic dimension variable";
  let description = [{
    Defines a symbolic dimension (e.g., 'batch_size' or 'seq_len') that can be 
    referenced by tensor types. Essential for compiling LLMs where sequence length varies.
  }];
  
  let arguments = (ins StrAttr:$name);
  let results = (outs Index:$dim);
  let assemblyFormat = "$name attr-dict";
}

def MatMulOp : TensorLang_Op<"matmul", [Pure]> {
  let summary = "Matrix multiplication";
  let description = [{
    Performs matrix multiplication.
    Supports both built-in tensors and LinearTensors.
    Verifies that inner dimensions match (including symbolic checks).
  }];

  let arguments = (ins TL_AnyTensor:$lhs, TL_AnyTensor:$rhs);
  let results = (outs TL_AnyTensor:$result);
  
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
  
  let hasVerifier = 1;
}

def PrintOp : TensorLang_Op<"print", []> {
  let summary = "Prints a tensor to standard output.";
  let arguments = (ins TL_AnyTensor:$input);
  let assemblyFormat = "$input attr-dict `:` type($input)";
}

#endif // TENSORLANG_DIALECT_TD
