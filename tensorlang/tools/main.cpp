#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/InitAllDialects.h" 
#include "mlir/InitAllPasses.h"
#include "tensorlang/Dialect/TensorLang/IR/TensorLangDialect.h"
#include "tensorlang/Conversion/TensorLangToLinalg/TensorLangToLinalg.h"
#include "tensorlang/Dialect/TensorLang/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Vector/Transforms/BufferizableOpInterfaceImpl.h"

// Include headers for other dialects we might need for testing
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  
  // Register only necessary dialects
  registry.insert<mlir::tensorlang::TensorLangDialect>();
  registry.insert<mlir::linalg::LinalgDialect>();
  registry.insert<mlir::tensor::TensorDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::scf::SCFDialect>();
  registry.insert<mlir::LLVM::LLVMDialect>();
  registry.insert<mlir::vector::VectorDialect>();
  
  // Register BufferizableOpInterface external models
  mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::vector::registerBufferizableOpInterfaceExternalModels(registry);

  // Register TensorLang passes
  mlir::tensorlang::registerConvertTensorLangToLinalgPass();
  mlir::tensorlang::registerVerifyLinearityPass();
  
  mlir::bufferization::registerBufferizationPasses();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "TensorLang Optimizer Driver\n", registry));
}
