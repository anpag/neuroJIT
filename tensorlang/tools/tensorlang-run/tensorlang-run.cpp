#include "tensorlang/ExecutionEngine/JitRunner.h"
#include "tensorlang/Dialect/TensorLang/IR/TensorLangDialect.h"
#include "tensorlang/Runtime/Runtime.h"
#include "tensorlang/Runtime/JitContext.h"
#include "tensorlang/Runtime/ModelRunner.h"

#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/InitLLVM.h"

// Bufferization Interfaces
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Vector/Transforms/BufferizableOpInterfaceImpl.h"

using namespace mlir;
using namespace mlir::tensorlang;
using namespace llvm;

static cl::opt<std::string> inputFilename(cl::Positional, cl::desc("<input file>"),
                                          cl::init("-"));

int main(int argc, char **argv) {
  if (argc > 1000) tensorlang_get_ir();
  InitLLVM y(argc, argv);
  cl::ParseCommandLineOptions(argc, argv, "TensorLang execution tool\n");

  DialectRegistry registry;
  registerBuiltinDialectTranslation(registry);
  registerLLVMDialectTranslation(registry);

  // Register BufferizableOpInterface external models
  mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::vector::registerBufferizableOpInterfaceExternalModels(registry);

  MLIRContext context(registry);
  context.getOrLoadDialect<TensorLangDialect>();
  context.getOrLoadDialect<func::FuncDialect>();
  context.getOrLoadDialect<LLVM::LLVMDialect>();
  context.getOrLoadDialect<arith::ArithDialect>();
  context.getOrLoadDialect<tensor::TensorDialect>();
  context.getOrLoadDialect<scf::SCFDialect>();

  // Parse input
  std::string errorMessage;
  std::unique_ptr<MemoryBuffer> file = openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(file), SMLoc());
  OwningOpRef<ModuleOp> module = parseSourceFile<ModuleOp>(sourceMgr, &context);
  if (!module)
    return 1;

  std::string ir;
  llvm::raw_string_ostream os(ir);
  module->print(os);
  mlir::tensorlang::JitContext::getInstance().setModuleIR(ir);
  
  // Default to Gemini as requested
  llvm::outs() << "[NeuroJIT] Using Runner: gemini\n";
  
  mlir::tensorlang::JitContext::getInstance().setModelRunner(
      mlir::tensorlang::ModelRunner::create("gemini"));

  // Create JIT Runner
  auto runnerOrError = JitRunner::create();
  if (auto err = runnerOrError.takeError()) {
    llvm::errs() << "Failed to create JIT runner: " << err << "\n";
    return 1;
  }
  auto runner = std::move(*runnerOrError);
  mlir::tensorlang::JitContext::getInstance().registerRunner(runner.get());

  // Execute
  auto& jitCtx = mlir::tensorlang::JitContext::getInstance();
  if (setjmp(jitCtx.getRecoveryPoint()) != 0) {
    llvm::errs() << "[NeuroJIT] Recovery point reached. Restarting simulation...\n";
    auto res = runner->invoke("main");
    if (!res) {
      llvm::errs() << "Restart failed: " << llvm::toString(res.takeError()) << "\n";
      return 1;
    }
    return res.get(); // Return the exit code
  }

  if (auto err = runner->run(*module)) {
    llvm::errs() << "Execution failed: " << err << "\n";
    return 1;
  }

  return 0;
}
