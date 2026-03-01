#include "tensorlang/ExecutionEngine/JitRunner.h"
#include "tensorlang/Dialect/TensorLang/IR/TensorLangDialect.h"
#include "tensorlang/Runtime/Runtime.h"
#include "tensorlang/Runtime/JitContext.h"
#include "tensorlang/Runtime/ModelRunner.h"

#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Vector/Transforms/BufferizableOpInterfaceImpl.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/InitLLVM.h"

using namespace mlir;
using namespace mlir::tensorlang;

static llvm::cl::opt<std::string> inputFilename(
    llvm::cl::Positional, llvm::cl::desc("<input file>"), llvm::cl::init("-"));
static llvm::cl::opt<std::string> runnerType(
    "runner", llvm::cl::desc("LLM runner (mock, gemini, llama)"),
    llvm::cl::init("llama"));
static llvm::cl::opt<std::string> modelPath(
    "model", llvm::cl::desc("Path to GGUF model"),
    llvm::cl::init("tensorlang/runtime/models/"
                   "qwen2.5-coder-7b-instruct-q4_k_m.gguf"));
static llvm::cl::opt<int> maxRestarts(
    "max-restarts", llvm::cl::desc("Maximum safe restarts during execution (default 5)"),
    llvm::cl::init(5));

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv, "NeuroJIT AI-Guided JIT Compiler\n");

  mlir::DialectRegistry registry;
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
  mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);
  mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::vector::registerBufferizableOpInterfaceExternalModels(registry);

  mlir::MLIRContext context(registry);
  context.getOrLoadDialect<TensorLangDialect>();
  context.getOrLoadDialect<func::FuncDialect>();
  context.getOrLoadDialect<LLVM::LLVMDialect>();
  context.getOrLoadDialect<arith::ArithDialect>();
  context.getOrLoadDialect<tensor::TensorDialect>();
  context.getOrLoadDialect<scf::SCFDialect>();
  context.getOrLoadDialect<vector::VectorDialect>();
  context.getOrLoadDialect<math::MathDialect>();

  std::string errorMessage;
  auto file = mlir::openInputFile(inputFilename, &errorMessage);
  if (!file) { llvm::errs() << errorMessage << "\n"; return 1; }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(file), llvm::SMLoc());
  auto module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  if (!module) { llvm::errs() << "Failed to parse input\n"; return 1; }

  std::string ir;
  { llvm::raw_string_ostream os(ir); module->print(os); }
  JitContext::getInstance().setModuleIR(ir);

  llvm::outs() << "[NeuroJIT] Runner: " << runnerType << "\n";
  auto runnerImpl = ModelRunner::create(runnerType);
  if ((runnerType == "llama" || runnerType == "gemini") &&
      runnerImpl->load(modelPath) != 0) {
    llvm::errs() << "[NeuroJIT] Model load failed — using mock\n";
    runnerImpl = ModelRunner::create("mock");
  }
  JitContext::getInstance().setModelRunner(std::move(runnerImpl));

  auto jitRunnerOrErr = JitRunner::create();
  if (auto err = jitRunnerOrErr.takeError()) {
    llvm::errs() << "JIT create failed: " << err << "\n";
    return 1;
  }
  auto jitRunner = std::move(*jitRunnerOrErr);
  JitContext::getInstance().registerRunner(jitRunner.get());

  int exitCode = 0;

  // Initial Execution
  if (auto err = jitRunner->run(*module)) {
    llvm::errs() << "Initial run failed: " << llvm::toString(std::move(err)) << "\n";
  }

  // Safe Crash Recovery Loop (Driven by tensorlang_assert_fail triggering requestRestart)
  int restartCount = 0;
  while (restartCount < maxRestarts) {
    std::string newIR;
    if (!JitContext::getInstance().consumeRestartRequest(newIR)) {
      break; // Safe exit
    }

    restartCount++;
    printf("[NeuroJIT] Episode crashed. Restart %d/%d initiated.\n", restartCount, (int)maxRestarts);

    // Sleep briefly to let the background worker generate and compile the fix
    // (In Phase 6 we will implement a proper Verification Sandbox to block the episode until fixed)
    std::this_thread::sleep_for(std::chrono::seconds(2));

    if (tensorlang_compile(newIR.c_str()) != 0) {
       llvm::errs() << "[NeuroJIT] Fallback compilation failed, retrying...\n";
       continue;
    }

    auto result = jitRunner->invoke("sim_main");
    if (!result) {
      exitCode = 1;
      // Loop continues, allowing repeated self-healing attempts
    } else {
      exitCode = result.get();
    }
  }

  if (restartCount >= maxRestarts) {
    llvm::errs() << "[NeuroJIT] Max healing restarts reached. Manual intervention required.\n";
    exitCode = 1;
  } else {
    printf("[NeuroJIT] Execution completed gracefully.\n");
  }

  return exitCode;
}