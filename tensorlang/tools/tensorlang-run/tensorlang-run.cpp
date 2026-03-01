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
    "max-restarts", llvm::cl::desc("Maximum simulation restarts (default 5)"),
    llvm::cl::init(5));

// ---------------------------------------------------------------------------
// Helper: parse a module from a string
// ---------------------------------------------------------------------------
[[maybe_unused]] static mlir::OwningOpRef<mlir::ModuleOp>
parseModuleFromString(mlir::MLIRContext& ctx, const std::string& ir) {
  llvm::SourceMgr sm;
  sm.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBuffer(ir), llvm::SMLoc());
  return mlir::parseSourceFile<mlir::ModuleOp>(sm, &ctx);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv, "NeuroJIT V2\n");

  // Register dialects
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

  // Parse input file
  std::string errorMessage;
  auto file = mlir::openInputFile(inputFilename, &errorMessage);
  if (!file) { llvm::errs() << errorMessage << "\n"; return 1; }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(file), llvm::SMLoc());
  auto module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  if (!module) { llvm::errs() << "Failed to parse input\n"; return 1; }

  // Store module IR for reflection
  std::string ir;
  { llvm::raw_string_ostream os(ir); module->print(os); }
  JitContext::getInstance().setModuleIR(ir);

  // Initialize model runner
  llvm::outs() << "[NeuroJIT V2] Runner: " << runnerType << "\n";
  auto runnerImpl = ModelRunner::create(runnerType);
  if ((runnerType == "llama" || runnerType == "gemini") &&
      runnerImpl->load(modelPath) != 0) {
    llvm::errs() << "[NeuroJIT V2] Model load failed — using mock\n";
    runnerImpl = ModelRunner::create("mock");
  }
  JitContext::getInstance().setModelRunner(std::move(runnerImpl));

  // Seed GA from registry if a previous best exists
  auto& ctx = JitContext::getInstance();
  if (ctx.hasLobe("Stability_v1")) {
    auto lobeResult = ctx.loadLobeResult("Stability_v1");
    if (lobeResult.survived) {
      printf("[NeuroJIT V2] Seeding GA from registry (score=%.2f)\n",
             lobeResult.score());
      // Reconstruct a ControlStrategy from saved result metadata
      // (In a fuller implementation, the JSON strategy is stored alongside the IR)
      // For now, seed with random population biased around known-good defaults
      ControlStrategy seed;
      seed.kp = 2.0f; seed.kd = 1.5f;
      seed.targetVelocity = -1.0f; seed.thrustClampMax = 4.0f;
      ctx.getGA().seed(seed);
    }
  }

  // Create JIT runner
  auto jitRunnerOrErr = JitRunner::create();
  if (auto err = jitRunnerOrErr.takeError()) {
    llvm::errs() << "JIT create failed: " << err << "\n";
    return 1;
  }
  auto jitRunner = std::move(*jitRunnerOrErr);
  JitContext::getInstance().registerRunner(jitRunner.get());

  // ---------------------------------------------------------------------------
  // Main execution loop — clean restarts, no longjmp
  // ---------------------------------------------------------------------------
  int restartCount = 0;
  int exitCode = 0;

  // Compile and run initial module
  if (auto err = jitRunner->run(*module)) {
    llvm::errs() << "Initial run failed: " << llvm::toString(std::move(err))
                 << "\n";
    // Don't exit — check for restart request first
  }

  // Restart loop: handles assert-triggered restarts cleanly
  while (restartCount < maxRestarts) {
    std::string newIR;
    if (!JitContext::getInstance().consumeRestartRequest(newIR)) {
      break; // No restart requested — normal completion
    }

    restartCount++;
    printf("[NeuroJIT V2] Restart %d/%d with new IR (%zu bytes)\n",
           restartCount, (int)maxRestarts, newIR.size());

    // Compile the new IR
    if (tensorlang_compile(newIR.c_str()) != 0) {
      llvm::errs() << "[NeuroJIT V2] Restart IR failed to compile\n";
      continue;
    }

    // Re-invoke main from the newly compiled module
    auto result = jitRunner->invoke("main");
    if (!result) {
      llvm::errs() << "[NeuroJIT V2] Restart invoke failed: "
                   << llvm::toString(result.takeError()) << "\n";
      exitCode = 1;
      break;
    }
    exitCode = result.get();
  }

  if (restartCount >= maxRestarts) {
    llvm::errs() << "[NeuroJIT V2] Max restarts reached. Exiting.\n";
    exitCode = 1;
  }

  printf("[NeuroJIT V2] Final best score: %.2f\n", ctx.getBestScore());
  printf("[NeuroJIT V2] GA generation: %d\n", ctx.getGA().generation());
  return exitCode;
}