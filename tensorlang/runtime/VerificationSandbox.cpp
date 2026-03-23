#include "tensorlang/Runtime/VerificationSandbox.h"
#include "tensorlang/ExecutionEngine/JitRunner.h"
#include "tensorlang/Runtime/JitContext.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace tensorlang {

VerificationSandbox::Result VerificationSandbox::verifyCandidate(const std::string& candidateIR, llvm::StringRef entryPoint, IEvaluator* evaluator) {
  printf("[Sandbox] Initializing completely isolated JIT environment...\n");

  auto evalRunnerOrErr = JitRunner::create();
  if (auto err = evalRunnerOrErr.takeError()) {
    fprintf(stderr, "[Sandbox] Failed to create isolated JIT\n");
    llvm::consumeError(std::move(err));
    return {VerificationResult::ExecutionFailed, -10000.0f};
  }
  auto evalRunner = std::move(*evalRunnerOrErr);

  printf("[Sandbox] Compiling candidate MLIR...\n");
  if (auto err = evalRunner->loadString(candidateIR)) {
    fprintf(stderr, "[Sandbox] Compilation failed (invalid MLIR)\n");
    llvm::consumeError(std::move(err));
    return {VerificationResult::CompileFailed, -10000.0f};
  }

  std::string cifaceName = "_mlir_ciface_" + entryPoint.str();
  auto sym = evalRunner->lookup(cifaceName);
  if (!sym) {
    llvm::consumeError(sym.takeError());
    sym = evalRunner->lookup(entryPoint);
    if (!sym) {
      llvm::Error lookupErr = sym.takeError();
      fprintf(stderr, "[Sandbox] Missing symbol '%s'. Error: %s\n", entryPoint.str().c_str(), llvm::toString(std::move(lookupErr)).c_str());
      return {VerificationResult::CompileFailed, -10000.0f};
    }
  }

  void* fnPtr = *sym;
  if (!fnPtr) {
    fprintf(stderr, "[Sandbox] Symbol resolved to null pointer.\n");
    return {VerificationResult::ExecutionFailed, -10000.0f};
  }

  printf("[Sandbox] Executing semantic checks via IEvaluator...\n");
  float fitness = evaluator->evaluate(fnPtr);

  printf("[Sandbox] VERIFICATION SUCCESS! Candidate passes compilation. Fitness: %f\n", fitness);
  return {VerificationResult::Success, fitness};
}

} // namespace tensorlang
} // namespace mlir
