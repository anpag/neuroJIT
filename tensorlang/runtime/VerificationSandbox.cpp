#include "tensorlang/Runtime/VerificationSandbox.h"
#include "tensorlang/ExecutionEngine/JitRunner.h"
#include "tensorlang/Runtime/JitContext.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace tensorlang {

VerificationSandbox::VerificationResult VerificationSandbox::verifyCandidate(const std::string& candidateIR) {
  printf("[Sandbox] Initializing completely isolated JIT environment...\n");

  auto evalRunnerOrErr = JitRunner::create();
  if (auto err = evalRunnerOrErr.takeError()) {
    fprintf(stderr, "[Sandbox] Failed to create isolated JIT\n");
    llvm::consumeError(std::move(err));
    return VerificationResult::ExecutionFailed;
  }
  auto evalRunner = std::move(*evalRunnerOrErr);

  printf("[Sandbox] Compiling candidate MLIR...\n");
  if (auto err = evalRunner->loadString(candidateIR)) {
    fprintf(stderr, "[Sandbox] Compilation failed (LLM hallucinated invalid MLIR)\n");
    llvm::consumeError(std::move(err));
    return VerificationResult::CompileFailed;
  }

  printf("[Sandbox] Executing semantic checks on get_thrust...\n");
  auto sym = evalRunner->lookup("_mlir_ciface_get_thrust");
  if (!sym) {
    llvm::consumeError(sym.takeError());
    sym = evalRunner->lookup("get_thrust");
    if (!sym) {
      llvm::Error lookupErr = sym.takeError();
      fprintf(stderr, "[Sandbox] Missing get_thrust symbol. Error: %s\n", llvm::toString(std::move(lookupErr)).c_str());
      return VerificationResult::CompileFailed; // Link failure counts as compile failure here
    }
  }

  auto get_thrust_fn = reinterpret_cast<float(*)(float, float)>(sym.get());

  // Semantic Oracle: Thrust must be >= 0 and reasonable for basic states
  float test1 = get_thrust_fn(100.0f, -10.0f); // High up, moving down fast
  float test2 = get_thrust_fn(5.0f, -2.0f);    // Near ground, moving down slowly
  float test3 = get_thrust_fn(0.0f, 0.0f);     // On ground, stopped

  if (test1 < 0.0f || test2 < 0.0f || test3 < 0.0f) {
    fprintf(stderr, "[Sandbox] Semantic failure: Output negative thrust.\n");
    return VerificationResult::SemanticFailed;
  }
  if (test1 > 20.0f || test2 > 20.0f || test3 > 20.0f) {
    fprintf(stderr, "[Sandbox] Semantic failure: Engine max capacity exceeded.\n");
    return VerificationResult::SemanticFailed;
  }
  if (test1 == test2 && test2 == test3) {
    fprintf(stderr, "[Sandbox] Semantic failure: Constant output, function ignores inputs.\n");
    return VerificationResult::SemanticFailed;
  }
  if (test1 <= test2) {
      printf("[Sandbox] Warning: Unintuitive thrust curve (test1=%f, test2=%f)\n", test1, test2);
  }

  printf("[Sandbox] VERIFICATION SUCCESS! Candidate passes semantic constraints.\n");
  return VerificationResult::Success;
}

} // namespace tensorlang
} // namespace mlir