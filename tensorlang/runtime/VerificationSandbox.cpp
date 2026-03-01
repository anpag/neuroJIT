#include "tensorlang/Runtime/VerificationSandbox.h"
#include "tensorlang/ExecutionEngine/JitRunner.h"
#include "tensorlang/Runtime/JitContext.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace tensorlang {

bool VerificationSandbox::verifyCandidate(const std::string& candidateIR) {
  printf("[Sandbox] Initializing isolated JIT environment...\n");

  auto evalRunnerOrErr = JitRunner::create();
  if (auto err = evalRunnerOrErr.takeError()) {
    fprintf(stderr, "[Sandbox] Failed to create isolated JIT\n");
    llvm::consumeError(std::move(err));
    return false;
  }
  auto evalRunner = std::move(*evalRunnerOrErr);

  // Temporarily swap the active runner in context so the C++ hooks talk to the sandbox
  auto& ctx = JitContext::getInstance();
  JitRunner* oldRunner = ctx.getRunner();
  ctx.registerRunner(evalRunner.get());

  printf("[Sandbox] Compiling candidate MLIR...\n");
  if (auto err = evalRunner->compileString(candidateIR)) {
    fprintf(stderr, "[Sandbox] Compilation failed (LLM hallucinated invalid MLIR)\n");
    llvm::consumeError(std::move(err));
    ctx.registerRunner(oldRunner);
    return false;
  }

  printf("[Sandbox] Executing semantic checks on get_thrust...\n");
  auto sym = evalRunner->lookup("get_thrust");
  if (!sym) {
    fprintf(stderr, "[Sandbox] Missing get_thrust symbol.\n");
    llvm::consumeError(sym.takeError());
    ctx.registerRunner(oldRunner);
    return false;
  }
  
  auto get_thrust_fn = reinterpret_cast<float(*)(float, float)>(sym.get());
  
  // Semantic Oracle: Thrust must be >= 0 and reasonable for basic states
  float test1 = get_thrust_fn(100.0f, -10.0f); // High up, moving down fast
  float test2 = get_thrust_fn(5.0f, -2.0f);    // Near ground, moving down slowly
  float test3 = get_thrust_fn(0.0f, 0.0f);     // On ground, stopped

  if (test1 < 0.0f || test2 < 0.0f || test3 < 0.0f) {
    fprintf(stderr, "[Sandbox] Semantic failure: Output negative thrust.\n");
    ctx.registerRunner(oldRunner);
    return false;
  }
  if (test1 > 20.0f || test2 > 20.0f || test3 > 20.0f) {
    fprintf(stderr, "[Sandbox] Semantic failure: Engine max capacity exceeded.\n");
    ctx.registerRunner(oldRunner);
    return false;
  }
  if (test1 <= test2) {
      // In a real controller, thrust should generally be higher when moving fast downwards
      // than when moving slow near the ground. We won't strictly fail on this, but it's a good check.
      printf("[Sandbox] Warning: Unintuitive thrust curve (test1=%f, test2=%f)\n", test1, test2);
  }

  printf("[Sandbox] Executing verification episode...\n");
  auto result = evalRunner->invoke("sim_main");

  // Check if the runtime triggered a crash during execution
  std::string dummy;
  bool crashed = ctx.consumeRestartRequest(dummy);

  // Restore the real runtime environment
  ctx.registerRunner(oldRunner);

  if (!result) {
    fprintf(stderr, "[Sandbox] Execution failed or returned error code.\n");
    llvm::consumeError(result.takeError());
    return false;
  }

  if (crashed) {
    printf("[Sandbox] Candidate logic tripped an assertion (Failed safety check).\n");
    return false;
  }

  printf("[Sandbox] VERIFICATION SUCCESS! Candidate is safe for production.\n");
  return true;
}

} // namespace tensorlang
} // namespace mlir