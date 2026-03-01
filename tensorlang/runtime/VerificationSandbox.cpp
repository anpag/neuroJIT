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