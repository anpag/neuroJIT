#ifndef TENSORLANG_RUNTIME_VERIFICATIONSANDBOX_H
#define TENSORLANG_RUNTIME_VERIFICATIONSANDBOX_H

#include <string>
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace tensorlang {

class IEvaluator {
public:
  virtual ~IEvaluator() = default;
  // Takes a compiled function pointer and returns a fitness score (higher is better)
  virtual float evaluate(void* functionPointer) = 0;
};

class VerificationSandbox {
public:
  enum class VerificationResult {
    Success,
    CompileFailed,
    SemanticFailed,
    ExecutionFailed
  };

  struct Result {
    VerificationResult status;
    float fitnessScore;
  };

  /// Compiles the candidate IR in a completely isolated JIT environment.
  /// Executes semantic checks on the function logic using the provided evaluator.
  /// Returns the specific result of the verification and a fitness score.
  static Result verifyCandidate(const std::string& candidateIR, llvm::StringRef entryPoint, IEvaluator* evaluator);
};

} // namespace tensorlang
} // namespace mlir

#endif // TENSORLANG_RUNTIME_VERIFICATIONSANDBOX_H
