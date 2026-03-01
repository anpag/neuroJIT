#ifndef TENSORLANG_RUNTIME_VERIFICATIONSANDBOX_H
#define TENSORLANG_RUNTIME_VERIFICATIONSANDBOX_H

#include <string>

namespace mlir {
namespace tensorlang {

class VerificationSandbox {
public:
  enum class VerificationResult {
    Success,
    CompileFailed,
    SemanticFailed,
    ExecutionFailed
  };

  /// Compiles the candidate IR in a completely isolated JIT environment.
  /// Executes semantic checks on the function logic.
  /// Returns the specific result of the verification.
  static VerificationResult verifyCandidate(const std::string& candidateIR);
};

} // namespace tensorlang
} // namespace mlir

#endif // TENSORLANG_RUNTIME_VERIFICATIONSANDBOX_H