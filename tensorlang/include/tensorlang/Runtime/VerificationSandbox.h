#ifndef TENSORLANG_RUNTIME_VERIFICATIONSANDBOX_H
#define TENSORLANG_RUNTIME_VERIFICATIONSANDBOX_H

#include <string>

namespace mlir {
namespace tensorlang {

class VerificationSandbox {
public:
  /// Compiles the candidate IR in a completely isolated JIT environment.
  /// Executes the 'sim_main' function to verify if the logic is safe.
  /// Returns true if the episode completes without tripping any assertions.
  static bool verifyCandidate(const std::string& candidateIR);
};

} // namespace tensorlang
} // namespace mlir

#endif // TENSORLANG_RUNTIME_VERIFICATIONSANDBOX_H