#ifndef TENSORLANG_RUNTIME_EVOLUTIONHARNESS_H
#define TENSORLANG_RUNTIME_EVOLUTIONHARNESS_H

#include "tensorlang/Runtime/JitContext.h"
#include "tensorlang/Runtime/OptimizationStrategy.h"
#include "tensorlang/Runtime/MLIRTemplates.h"
#include <string>

namespace mlir {
class MLIRContext;
namespace tensorlang {

/// Configuration for the evolution loop.
struct HarnessConfig {
  int maxGenerations = 50;
  int episodesPerGeneration = 1;   // Evaluate each strategy N times
  double targetScore = 80.0;       // Stop early if this score is reached
  bool verbose = true;
  std::string logFile = "";        // Empty = stdout only
};

/// Runs the full evolution loop:
/// 1. Propose strategy from GA
/// 2. Instantiate MLIR from template
/// 3. Compile and hot-swap
/// 4. Run simulation episode
/// 5. Record fitness
/// 6. Repeat
///
/// Returns the best score achieved.
double runEvolutionLoop(JitRunner* runner,
                        MLIRContext* context,
                        const std::string& baseModuleIR,
                        const HarnessConfig& config);

} // namespace tensorlang
} // namespace mlir

#endif // TENSORLANG_RUNTIME_EVOLUTIONHARNESS_H