#ifndef TENSORLANG_RUNTIME_OPTIMIZATIONSTRATEGY_H
#define TENSORLANG_RUNTIME_OPTIMIZATIONSTRATEGY_H

#include <string>
#include <vector>
#include <variant>
#include <cmath>

namespace mlir {
namespace tensorlang {

/// Parameters for a PD/PID control strategy.
/// All fields are normalized to float to simplify JSON parsing and GA operations.
struct ControlStrategy {
  float kp = 1.0f;
  float ki = 0.0f;
  float kd = 0.0f;
  float targetVelocity = -1.0f;
  float thrustClampMin = 0.0f;
  float thrustClampMax = 5.0f;
};

/// Parameters for a tiling/vectorization strategy applied to linalg ops.
struct TilingStrategy {
  std::string targetFunction;
  std::vector<int64_t> tileSizes;
  bool vectorize = false;
  int unrollFactor = 1;
};

/// Discriminated union of all strategy types.
using OptimizationStrategy = std::variant<ControlStrategy, TilingStrategy>;

/// Quantitative result from a single simulation episode.
/// This is the fitness function — without this, no evolution claim is valid.
struct SimulationResult {
  double impactVelocity = -999.0; ///< m/s at ground contact. Target: > -1.0
  double fuelConsumed = 999.0;    ///< Arbitrary units. Lower is better.
  double settlingTime = 999.0;    ///< Simulation ticks to reach |v| < 1.0
  bool survived = false;          ///< Completed without assert violation

  /// Scalar fitness. Higher is better. Negative means crash.
  double score() const {
    if (!survived) return -1000.0;
    return 100.0
        - std::abs(impactVelocity) * 10.0
        - fuelConsumed * 0.1
        - settlingTime * 0.2;
  }
};

/// A fully evaluated individual for the GA population.
struct Individual {
  ControlStrategy strategy;
  SimulationResult result;
  int generation = 0;
};

} // namespace tensorlang
} // namespace mlir

#endif // TENSORLANG_RUNTIME_OPTIMIZATIONSTRATEGY_H