#ifndef TENSORLANG_RUNTIME_STRATEGYGA_H
#define TENSORLANG_RUNTIME_STRATEGYGA_H

#include "tensorlang/Runtime/OptimizationStrategy.h"
#include <vector>
#include <random>
#include <string>
#include <functional>

namespace mlir {
namespace tensorlang {

class ModelRunner;

/// Genetic algorithm operating on ControlStrategy parameter vectors.
/// The LLM is used ONLY as a mutation oracle when the GA plateaus.
/// All selection, crossover, and standard mutation are pure C++.
class StrategyGA {
public:
  static constexpr int kPopSize = 16;
  static constexpr int kEliteCount = 4;
  static constexpr int kPlateauGenerations = 5;

  explicit StrategyGA(unsigned seed = 42);

  /// Seed the initial population with a known-good individual (from registry)
  /// plus random perturbations.
  void seed(const ControlStrategy& known_good);

  /// Seed randomly if no prior knowledge exists.
  void seedRandom();

  /// Record the fitness of the most recently proposed individual.
  void recordFitness(const SimulationResult& result);

  /// Propose the next individual to evaluate.
  /// Internally handles selection, crossover, mutation, and LLM fallback.
  ControlStrategy proposeNext(ModelRunner* runner);

  /// Returns the best individual seen so far.
  const Individual& best() const;

  int generation() const { return generation_; }
  bool plateaued() const { return plateauCount_ >= kPlateauGenerations; }

private:
  ControlStrategy crossover(const ControlStrategy& a,
                              const ControlStrategy& b);
  ControlStrategy mutate(const ControlStrategy& s, float temperature);
  ControlStrategy llmMutate(const ControlStrategy& s, ModelRunner* runner);
  Individual& tournamentSelect();
  float clampParam(float val, float lo, float hi);

  std::vector<Individual> population_;
  Individual* lastProposed_ = nullptr;
  Individual best_;
  int generation_ = 0;
  int plateauCount_ = 0;
  double lastBestScore_ = -1e9;

  std::mt19937 rng_;
  std::normal_distribution<float> gaussNoise_{0.0f, 0.15f};
  std::uniform_real_distribution<float> uniform_{0.0f, 1.0f};
};

} // namespace tensorlang
} // namespace mlir

#endif // TENSORLANG_RUNTIME_STRATEGYGA_H