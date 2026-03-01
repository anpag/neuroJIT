#include "tensorlang/Runtime/StrategyGA.h"
#include "tensorlang/Runtime/ModelRunner.h"
#include "llvm/Support/FormatVariadic.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <sstream>

namespace mlir {
namespace tensorlang {

StrategyGA::StrategyGA(unsigned seed) : rng_(seed) {}

void StrategyGA::seed(const ControlStrategy& known_good) {
  population_.clear();
  // Elite: the known-good individual
  Individual elite;
  elite.strategy = known_good;
  elite.generation = 0;
  population_.push_back(elite);
  // Fill rest with small perturbations
  std::normal_distribution<float> small_noise(0.0f, 0.05f);
  for (int i = 1; i < kPopSize; i++) {
    Individual ind;
    ind.strategy = mutate(known_good, 0.1f);
    ind.generation = 0;
    population_.push_back(ind);
  }
  best_ = population_[0];
}

void StrategyGA::seedRandom() {
  population_.clear();
  std::uniform_real_distribution<float> kp_dist(0.5f, 4.0f);
  std::uniform_real_distribution<float> ki_dist(0.0f, 0.5f);
  std::uniform_real_distribution<float> kd_dist(0.0f, 2.0f);
  std::uniform_real_distribution<float> tv_dist(-3.0f, -0.5f);
  std::uniform_real_distribution<float> tc_dist(2.0f, 8.0f);
  for (int i = 0; i < kPopSize; i++) {
    Individual ind;
    ind.strategy.kp = kp_dist(rng_);
    ind.strategy.ki = ki_dist(rng_);
    ind.strategy.kd = kd_dist(rng_);
    ind.strategy.targetVelocity = tv_dist(rng_);
    ind.strategy.thrustClampMax = tc_dist(rng_);
    ind.generation = 0;
    population_.push_back(ind);
  }
  best_ = population_[0];
}

void StrategyGA::recordFitness(const SimulationResult& result) {
  if (!lastProposed_) return;
  lastProposed_->result = result;

  if (result.score() > best_.result.score()) {
    best_ = *lastProposed_;
    plateauCount_ = 0;
    lastBestScore_ = result.score();
    printf("[GA] Gen %d: New best score=%.2f (kp=%.3f ki=%.3f kd=%.3f)\n",
           generation_, result.score(),
           best_.strategy.kp, best_.strategy.ki, best_.strategy.kd);
  } else {
    plateauCount_++;
  }

  // Sort population descending by score
  std::sort(population_.begin(), population_.end(),
            [](const Individual& a, const Individual& b) {
              return a.result.score() > b.result.score();
            });

  generation_++;
}

ControlStrategy StrategyGA::proposeNext(ModelRunner* runner) {
  // If population is not yet fully evaluated, return the next unevaluated
  for (auto& ind : population_) {
    if (!ind.result.survived && ind.result.impactVelocity == -999.0) {
      lastProposed_ = &ind;
      return ind.strategy;
    }
  }

  // Full generation evaluated. Evolve.
  // Keep elites
  std::vector<Individual> next_pop(
      population_.begin(),
      population_.begin() + std::min(kEliteCount, (int)population_.size()));

  // Fill with crossover + mutation
  while ((int)next_pop.size() < kPopSize) {
    auto& parentA = tournamentSelect();
    auto& parentB = tournamentSelect();
    Individual child;
    child.strategy = crossover(parentA.strategy, parentB.strategy);

    // Use LLM mutation if GA has plateaued, else standard gaussian
    if (plateaued() && runner) {
      child.strategy = llmMutate(child.strategy, runner);
      printf("[GA] LLM oracle mutation triggered (plateau=%d)\n", plateauCount_);
    } else {
      child.strategy = mutate(child.strategy, 0.15f);
    }
    child.generation = generation_;
    next_pop.push_back(child);
  }

  population_ = std::move(next_pop);
  lastProposed_ = &population_.back();
  return lastProposed_->strategy;
}

const Individual& StrategyGA::best() const { return best_; }

// ---------------------------------------------------------------------------
// Private
// ---------------------------------------------------------------------------

ControlStrategy StrategyGA::crossover(const ControlStrategy& a,
                                       const ControlStrategy& b) {
  float alpha = uniform_(rng_);
  ControlStrategy child;
  child.kp             = alpha * a.kp             + (1-alpha) * b.kp;
  child.ki             = alpha * a.ki             + (1-alpha) * b.ki;
  child.kd             = alpha * a.kd             + (1-alpha) * b.kd;
  child.targetVelocity = alpha * a.targetVelocity + (1-alpha) * b.targetVelocity;
  child.thrustClampMax = alpha * a.thrustClampMax + (1-alpha) * b.thrustClampMax;
  return child;
}

ControlStrategy StrategyGA::mutate(const ControlStrategy& s, float temperature) {
  std::normal_distribution<float> noise(0.0f, temperature);
  ControlStrategy m = s;
  m.kp             = clampParam(s.kp             + noise(rng_), 0.0f, 10.0f);
  m.ki             = clampParam(s.ki             + noise(rng_), 0.0f, 2.0f);
  m.kd             = clampParam(s.kd             + noise(rng_), 0.0f, 5.0f);
  m.targetVelocity = clampParam(s.targetVelocity + noise(rng_), -5.0f, -0.1f);
  m.thrustClampMax = clampParam(s.thrustClampMax + noise(rng_), 1.0f, 10.0f);
  return m;
}

ControlStrategy StrategyGA::llmMutate(const ControlStrategy& s,
                                       ModelRunner* runner) {
  // Ask LLM to suggest new parameters given current performance.
  // The LLM is constrained by GBNF grammar to return only JSON.
  std::string prompt = llvm::formatv(
      "Current PID controller performance has plateaued.\n"
      "Current parameters: kp={0:f}, ki={1:f}, kd={2:f}, "
      "target_velocity={3:f}, thrust_clamp_max={4:f}\n"
      "Best score so far: {5:f}\n"
      "Suggest improved parameters. "
      "Return JSON matching exactly: "
      "{{\"kp\":N,\"ki\":N,\"kd\":N,"
      "\"target_velocity\":N,\"thrust_clamp_max\":N}}",
      s.kp, s.ki, s.kd, s.targetVelocity, s.thrustClampMax,
      best_.result.score()).str();

  std::string response = runner->query(prompt);

  // Parse the JSON response
  // Use the same nlohmann::json that StrategyCache already pulls in
  ControlStrategy result = s; // fallback to current if parse fails
  bool parse_failed = false;
  auto extract = [&](const std::string& key) -> float {
    auto pos = response.find("\"" + key + "\"");
    if (pos == std::string::npos) { parse_failed = true; return 0.0f; }
    auto colon = response.find(':', pos);
    if (colon == std::string::npos) { parse_failed = true; return 0.0f; }
    const char* start = response.c_str() + colon + 1;
    char* end;
    float val = std::strtof(start, &end);
    if (end == start) parse_failed = true;
    return val;
  };
  
  float n_kp = extract("kp");
  float n_ki = extract("ki");
  float n_kd = extract("kd");
  float n_tv = extract("target_velocity");
  float n_tc = extract("thrust_clamp_max");

  if (parse_failed) {
    printf("[GA] LLM parse failed, using standard mutation fallback\n");
    return mutate(s, 0.3f); // wider mutation on failure
  }

  result.kp             = clampParam(n_kp,             0.0f, 10.0f);
  result.ki             = clampParam(n_ki,             0.0f, 2.0f);
  result.kd             = clampParam(n_kd,             0.0f, 5.0f);
  result.targetVelocity = clampParam(n_tv,-5.0f, -0.1f);
  result.thrustClampMax = clampParam(n_tc,1.0f, 10.0f);
  return result;
}

Individual& StrategyGA::tournamentSelect() {
  int a = rng_() % population_.size();
  int b = rng_() % population_.size();
  return population_[a].result.score() >= population_[b].result.score()
             ? population_[a]
             : population_[b];
}

float StrategyGA::clampParam(float val, float lo, float hi) {
  return std::max(lo, std::min(hi, val));
}

} // namespace tensorlang
} // namespace mlir