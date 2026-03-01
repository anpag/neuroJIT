#include "tensorlang/Runtime/EvolutionHarness.h"
#include "tensorlang/ExecutionEngine/JitRunner.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdio>
#include <fstream>
#include <chrono>

// Forward declarations from Runtime.cpp C API
extern "C" {
int   tensorlang_compile(const char* ir);
void* tensorlang_get_symbol_address(const char* name);
}

namespace mlir {
namespace tensorlang {

double runEvolutionLoop(JitRunner* jitRunner,
                        MLIRContext* /*context*/,
                        const std::string& /*baseModuleIR*/,
                        const HarnessConfig& cfg) {
  auto& ctx = JitContext::getInstance();
  auto& ga  = ctx.getGA();

  std::ofstream logStream;
  if (!cfg.logFile.empty()) {
    logStream.open(cfg.logFile);
    if (!logStream.is_open()) {
      fprintf(stderr, "[Harness] Warning: cannot open log file %s\n",
              cfg.logFile.c_str());
    }
  }

  auto log = [&](const std::string& msg) {
    printf("%s\n", msg.c_str());
    if (logStream.is_open()) logStream << msg << "\n";
  };

  log("=== NeuroJIT V2 Evolution Harness ===");
  log("Max generations: " + std::to_string(cfg.maxGenerations));
  log("Target score:    " + std::to_string(cfg.targetScore));

  // CSV header for result tracking
  if (logStream.is_open()) {
    logStream << "generation,score,impact_vel,fuel,kp,ki,kd,target_v\n";
  }

  double bestScore = -1000.0;

  for (int gen = 0; gen < cfg.maxGenerations; gen++) {
    auto genStart = std::chrono::high_resolution_clock::now();

    // 1. Ask GA for the next strategy to evaluate
    ControlStrategy strategy = ga.proposeNext(ctx.getModelRunner());

    if (cfg.verbose) {
      printf("[Harness] Gen %d: trying kp=%.3f ki=%.3f kd=%.3f tv=%.3f\n",
             gen, strategy.kp, strategy.ki, strategy.kd,
             strategy.targetVelocity);
    }

    // 2. Instantiate guaranteed-valid MLIR from template
    std::string newIR = instantiateControllerMLIR(strategy);

    // 3. Compile and hot-swap get_thrust
    if (tensorlang_compile(newIR.c_str()) != 0) {
      fprintf(stderr, "[Harness] Gen %d: compile failed — skipping\n", gen);
      fprintf(stderr, "Generated IR:\n%s\n", newIR.c_str());
      // Record as failed episode so GA penalises this region
      SimulationResult fail;
      fail.survived = false;
      ga.recordFitness(fail);
      continue;
    }

    // 4. Run the simulation episode(s)
    SimulationResult episodeResult;
    episodeResult.survived = true;
    episodeResult.impactVelocity = 0.0;
    episodeResult.fuelConsumed   = 0.0;

    for (int ep = 0; ep < cfg.episodesPerGeneration; ep++) {
      auto result = jitRunner->invoke("main");
      if (!result) {
        fprintf(stderr, "[Harness] Gen %d ep %d: invoke failed\n", gen, ep);
        episodeResult.survived = false;
        break;
      }
      // The SimulationResult was recorded by tensorlang_stop_timer.
      // Pull the latest recorded result from context.
      auto latestResult = ctx.getLastResult();
      // Average across episodes
      episodeResult.impactVelocity += latestResult.impactVelocity
                                      / cfg.episodesPerGeneration;
      episodeResult.fuelConsumed   += latestResult.fuelConsumed
                                      / cfg.episodesPerGeneration;
      episodeResult.settlingTime   += latestResult.settlingTime
                                      / cfg.episodesPerGeneration;
    }

    // 5. Check if restart was requested by assert
    std::string restartIR;
    if (ctx.consumeRestartRequest(restartIR)) {
      // Assert fired — this is a crash. Compile the fallback and continue.
      printf("[Harness] Gen %d: crash detected, applying fallback\n", gen);
      tensorlang_compile(restartIR.c_str());
      episodeResult.survived = false;
    }

    double score = episodeResult.score();
    if (score > bestScore) bestScore = score;

    // 6. Log result
    auto genEnd = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(genEnd - genStart).count();

    if (cfg.verbose) {
      printf("[Harness] Gen %d: score=%.2f impact=%.3f fuel=%.1f (%.2fs)\n",
             gen, score, episodeResult.impactVelocity,
             episodeResult.fuelConsumed, elapsed);
    }

    if (logStream.is_open()) {
      char row[256];
      std::snprintf(row, sizeof(row),
          "%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f",
          gen, score, episodeResult.impactVelocity, episodeResult.fuelConsumed,
          strategy.kp, strategy.ki, strategy.kd, strategy.targetVelocity);
      logStream << row << "\n";
      logStream.flush();
    }

    // 7. Early stopping
    if (score >= cfg.targetScore) {
      log("[Harness] Target score reached at generation " +
          std::to_string(gen));
      break;
    }
  }

  log("[Harness] Evolution complete. Best score: " +
      std::to_string(bestScore));
  return bestScore;
}

} // namespace tensorlang
} // namespace mlir