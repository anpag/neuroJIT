#include "tensorlang/Runtime/EvolutionHarness.h"
#include "tensorlang/ExecutionEngine/JitRunner.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
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

static mlir::OwningOpRef<mlir::ModuleOp>
parseModuleFromString(mlir::MLIRContext* ctx, const std::string& ir) {
  llvm::SourceMgr sm;
  sm.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBuffer(ir), llvm::SMLoc());
  return mlir::parseSourceFile<mlir::ModuleOp>(sm, ctx);
}

double runEvolutionLoop(JitRunner* jitRunner,
                        MLIRContext* context,
                        const std::string& baseModuleIR,
                        const HarnessConfig& cfg) {
  auto& ctx = JitContext::getInstance();
  auto& ga  = ctx.getGA();

  // Disable background worker to prevent concurrent JIT compilation crashes
  ctx.setOnlineOptimization(false);

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

  // We evaluate individuals one by one. A full GA generation is complete
  // when the entire population has been evaluated.
  // We run until the GA reaches the desired number of generations.
  int maxEvals = cfg.maxGenerations * StrategyGA::kPopSize;

  for (int eval = 0; eval < maxEvals; eval++) {
    auto genStart = std::chrono::high_resolution_clock::now();

    int currentGAGen = ga.generation();

    // 1. Ask GA for the next strategy to evaluate
    ControlStrategy strategy = ga.proposeNext(ctx.getModelRunner());

    if (cfg.verbose) {
      printf("[Harness] Gen %d (Eval %d): trying kp=%.3f ki=%.3f kd=%.3f tv=%.3f\n",
             currentGAGen, eval, strategy.kp, strategy.ki, strategy.kd,
             strategy.targetVelocity);
    }

    // 2. Instantiate guaranteed-valid MLIR from template
    std::string strategyIR = instantiateControllerMLIR(strategy);

    // Merge baseModuleIR with the newly generated strategyIR
    mlir::OwningOpRef<mlir::ModuleOp> baseMod = parseModuleFromString(context, baseModuleIR);
    mlir::OwningOpRef<mlir::ModuleOp> stratMod = parseModuleFromString(context, strategyIR);

    if (!baseMod || !stratMod) {
      fprintf(stderr, "[Harness] Gen %d (Eval %d): failed to parse modules\n", currentGAGen, eval);
      SimulationResult fail;
      fail.survived = false;
      ga.recordFitness(fail);
      continue;
    }

    // Remove old @get_thrust
    auto oldFunc = baseMod->lookupSymbol<mlir::func::FuncOp>("get_thrust");
    if (oldFunc) {
      oldFunc.erase();
    }

    // Clone new @get_thrust into base module
    auto newFunc = stratMod->lookupSymbol<mlir::func::FuncOp>("get_thrust");
    if (newFunc) {
      baseMod->push_back(newFunc->clone());
    }

    std::string mergedIR;
    {
      llvm::raw_string_ostream os(mergedIR);
      baseMod->print(os);
    }

    // 3. Create a fresh JIT environment for this evaluation
    auto evalRunnerOrErr = JitRunner::create();
    if (auto err = evalRunnerOrErr.takeError()) {
      fprintf(stderr, "[Harness] Gen %d (Eval %d): failed to create JIT\n", currentGAGen, eval);
      llvm::consumeError(std::move(err));
      continue;
    }
    auto evalRunner = std::move(*evalRunnerOrErr);

    JitRunner* oldRunner = ctx.getRunner();
    ctx.registerRunner(evalRunner.get());

    // 4. Compile the merged module
    if (auto err = evalRunner->loadModule(baseMod.get())) {
      fprintf(stderr, "[Harness] Gen %d (Eval %d): compile failed — skipping\n", currentGAGen, eval);
      llvm::consumeError(std::move(err));
      SimulationResult fail;
      fail.survived = false;
      ga.recordFitness(fail);
      ctx.registerRunner(oldRunner);
      continue;
    }

    // 5. Run the simulation episode(s)
    SimulationResult episodeResult;
    episodeResult.survived = true;
    episodeResult.impactVelocity = 0.0;
    episodeResult.fuelConsumed   = 0.0;
    episodeResult.settlingTime   = 0.0;

    int validEpisodes = 0;
    bool allSettled = true;

    for (int ep = 0; ep < cfg.episodesPerGeneration; ep++) {
      auto result = evalRunner->invoke("sim_main");
      if (!result) {
        llvm::Error err = result.takeError();
        fprintf(stderr, "[Harness] Gen %d (Eval %d) ep %d: invoke failed: %s\n", currentGAGen, eval, ep, llvm::toString(std::move(err)).c_str());
        episodeResult.survived = false;
        break;
      }

      // 5. Check if restart was requested by assert (crash inside MLIR)
      std::string restartIR;
      if (ctx.consumeRestartRequest(restartIR)) {
        printf("[Harness] Gen %d (Eval %d): crash detected, applying fallback\n", currentGAGen, eval);
        tensorlang_compile(restartIR.c_str());
        episodeResult.survived = false;
        break;
      }

      // The SimulationResult was recorded by tensorlang_stop_timer.
      auto latestResult = ctx.getLastResult();
      if (!latestResult.survived) {
        episodeResult.survived = false;
        break;
      }

      validEpisodes++;
      episodeResult.impactVelocity += latestResult.impactVelocity;
      episodeResult.fuelConsumed   += latestResult.fuelConsumed;
      
      if (latestResult.settlingTime >= 999.0) {
        allSettled = false;
      } else {
        episodeResult.settlingTime += latestResult.settlingTime;
      }
    }

    if (episodeResult.survived) {
      if (validEpisodes != cfg.episodesPerGeneration) {
        fprintf(stderr, "FATAL: Survived but validEpisodes (%d) != requested (%d)\n", 
                validEpisodes, cfg.episodesPerGeneration);
        std::abort();
      }

      // Average across episodes
      episodeResult.impactVelocity /= validEpisodes;
      episodeResult.fuelConsumed   /= validEpisodes;
      
      if (allSettled) {
        episodeResult.settlingTime /= validEpisodes;
      } else {
        episodeResult.settlingTime = 999.0;
      }
    }

    double score = episodeResult.score();
    if (score > bestScore) bestScore = score;

    // 6. Log result
    auto genEnd = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(genEnd - genStart).count();

    ga.recordFitness(episodeResult);

    if (cfg.verbose) {
      printf("[Harness] Gen %d (Eval %d): score=%.2f impact=%.3f fuel=%.1f (%.2fs)\n",
             currentGAGen, eval, score, episodeResult.impactVelocity,
             episodeResult.fuelConsumed, elapsed);
    }

    if (logStream.is_open()) {
      char row[256];
      std::snprintf(row, sizeof(row),
          "%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f",
          currentGAGen, score, episodeResult.impactVelocity, episodeResult.fuelConsumed,
          strategy.kp, strategy.ki, strategy.kd, strategy.targetVelocity);
      logStream << row << "\n";
      logStream.flush();
    }

    // 7. Early stopping
    if (score >= cfg.targetScore) {
      log("[Harness] Target score reached at generation " +
          std::to_string(currentGAGen));
      ctx.registerRunner(oldRunner);
      break;
    }

    ctx.registerRunner(oldRunner);
  }

  log("[Harness] Evolution complete. Best score: " +
      std::to_string(bestScore));
  return bestScore;
}

} // namespace tensorlang
} // namespace mlir