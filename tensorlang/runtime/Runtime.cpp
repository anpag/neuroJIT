#include "Runtime.h"
#include "tensorlang/Runtime/JitContext.h"
#include "tensorlang/Runtime/OptimizationStrategy.h"
#include "tensorlang/Runtime/MLIRTemplates.h"
#include "tensorlang/ExecutionEngine/JitRunner.h"
#include "llvm/Support/raw_ostream.h"

#include <chrono>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <string>

// ---------------------------------------------------------------------------
// Module-level timer state
// ---------------------------------------------------------------------------

static thread_local std::chrono::time_point<
    std::chrono::high_resolution_clock> g_timer_start;
static thread_local float g_fuel_consumed = 0.0f;
static thread_local float g_settling_ticks = 999.0f;
static thread_local bool  g_settled = false;

extern "C" {

// ---------------------------------------------------------------------------
// Timer and telemetry
// ---------------------------------------------------------------------------

void tensorlang_start_timer() {
  g_timer_start   = std::chrono::high_resolution_clock::now();
  g_fuel_consumed = 0.0f;
  g_settling_ticks = 999.0f;
  g_settled = false;
}

void tensorlang_record_thrust(float thrust) {
  // Called each sim tick to accumulate fuel usage.
  // Fuel cost = |thrust| * dt (dt=0.1 implicit)
  g_fuel_consumed += std::abs(thrust) * 0.1f;
}

void tensorlang_stop_timer(float final_v) {
  auto end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration<double>(end - g_timer_start).count();

  auto& ctx = mlir::tensorlang::JitContext::getInstance();

  // Build the SimulationResult — this is the fitness function.
  mlir::tensorlang::SimulationResult result;
  result.survived        = true; // If we reach stop_timer, no crash occurred
  result.impactVelocity  = static_cast<double>(final_v);
  result.fuelConsumed    = static_cast<double>(g_fuel_consumed);
  result.settlingTime    = static_cast<double>(g_settling_ticks);

  ctx.recordResult(result);
  ctx.getGA().recordFitness(result);

  printf("[Telemetry] Score=%.2f | ImpactV=%.3f m/s | Fuel=%.1f | "
         "Ticks=%.0f | Elapsed=%.3fs | Best=%.2f\n",
         result.score(), result.impactVelocity, result.fuelConsumed,
         result.settlingTime, elapsed, ctx.getBestScore());

  // Check if we should persist this as the best known lobe
  static constexpr double kSaveFitnessThreshold = 50.0;
  if (result.score() > kSaveFitnessThreshold &&
      result.score() > ctx.getBestScore() - 1.0) {
    std::string ir = ctx.getModuleIR();
    ctx.saveLobe("Stability_v1", ir, result);
  }

  // Non-blocking: submit an optimization request to the background worker.
  // The simulation loop continues immediately — it does NOT wait for the LLM.
  static constexpr double kOptimizationTriggerScore = 60.0;
  if (result.score() < kOptimizationTriggerScore && !ctx.getWorker().isBusy()) {
    // Ask the GA to propose the next strategy to try
    auto* runner = ctx.getModelRunner();
    mlir::tensorlang::ControlStrategy next =
        ctx.getGA().proposeNext(runner);

    // Instantiate guaranteed-valid MLIR from the template
    std::string newIR = mlir::tensorlang::instantiateControllerMLIR(next);

    // Submit to worker — non-blocking
    mlir::tensorlang::OptimizationRequest req;
    req.functionName = "get_thrust";
    req.baselineIR   = newIR; // pre-compiled by template, not by LLM
    req.baseline     = result;
    ctx.getWorker().submit(std::move(req));
  }
}

// ---------------------------------------------------------------------------
// Simulation I/O
// ---------------------------------------------------------------------------

void tensorlang_print_status(float h, float v) {
  // Track settling: first tick where |v| < 1.0 m/s
  if (!g_settled && std::abs(v) < 1.0f) {
    // We don't have a tick counter here — settlingTime is updated externally
    g_settled = true;
  }
  printf("|     *     | Alt: %6.2f m, Vel: %6.2f m/s\n", h, v);
  fflush(stdout);
}

float tensorlang_get_random() {
  return static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 0.5f;
}

void tensorlang_print_f32(float* /*data*/, int64_t /*rank*/,
                          int64_t* /*shape*/) {
  // Stub — implement if needed
}

// ---------------------------------------------------------------------------
// Assert: SAFE FAILURE — no longjmp, no UB
// ---------------------------------------------------------------------------

void tensorlang_assert_fail(int64_t /*loc*/) {
  auto& ctx = mlir::tensorlang::JitContext::getInstance();

  // Record this as a failed episode
  mlir::tensorlang::SimulationResult crashResult;
  crashResult.survived = false;
  ctx.recordResult(crashResult);
  ctx.getGA().recordFitness(crashResult);

  printf("[Assert] Violation detected. Requesting restart via flag.\n");

  // Check registry for a known-good lobe to restart with
  std::string recoveryIR;
  if (ctx.hasLobe("Stability_v1")) {
    recoveryIR = ctx.loadLobe("Stability_v1");
    printf("[Assert] Using cached Stability_v1 for restart.\n");
  } else {
    // Fall back to a safe hardcoded P controller
    mlir::tensorlang::ControlStrategy safe;
    safe.kp = 2.0f;
    safe.kd = 1.5f;
    safe.targetVelocity = -1.0f;
    safe.thrustClampMax = 4.0f;
    recoveryIR = mlir::tensorlang::instantiateControllerMLIR(safe);
    printf("[Assert] No registry entry — using safe hardcoded fallback.\n");
  }

  // Signal the main loop to restart cleanly.
  // The main loop in tensorlang-run.cpp checks this flag after each run()
  // and reloads the module. No stack unwinding. No UB.
  ctx.requestRestart(recoveryIR);

  // IMPORTANT: Return normally. The main loop handles the restart.
  // Do NOT call longjmp, abort, or exit here.
}

// ---------------------------------------------------------------------------
// JIT reflection API
// ---------------------------------------------------------------------------

char* tensorlang_get_ir() {
  // Thread-local buffer — avoids heap allocation and ownership issues
  static thread_local std::string buffer;
  buffer = mlir::tensorlang::JitContext::getInstance().getModuleIR();
  return buffer.data();
}

char* tensorlang_query_model(const char* prompt) {
  // Direct query — used for one-off requests, not the main evolution loop.
  // The main loop uses the GA + worker pipeline instead.
  static thread_local std::string buffer;
  auto* runner = mlir::tensorlang::JitContext::getInstance().getModelRunner();
  if (!runner) {
    buffer = R"({"kp":1.5,"ki":0.0,"kd":1.0,"target_velocity":-1.0,"thrust_clamp_max":4.0})";
    return buffer.data();
  }
  buffer = runner->query(prompt);
  return buffer.data();
}

int tensorlang_compile(const char* ir_string) {
  if (!ir_string) return -1;
  auto* runner = mlir::tensorlang::JitContext::getInstance().getRunner();
  if (!runner) return -1;
  if (auto err = runner->compileString(ir_string)) {
    llvm::errs() << "[JIT] Compile error: " << llvm::toString(std::move(err))
                 << "\n";
    return -1;
  }
  return 0;
}

void* tensorlang_get_symbol_address(const char* name) {
  auto* runner = mlir::tensorlang::JitContext::getInstance().getRunner();
  if (!runner) return nullptr;
  auto sym = runner->lookup(name);
  if (!sym) {
    llvm::consumeError(sym.takeError());
    return nullptr;
  }
  return *sym;
}

void* tensorlang_get_current_impl(const char* default_name) {
  auto& ctx = mlir::tensorlang::JitContext::getInstance();
  void* opt = ctx.getOptimizedFunction();
  if (opt) return opt;
  return tensorlang_get_symbol_address(default_name);
}

void tensorlang_optimize_async(const char* /*prompt*/,
                               const char* /*target_name*/) {
  // Deprecated entry point. Use the GA + worker pipeline.
  // Kept as a no-op stub so existing MLIR files that call it still compile.
  printf("[Runtime] tensorlang_optimize_async: use GA pipeline instead\n");
}

// Timer stubs (kept for ABI compatibility with existing .mlir files)
void tensorlang_start_timer_noarg() { tensorlang_start_timer(); }

} // extern "C"