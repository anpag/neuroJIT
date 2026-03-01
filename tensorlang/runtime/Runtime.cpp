#include "Runtime.h"
#include "tensorlang/Runtime/JitContext.h"
#include "tensorlang/Runtime/OptimizationWorker.h"
#include "tensorlang/ExecutionEngine/JitRunner.h"
#include "llvm/Support/raw_ostream.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>

extern "C" {

// ===========================================================================
// RUNTIME CONTRACT BOUNDARY (MLIR -> C++)
// ===========================================================================
// The functions below are invoked directly from compiled MLIR modules.
// They handle general telemetry and the asynchronous submission of failed 
// IR to the LLM worker.
// ===========================================================================

void tensorlang_print_status(float h, float v) {
  printf("|     *     | Alt: %6.2f m, Vel: %6.2f m/s\n", h, v);
  fflush(stdout);
}

float tensorlang_get_random() {
  return static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 0.5f;
}

void tensorlang_print_f32(float* /*data*/, int64_t /*rank*/,
                          int64_t* /*shape*/) {
  // Stub
}

// ---------------------------------------------------------------------------
// Async Optimization and Healing Triggers
// ---------------------------------------------------------------------------

void tensorlang_assert_fail(int64_t loc) {
  auto& ctx = mlir::tensorlang::JitContext::getInstance();

  printf("[Assert] Violation detected at loc %ld. Submitting async repair request...\n", (long)loc);

  if (ctx.isOnlineOptimizationEnabled() && !ctx.getWorker().isBusy()) {
    mlir::tensorlang::OptimizationRequest req;
    // For now, hardcode "get_thrust" or make it generic based on profiling in Phase 7
    req.functionName = "get_thrust"; 
    req.originalIR = ctx.getModuleIR();
    req.errorMessage = "Assertion failed at location " + std::to_string(loc);
    
    ctx.getWorker().submit(std::move(req));
  } else {
    printf("[Assert] Worker busy or offline optimization disabled.\n");
  }

  // Check registry for a known-good lobe to restart with
  std::string recoveryIR = ctx.loadLobe("latest_async_repair");
  
  if (recoveryIR.empty()) {
    printf("[Assert] No repair lobe found. Restarting from baseline.\n");
    recoveryIR = ctx.getModuleIR(); // fallback to current logic (which will likely just crash again until healed)
  } else {
    printf("[Assert] Initiating restart with cached repair lobe.\n");
  }

  // Signal the main loop to restart cleanly. No Undefined Behavior.
  ctx.requestRestart(recoveryIR);
}

void tensorlang_optimize_async(const char* /*prompt*/, const char* target_name) {
  auto& ctx = mlir::tensorlang::JitContext::getInstance();
  if (ctx.isOnlineOptimizationEnabled() && !ctx.getWorker().isBusy()) {
    mlir::tensorlang::OptimizationRequest req;
    req.functionName = target_name ? target_name : "main";
    req.originalIR = ctx.getModuleIR();
    req.errorMessage = ""; // Empty implies standard PGO/optimization request
    ctx.getWorker().submit(std::move(req));
  }
}

// ---------------------------------------------------------------------------
// Legacy Timer / Profiling Stubs (can be adapted for real profiling later)
// ---------------------------------------------------------------------------

static thread_local std::chrono::time_point<std::chrono::high_resolution_clock> g_timer_start;

void tensorlang_start_timer() {
  g_timer_start = std::chrono::high_resolution_clock::now();
}

void tensorlang_stop_timer(float /*final_val*/) {
  auto end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration<double>(end - g_timer_start).count();
  printf("[Telemetry] Episode elapsed: %.3fs\n", elapsed);
}

void tensorlang_record_thrust(float /*thrust*/) {
  // Can track variables here if we build dynamic profiling
}

// ---------------------------------------------------------------------------
// JIT reflection API
// ---------------------------------------------------------------------------

char* tensorlang_get_ir() {
  static thread_local std::string buffer;
  buffer = mlir::tensorlang::JitContext::getInstance().getModuleIR();
  return buffer.data();
}

int tensorlang_compile(const char* ir_string) {
  if (!ir_string) return -1;
  auto* runner = mlir::tensorlang::JitContext::getInstance().getRunner();
  if (!runner) return -1;
  if (auto err = runner->compileString(ir_string)) {
    llvm::errs() << "[JIT] Compile error: " << llvm::toString(std::move(err)) << "\n";
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

// Backward compatibility stub
void tensorlang_start_timer_noarg() { tensorlang_start_timer(); }

} // extern "C"