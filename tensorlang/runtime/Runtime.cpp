#include "Runtime.h"
#include "tensorlang/Runtime/JitContext.h"
#include "tensorlang/ExecutionEngine/JitRunner.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>
#include <vector>
#include <cstring>
#include <thread>
#include <algorithm>

extern "C" {

static std::chrono::time_point<std::chrono::high_resolution_clock> start_time;

void tensorlang_start_timer() {
  start_time = std::chrono::high_resolution_clock::now();
}

void tensorlang_stop_timer(float final_v) {
  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end_time - start_time;
  double latency = diff.count();
  auto& ctx = mlir::tensorlang::JitContext::getInstance();
  
  ctx.recordTelemetry(static_cast<double>(final_v), latency);
  
  printf("[Profiling] Simulation Latency: %.6f s (Avg: %.6f s), Final Vel: %.2f m/s (Best: %.2f m/s)\n", 
         latency, ctx.getAverageLatency(), final_v, ctx.getBestImpactVelocity());

  // Phase 6: Recursive Architecture Optimization (The Curiosity Drive)
  // We perform evolution sequentially AFTER the simulation to ensure stability.
  printf("[Curiosity] Entering Refinement Phase...\n");
  
  auto* runner = ctx.getModelRunner();
  if (runner) {
    std::string ir = ctx.getModuleIR();
    std::stringstream prompt_ss;
    prompt_ss << "ADAPTIVE REFINEMENT ENGINE (Generation " << ctx.getHealingAttempts() << "):\n"
              << "Current Best Impact Velocity: " << ctx.getBestImpactVelocity() << " m/s\n"
              << "Current Performance: " << latency << " s\n\n"
              << "GOAL: Proactively mutate the @get_thrust function. "
              << "Try a NEW architectural approach (e.g. PID, derivative-aware control, or state-based logic) "
              << "to achieve a landing closer to -0.5 m/s with higher fuel efficiency.\n"
              << "Return ONLY the FULL MLIR module.\n\n" << ir;
    
    std::string response = runner->query(prompt_ss.str());
    if (response.find("(error") == std::string::npos) {
       printf("[Curiosity] New architecture received. Size: %zu bytes.\n", response.size());
       if (tensorlang_compile(response.c_str()) == 0) {
         void* fnPtr = tensorlang_get_symbol_address("get_thrust");
         if (fnPtr) {
           printf("[Curiosity] Evolution successful. New architecture synthesized.\n");
           ctx.setOptimizedFunction(fnPtr);
         }
       }
    }
  }
}

void tensorlang_print_f32(float* data, int64_t rank, int64_t* shape) {
  // ... (existing implementation)
}

void tensorlang_print_status(float h, float v) {
  printf("|     *     | Alt: %6.2f m, Vel: %6.2f m/s\n", h, v);
  fflush(stdout);
}

char* tensorlang_get_ir() {
  std::string ir = mlir::tensorlang::JitContext::getInstance().getModuleIR();
  char* c_str = new char[ir.length() + 1];
  std::strcpy(c_str, ir.c_str());
  return c_str;
}

char* tensorlang_query_model(const char* prompt) {
  auto* runner = mlir::tensorlang::JitContext::getInstance().getModelRunner();
  if (!runner) {
    llvm::errs() << "[Runtime] Error: ModelRunner is null.\n";
    return nullptr;
  }
  
  std::string response = runner->query(prompt);
  llvm::errs() << "[Runtime] Model returned " << response.length() << " bytes.\n";
  
  char* c_str = new char[response.length() + 1];
  std::strcpy(c_str, response.c_str());
  return c_str;
}

int tensorlang_compile(const char* ir_string) {
  if (!ir_string) {
    llvm::errs() << "[Runtime] Error: ir_string is NULL in tensorlang_compile.\n";
    return -1;
  }

  auto* runner = mlir::tensorlang::JitContext::getInstance().getRunner();
  if (!runner) return -1;
  
  if (auto err = runner->compileString(ir_string)) {
    llvm::errs() << "JIT Compilation Failed: " << llvm::toString(std::move(err)) << "\n";
    return -1;
  }
  return 0; 
}

void* tensorlang_get_symbol_address(const char* name) {
  auto* runner = mlir::tensorlang::JitContext::getInstance().getRunner();
  if (!runner) return nullptr;
  
  auto symOrErr = runner->lookup(name);
  if (!symOrErr) {
    llvm::consumeError(symOrErr.takeError());
    return nullptr;
  }
  return *symOrErr;
}

void* tensorlang_get_current_impl(const char* default_name) {
  auto& ctx = mlir::tensorlang::JitContext::getInstance();
  
  // 1. Check if optimized version exists (Atomic Load Acquire)
  void* optimized = ctx.getOptimizedFunction();
  if (optimized) {
    return optimized;
  }
  
  // 2. Fallback to default
  return tensorlang_get_symbol_address(default_name);
}

void tensorlang_optimize_async(const char* prompt, const char* target_name) {
  auto& ctx = mlir::tensorlang::JitContext::getInstance();
  
  // 1. Check if already optimizing (CAS)
  if (!ctx.tryStartOptimization()) {
    return;
  }
  
  std::string promptStr = prompt;
  std::string targetNameStr = target_name;
  
  // 2. Check Cache
  std::string cachedIR = ctx.getStrategyCache().lookup(promptStr);
  if (!cachedIR.empty()) {
    llvm::errs() << "[Async] Cache hit for optimization (" << targetNameStr << ")!\n";
    if (tensorlang_compile(cachedIR.c_str()) == 0) {
      void* fnPtr = tensorlang_get_symbol_address(targetNameStr.c_str());
      if (fnPtr) ctx.setOptimizedFunction(fnPtr);
    }
    ctx.finishOptimization();
    return;
  }
  
  // 3. Spawn detached thread for optimization
  std::thread([promptStr, targetNameStr]() {
    llvm::errs() << "[Async] Cache miss. Starting optimization thread...\n";
    auto& ctx = mlir::tensorlang::JitContext::getInstance();
    
    auto* runner = ctx.getModelRunner();
    if (!runner) {
      ctx.finishOptimization();
      return;
    }
    
    std::string response = runner->query(promptStr);
    if (response.find("(error") != std::string::npos) {
      llvm::errs() << "[Async] Query failed.\n";
      ctx.finishOptimization();
      return;
    }
    
    // Cache the result
    ctx.getStrategyCache().insert(promptStr, response);
    
    if (tensorlang_compile(response.c_str()) != 0) {
      llvm::errs() << "[Async] Compilation failed.\n";
      ctx.finishOptimization();
      return;
    }
    
    void* fnPtr = tensorlang_get_symbol_address(targetNameStr.c_str());
    if (fnPtr) {
      llvm::errs() << "[Async] Hot-swap successful! New implementation ready.\n";
      ctx.setOptimizedFunction(fnPtr);
    }
    
    ctx.finishOptimization();
  }).detach();
}

void tensorlang_assert_fail(int64_t loc) {
  auto& ctx = mlir::tensorlang::JitContext::getInstance();
  ctx.incrementHealingAttempts();
  
  if (ctx.getHealingAttempts() > 3) {
    llvm::errs() << "[System 2] Self-healing attempted but failed to prevent crash. Manual intervention required.\n";
    exit(1);
  }
  llvm::errs() << "[System 2] CRASH IMMINENT! Violation detected (Attempt " << ctx.getHealingAttempts() << ").\n";
  
  char* ir = tensorlang_get_ir();
  if (!ir) {
    llvm::errs() << "[System 2] Error: Could not retrieve IR.\n";
    return;
  }
  
  std::string prompt = "The following Lunar Lander simulation failed. The lander crashed. "
                       "Rewrite the 'get_thrust' function to land safely (soft landing). "
                       "Return ONLY the FULL MLIR module. Do not include markdown backticks. "
                       "IMPORTANT: For arith.cmpf, use the type of the operands after the colon (e.g., : f32). "
                       "Do not use arith.maxf or arith.minf. Use arith.select.\n\n" + std::string(ir);
  
  std::string promptStr = prompt;
  std::string cachedIR = ctx.getStrategyCache().lookup(promptStr);

  if (!cachedIR.empty()) {
    llvm::errs() << "[System 2] Cache hit! Applying fix immediately...\n";
    if (tensorlang_compile(cachedIR.c_str()) == 0) {
      llvm::errs() << "[System 2] Success! Logic updated. Unwinding stack for restart...\n";
      delete[] ir;
      std::longjmp(ctx.getRecoveryPoint(), 1);
    } else {
      llvm::errs() << "[System 2] Failed to compile cached fix. Proceeding to query...\n";
    }
  }

  llvm::errs() << "[System 2] Cache miss. Querying Gemini for a fix...\n";
  char* fixed_ir = tensorlang_query_model(prompt.c_str());
  
  if (fixed_ir) {
    llvm::errs() << "[System 2] Hot-swapping fixed code...\n";
    ctx.getStrategyCache().insert(promptStr, std::string(fixed_ir));

    if (tensorlang_compile(fixed_ir) == 0) {
      llvm::errs() << "[System 2] Success! Logic updated. Unwinding stack for restart...\n";
      delete[] ir;
      delete[] fixed_ir;
      // UNWIND AND RESTART
      std::longjmp(ctx.getRecoveryPoint(), 1);
    } else {
      llvm::errs() << "[System 2] Failed to compile the fix.\n";
    }
    delete[] fixed_ir;
  } else {
    llvm::errs() << "[System 2] Gemini failed to provide a fix.\n";
  }
  delete[] ir;
}

} // extern "C"
