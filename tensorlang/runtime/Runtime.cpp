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
    // print error
    llvm::errs() << "JIT Compilation Failed: " << err << "\n";
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
    // Already optimizing, ignore request
    return;
  }
  
  std::string promptStr = prompt;
  std::string targetNameStr = target_name;
  
  // 2. Spawn detached thread
  std::thread([promptStr, targetNameStr]() {
    llvm::errs() << "[Async] Starting optimization thread...\n";
    
    // We cannot capture ctx ref safely if thread outlives main? 
    // JitContext is singleton static, so it should be fine until exit.
    auto& ctx = mlir::tensorlang::JitContext::getInstance();
    
    // A. Query Model
    char* ir = tensorlang_query_model(promptStr.c_str());
    if (!ir) {
      llvm::errs() << "[Async] Query failed.\n";
      ctx.finishOptimization();
      return;
    }
    
    // B. Compile
    // Note: Compile might not be thread safe if LLVMContext isn't handled right. 
    // But we fixed compileString to use local context.
    if (tensorlang_compile(ir) != 0) {
      llvm::errs() << "[Async] Compilation failed.\n";
      delete[] ir;
      ctx.finishOptimization();
      return;
    }
    delete[] ir;
    
    // C. Lookup new symbol
    void* fnPtr = tensorlang_get_symbol_address(targetNameStr.c_str());
    if (fnPtr) {
      llvm::errs() << "[Async] Hot-swap successful! New implementation ready.\n";
      // D. Atomically update (Atomic Store Release)
      ctx.setOptimizedFunction(fnPtr);
    } else {
      llvm::errs() << "[Async] Failed to find target symbol: " << targetNameStr << "\n";
    }
    
    ctx.finishOptimization();
  }).detach();
}

static int healing_attempts = 0;

void tensorlang_assert_fail(int64_t loc) {
  if (healing_attempts++ > 3) {
    llvm::errs() << "[System 2] Self-healing attempted but failed to prevent crash. Manual intervention required.\n";
    exit(1);
  }
  llvm::errs() << "[System 2] CRASH IMMINENT! Violation detected.\n";
  
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
  
  llvm::errs() << "[System 2] Querying Gemini for a fix...\n";
  char* fixed_ir = tensorlang_query_model(prompt.c_str());
  
  if (fixed_ir) {
    llvm::errs() << "[System 2] Hot-swapping fixed code...\n";
    if (tensorlang_compile(fixed_ir) == 0) {
      llvm::errs() << "[System 2] Success! Logic updated. Restarting simulation...\n";
      void* new_main = tensorlang_get_symbol_address("main");
      if (new_main) {
        ((int(*)())new_main)();
        exit(0);
      }
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
