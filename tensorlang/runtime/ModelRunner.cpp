#include "tensorlang/Runtime/ModelRunner.h"
#include "tensorlang/Runtime/Runners.h"
#include <iostream>

namespace mlir {
namespace tensorlang {

std::unique_ptr<ModelRunner> ModelRunner::create(const std::string& type) {
  if (type == "gemini") {
    return createGeminiModelRunner();
  } else if (type == "llama") {
    return createLlamaCppModelRunner();
  }
  return std::make_unique<MockModelRunner>();
}

int MockModelRunner::load(const std::string& modelPath) {
  std::cout << "[MockModelRunner] Loading model from: " << modelPath << std::endl;
  return 0; // Success
}

std::string MockModelRunner::query(const std::string& prompt) {
  if (prompt.find("Lunar Lander") != std::string::npos) {
    return R"(module {
  llvm.func @printf(!llvm.ptr, ...) -> i32
  llvm.mlir.global internal constant @msg("------------------------------------------------\0A[NeuroJIT] SYSTEM RECOVERY SUCCESSFUL\0A[NeuroJIT] Logic Patched: Soft Landing Sequence Engaged\0A[NeuroJIT] Simulation Finished Safely.\0A------------------------------------------------\0A\00")
  
  func.func @main() -> i32 {
    %fmt = llvm.mlir.addressof @msg : !llvm.ptr
    llvm.call @printf(%fmt) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
    %c0 = arith.constant 0 : i32
    return %c0 : i32
  }
})";
  }
  
  if (prompt.find("Optimize") != std::string::npos) {
    return R"(module {
      func.func @main() -> i32 {
        %c0 = arith.constant 0 : i32
        return %c0 : i32
      }
    })";
  }
  
  return "(null)";
}

} // namespace tensorlang
} // namespace mlir
