#include "tensorlang/Runtime/ModelRunner.h"
#include <iostream>

namespace mlir {
namespace tensorlang {

// std::unique_ptr<ModelRunner> ModelRunner::create(const std::string& type) { ... } moved to GeminiModelRunner.cpp

int MockModelRunner::load(const std::string& modelPath) {
  std::cout << "[MockModelRunner] Loading model from: " << modelPath << std::endl;
  return 0; // Success
}

std::string MockModelRunner::query(const std::string& prompt) {
  std::cout << "[MockModelRunner] Received prompt:\n" << prompt << std::endl;
  
  if (prompt.find("Optimize") != std::string::npos) {
    return R"(module {
  llvm.func @printf(!llvm.ptr, ...) -> i32
  llvm.mlir.global internal constant @str_opt("Optimized code running! (Optimization successful, kernel replaced)\0A\00")
  
  func.func @main_optimized() -> i32 {
    %fmt = llvm.mlir.addressof @str_opt : !llvm.ptr
    %0 = llvm.call @printf(%fmt) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
    %ret = arith.constant 0 : i32
    return %ret : i32
  }
})";
  }
  
  return "(null)";
}

} // namespace tensorlang
} // namespace mlir
