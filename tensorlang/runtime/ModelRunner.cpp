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
  // Return a JSON mutation as expected by ASTMutator
  return R"({
    "target_function": "matmul",
    "mutations": [
      { "type": "unroll", "loop_depth": 1, "factor": 4 }
    ]
  })";
}

} // namespace tensorlang
} // namespace mlir
