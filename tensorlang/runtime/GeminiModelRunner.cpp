#include "tensorlang/Runtime/ModelRunner.h"
#include <iostream>

namespace mlir {
namespace tensorlang {

class GeminiModelRunner : public ModelRunner {
public:
  GeminiModelRunner() {
    std::cerr << "[GeminiRunner] Error: CURL not found during build. Gemini runner is disabled." << std::endl;
  }

  int load(const std::string& modelPath) override {
    return -1;
  }

  std::string query(const std::string& input_code) override {
    return "(error: gemini runner disabled)";
  }
};

std::unique_ptr<ModelRunner> createGeminiModelRunner() {
  return std::make_unique<GeminiModelRunner>();
}

} // namespace tensorlang
} // namespace mlir
