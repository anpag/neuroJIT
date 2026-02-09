#ifndef TENSORLANG_RUNTIME_MODELRUNNER_H
#define TENSORLANG_RUNTIME_MODELRUNNER_H

#include <string>
#include <memory>

namespace mlir {
namespace tensorlang {

/// Abstract interface for running LLM inference.
class ModelRunner {
public:
  virtual ~ModelRunner() = default;

  /// Loads the model from the specified path.
  /// Returns 0 on success, non-zero on failure.
  virtual int load(const std::string& modelPath) = 0;

  /// Runs inference on the given prompt.
  /// Returns the generated text.
  virtual std::string query(const std::string& prompt) = 0;

  /// Factory method to create a specific runner type.
  static std::unique_ptr<ModelRunner> create(const std::string& type);
};

/// A mock runner for testing without a GPU/Model.
class MockModelRunner : public ModelRunner {
public:
  int load(const std::string& modelPath) override;
  std::string query(const std::string& prompt) override;
};

} // namespace tensorlang
} // namespace mlir

#endif // TENSORLANG_RUNTIME_MODELRUNNER_H
