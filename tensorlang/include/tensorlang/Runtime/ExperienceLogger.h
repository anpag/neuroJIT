#ifndef TENSORLANG_RUNTIME_EXPERIENCELOGGER_H
#define TENSORLANG_RUNTIME_EXPERIENCELOGGER_H

#include <string>

namespace mlir {
namespace tensorlang {

struct ExperienceRecord {
  int episode = 0;
  std::string failureType; // "assert_fail", "compile_error", "semantic_fail", "timeout", "optimization"
  std::string irBefore;
  std::string fullPrompt;
  std::string generatedPatch;
  bool compiled = false;
  bool sandboxPassed = false;
  double performanceDeltaMs = 0.0;
  double reward = 0.0;
  std::string modelName = "qwen2.5-coder-7b";
  int adapterVersion = 0;
};

class ExperienceLogger {
public:
  /// Appends the experience record as a JSON line to tensorlang_training_data.jsonl
  static void logExperience(const ExperienceRecord& record);
};

} // namespace tensorlang
} // namespace mlir

#endif // TENSORLANG_RUNTIME_EXPERIENCELOGGER_H