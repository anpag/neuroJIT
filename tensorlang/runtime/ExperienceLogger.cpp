#include "tensorlang/Runtime/ExperienceLogger.h"
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <mutex>

namespace mlir {
namespace tensorlang {

static std::mutex gLoggerMutex;

// Minimal escape logic to build safe JSON strings without a full library dependency
static std::string escapeJSON(const std::string& input) {
  std::ostringstream ss;
  for (char c : input) {
    if (c == '"') ss << "\\\"";
    else if (c == '\\') ss << "\\\\";
    else if (c == '\n') ss << "\\n";
    else if (c == '\r') ss << "\\r";
    else if (c == '\t') ss << "\\t";
    else ss << c;
  }
  return ss.str();
}

static std::string getCurrentTimestamp() {
  auto now = std::chrono::system_clock::now();
  std::time_t now_time = std::chrono::system_clock::to_time_t(now);
  std::tm* gmt = std::gmtime(&now_time);
  std::ostringstream ss;
  ss << std::put_time(gmt, "%Y-%m-%dT%H:%M:%SZ");
  return ss.str();
}

void ExperienceLogger::logExperience(const ExperienceRecord& record) {
  std::lock_guard<std::mutex> lock(gLoggerMutex);
  std::ofstream out("tensorlang_training_data.jsonl", std::ios::app);
  if (!out.is_open()) return;

  out << "{"
      << "\"timestamp\":\"" << getCurrentTimestamp() << "\","
      << "\"episode\":" << record.episode << ","
      << "\"failure_type\":\"" << escapeJSON(record.failureType) << "\","
      << "\"ir_before\":\"" << escapeJSON(record.irBefore) << "\","
      << "\"full_prompt\":\"" << escapeJSON(record.fullPrompt) << "\","
      << "\"generated_patch\":\"" << escapeJSON(record.generatedPatch) << "\","
      << "\"compiled\":" << (record.compiled ? "true" : "false") << ","
      << "\"sandbox_passed\":" << (record.sandboxPassed ? "true" : "false") << ","
      << "\"performance_delta_ms\":" << record.performanceDeltaMs << ","
      << "\"reward\":" << record.reward << ","
      << "\"model_name\":\"" << escapeJSON(record.modelName) << "\","
      << "\"adapter_version\":" << record.adapterVersion
      << "}\n";
}

} // namespace tensorlang
} // namespace mlir