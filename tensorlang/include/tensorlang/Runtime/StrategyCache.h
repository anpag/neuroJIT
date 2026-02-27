#ifndef TENSORLANG_RUNTIME_STRATEGYCACHE_H
#define TENSORLANG_RUNTIME_STRATEGYCACHE_H

#include <string>
#include <unordered_map>
#include <mutex>

namespace mlir {
namespace tensorlang {

class StrategyCache {
public:
  StrategyCache();
  
  // Look up an optimized IR given the original IR/Prompt
  std::string lookup(const std::string& originalIR);
  
  // Add a new strategy to the cache
  void insert(const std::string& originalIR, const std::string& optimizedIR);

  void load();
  void save();

  // Normalize IR for consistent hashing
  static std::string normalizeIR(const std::string& ir);

private:
  std::string getCacheFilePath() const;
  std::string hash(const std::string& normalizedIR) const;

  std::unordered_map<std::string, std::string> cache_;
  std::mutex mutex_;
  bool loaded_ = false;
};

} // namespace tensorlang
} // namespace mlir

#endif // TENSORLANG_RUNTIME_STRATEGYCACHE_H
