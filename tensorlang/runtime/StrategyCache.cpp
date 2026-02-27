#include "tensorlang/Runtime/StrategyCache.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#define JSON_NOEXCEPTION 1
#include "nlohmann/json.hpp"

#include <iostream>
#include <fstream>
#include <regex>
#include <cstdlib>

using json = nlohmann::json;

namespace mlir {
namespace tensorlang {

StrategyCache::StrategyCache() {}

std::string StrategyCache::getCacheFilePath() const {
  const char* homeDir = std::getenv("HOME");
  if (!homeDir) return "cache.json"; // Fallback to current directory

  std::string dir = std::string(homeDir) + "/.neurojit";
  if (!llvm::sys::fs::exists(dir)) {
    llvm::sys::fs::create_directory(dir);
  }
  return dir + "/cache.json";
}

std::string StrategyCache::hash(const std::string& normalizedIR) const {
  llvm::SHA256 Hasher;
  Hasher.update(normalizedIR);
  std::array<uint8_t, 32> Hash = Hasher.final();
  
  std::string hex;
  hex.reserve(64);
  const char* hexChars = "0123456789abcdef";
  for (uint8_t byte : Hash) {
    hex.push_back(hexChars[(byte >> 4) & 0x0F]);
    hex.push_back(hexChars[byte & 0x0F]);
  }
  return hex;
}

std::string StrategyCache::normalizeIR(const std::string& ir) {
  // 1. Remove line comments //
  std::regex comment_regex("//.*");
  std::string no_comments = std::regex_replace(ir, comment_regex, "");

  // 2. Replace all whitespace (newlines, tabs, multiple spaces) with a single space
  std::regex whitespace_regex("\\s+");
  std::string normalized = std::regex_replace(no_comments, whitespace_regex, " ");

  // 3. Trim leading/trailing whitespace
  if (!normalized.empty()) {
    normalized.erase(0, normalized.find_first_not_of(" "));
    normalized.erase(normalized.find_last_not_of(" ") + 1);
  }

  return normalized;
}

void StrategyCache::load() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (loaded_) return;

  std::string path = getCacheFilePath();
  std::ifstream file(path);
  if (!file.is_open()) {
    loaded_ = true;
    return;
  }

  std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  json j = json::parse(content, nullptr, false);
  
  if (!j.is_discarded() && j.is_object()) {
    for (auto& el : j.items()) {
      if (el.value().is_string()) {
        cache_[el.key()] = el.value().get<std::string>();
      }
    }
  }

  loaded_ = true;
  llvm::errs() << "[StrategyCache] Loaded " << cache_.size() << " entries from " << path << "\n";
}

void StrategyCache::save() {
  std::lock_guard<std::mutex> lock(mutex_);
  
  json j = json::object();
  for (const auto& pair : cache_) {
    j[pair.first] = pair.second;
  }

  std::string path = getCacheFilePath();
  std::ofstream file(path);
  if (file.is_open()) {
    file << j.dump(2);
    llvm::errs() << "[StrategyCache] Saved to " << path << "\n";
  } else {
    llvm::errs() << "[StrategyCache] Error: Could not open " << path << " for writing.\n";
  }
}

std::string StrategyCache::lookup(const std::string& originalIR) {
  if (!loaded_) load();
  
  std::string normalized = normalizeIR(originalIR);
  std::string h = hash(normalized);

  std::lock_guard<std::mutex> lock(mutex_);
  auto it = cache_.find(h);
  if (it != cache_.end()) {
    return it->second;
  }
  return "";
}

void StrategyCache::insert(const std::string& originalIR, const std::string& optimizedIR) {
  if (!loaded_) load();

  std::string normalized = normalizeIR(originalIR);
  std::string h = hash(normalized);

  {
    std::lock_guard<std::mutex> lock(mutex_);
    cache_[h] = optimizedIR;
  }
  save(); // Persist immediately for this version
}

} // namespace tensorlang
} // namespace mlir
