#include "tensorlang/Runtime/JitContext.h"
#include <mutex>
#include <fstream>
#include <filesystem>
#include <sstream>
#include <thread>

namespace fs = std::filesystem;

namespace mlir {
namespace tensorlang {

JitContext& JitContext::getInstance() {
  static JitContext instance;
  return instance;
}

void JitContext::registerRunner(JitRunner* runner) {
  this->runner = runner;
}

JitRunner* JitContext::getRunner() const {
  return runner;
}

void JitContext::setModelRunner(std::unique_ptr<ModelRunner> mr) {
  modelRunner = std::move(mr);
}

ModelRunner* JitContext::getModelRunner() const {
  return modelRunner.get();
}

void JitContext::setModuleIR(const std::string& ir) {
  currentIR = ir;
}

std::string JitContext::getModuleIR() const {
  return currentIR;
}

void JitContext::setOptimizedFunction(void* fnPtr) {
  optimizedFunctionPtr.store(fnPtr, std::memory_order_release);
}

void* JitContext::getOptimizedFunction() const {
  return optimizedFunctionPtr.load(std::memory_order_acquire);
}

bool JitContext::tryStartOptimization() {
  bool expected = false;
  return isOptimizing.compare_exchange_strong(expected, true);
}

void JitContext::finishOptimization() {
  isOptimizing.store(false);
}

void JitContext::saveLobe(const std::string& name, const std::string& ir) {
  // L1: Update Memory Cache immediately
  {
    std::lock_guard<std::mutex> lock(cacheMutex);
    lobeCache[name] = ir;
  }

  // L2: Async Write-Back to Disk
  std::thread([name, ir]() {
    const char* home_env = std::getenv("HOME");
    if (!home_env) return;
    std::string home(home_env);
    std::string dir = home + "/.neurojit/registry";
    fs::create_directories(dir);
    
    std::ofstream out(dir + "/" + name + ".mlir");
    out << ir;
    out.close();
    printf("[Registry] Lobe persisted to disk: %s\n", name.c_str());
  }).detach();
}

std::string JitContext::loadLobe(const std::string& name) {
  // 1. Try L1 Memory Cache
  {
    std::lock_guard<std::mutex> lock(cacheMutex);
    if (lobeCache.count(name)) {
      printf("[Registry] L1 Cache Hit: %s\n", name.c_str());
      return lobeCache[name];
    }
  }

  // 2. Fallback to L2 Disk Registry
  const char* home_env = std::getenv("HOME");
  if (!home_env) return "";
  std::string home(home_env);
  std::string path = home + "/.neurojit/registry/" + name + ".mlir";
  
  std::ifstream in(path);
  if (!in.is_open()) return "";
  
  std::stringstream ss;
  ss << in.rdbuf();
  std::string ir = ss.str();
  
  // Backfill L1 Cache
  {
    std::lock_guard<std::mutex> lock(cacheMutex);
    lobeCache[name] = ir;
  }

  printf("[Registry] L2 Disk Hit & L1 Backfill: %s\n", name.c_str());
  return ir;
}

bool JitContext::hasLobe(const std::string& name) {
  {
    std::lock_guard<std::mutex> lock(cacheMutex);
    if (lobeCache.count(name)) return true;
  }
  
  const char* home_env = std::getenv("HOME");
  if (!home_env) return false;
  std::string home(home_env);
  std::string path = home + "/.neurojit/registry/" + name + ".mlir";
  return fs::exists(path);
}

} // namespace tensorlang
} // namespace mlir
