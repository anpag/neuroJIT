#include "tensorlang/Runtime/JitContext.h"
#include <fstream>
#include <filesystem>
#include <sstream>
#include <thread>
#include <cstdlib>

namespace fs = std::filesystem;

namespace mlir {
namespace tensorlang {

JitContext::JitContext() {
  initWorker();
}

void JitContext::initWorker() {
  worker_ = std::make_unique<OptimizationWorker>(
      [this](void* fnPtr, const std::string& newIR) {
        setOptimizedFunction(fnPtr);
        // Persist successfully compiled dynamic fixes
        saveLobe("latest_async_repair", newIR);
        printf("[Worker] Hot-swap complete. New logic is active.\n");
      });
}

JitContext& JitContext::getInstance() {
  static JitContext instance;
  return instance;
}

void JitContext::registerRunner(JitRunner* r) { runner_ = r; }
JitRunner* JitContext::getRunner() const { return runner_; }

void JitContext::setModelRunner(std::unique_ptr<ModelRunner> mr) {
  modelRunner_ = std::move(mr);
}
ModelRunner* JitContext::getModelRunner() const { return modelRunner_.get(); }

void JitContext::setModuleIR(const std::string& ir) {
  std::lock_guard<std::mutex> lock(irMutex_);
  currentIR_ = ir;
}

std::string JitContext::getModuleIR() const {
  std::lock_guard<std::mutex> lock(irMutex_);
  return currentIR_;
}

void JitContext::setOptimizedFunction(void* fnPtr) {
  optimizedFunctionPtr_.store(fnPtr, std::memory_order_release);
}

void* JitContext::getOptimizedFunction() const {
  return optimizedFunctionPtr_.load(std::memory_order_acquire);
}

void JitContext::requestRestart(const std::string& newIR) {
  std::lock_guard<std::mutex> lock(restartMutex_);
  restartPending_ = true;
  pendingIR_ = newIR;
}

bool JitContext::consumeRestartRequest(std::string& outNewIR) {
  std::lock_guard<std::mutex> lock(restartMutex_);
  if (!restartPending_) return false;
  outNewIR = std::move(pendingIR_);
  restartPending_ = false;
  return true;
}

void JitContext::shutdown() {
  worker_.reset(); // Joins the thread and destroys the worker safely
  modelRunner_.reset();
}

OptimizationWorker& JitContext::getWorker() { return *worker_; }

// ---------------------------------------------------------------------------
// Lobe Registry
// ---------------------------------------------------------------------------

static std::string lobeDir() {
  const char* home = std::getenv("HOME");
  if (!home) return "/tmp/.neurojit/registry";
  return std::string(home) + "/.neurojit/registry";
}

void JitContext::saveLobe(const std::string& name, const std::string& ir) {
  {
    std::lock_guard<std::mutex> lock(lobeMutex_);
    lobeCache_[name] = ir;
  }
  std::thread([name, ir]() {
    std::string dir = lobeDir();
    fs::create_directories(dir);
    std::string tmp  = dir + "/" + name + ".mlir.tmp";
    std::string dest = dir + "/" + name + ".mlir";
    {
      std::ofstream out(tmp);
      if (!out) return;
      out << ir;
    } 
    std::rename(tmp.c_str(), dest.c_str());
  }).detach();
}

std::string JitContext::loadLobe(const std::string& name) {
  {
    std::lock_guard<std::mutex> lock(lobeMutex_);
    auto it = lobeCache_.find(name);
    if (it != lobeCache_.end()) return it->second;
  }
  std::string path = lobeDir() + "/" + name + ".mlir";
  std::ifstream in(path);
  if (!in.is_open()) return "";
  std::ostringstream ss;
  ss << in.rdbuf();
  std::string ir = ss.str();
  if (!ir.empty()) {
    std::lock_guard<std::mutex> lock(lobeMutex_);
    lobeCache_[name] = ir;
  }
  return ir;
}

bool JitContext::hasLobe(const std::string& name) {
  {
    std::lock_guard<std::mutex> lock(lobeMutex_);
    if (lobeCache_.count(name)) return true;
  }
  return fs::exists(lobeDir() + "/" + name + ".mlir");
}

} // namespace tensorlang
} // namespace mlir