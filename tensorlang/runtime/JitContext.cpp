#include "tensorlang/Runtime/JitContext.h"
#include "tensorlang/Runtime/MLIRTemplates.h"
#include <fstream>
#include <filesystem>
#include <sstream>
#include <thread>
#include <numeric>
#include <cstdlib>

namespace fs = std::filesystem;

namespace mlir {
namespace tensorlang {

// ---------------------------------------------------------------------------
// Worker process function (defined in OptimizationWorker.cpp, declared here
// so JitContext can wire the callback)
// ---------------------------------------------------------------------------

JitContext::JitContext() {
  ga_ = std::make_unique<StrategyGA>();
  ga_->seedRandom();
  initWorker();
}

void JitContext::initWorker() {
  // The hot-swap callback runs on the worker thread.
  // It atomically updates the function pointer so the sim loop picks it up.
  worker_ = std::make_unique<OptimizationWorker>(
      [this](void* fnPtr, const ControlStrategy& s) {
        setOptimizedFunction(fnPtr);
        printf("[Worker] Hot-swap complete. New strategy active.\n");
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

void JitContext::recordResult(const SimulationResult& result) {
  std::lock_guard<std::mutex> lock(statsMutex_);
  history_.push_back(result);
  if (result.score() > bestIndividual_.result.score()) {
    bestIndividual_.result = result;
  }
}

SimulationResult JitContext::getLastResult() const {
  std::lock_guard<std::mutex> lock(statsMutex_);
  if (history_.empty()) return {};
  return history_.back();
}

double JitContext::getAverageScore() const {
  std::lock_guard<std::mutex> lock(statsMutex_);
  if (history_.empty()) return -1000.0;
  double sum = 0.0;
  for (auto& r : history_) sum += r.score();
  return sum / history_.size();
}

double JitContext::getBestScore() const {
  std::lock_guard<std::mutex> lock(statsMutex_);
  return bestIndividual_.result.score();
}

const Individual& JitContext::getBestIndividual() const {
  std::lock_guard<std::mutex> lock(statsMutex_);
  return bestIndividual_;
}

OptimizationWorker& JitContext::getWorker() { return *worker_; }
StrategyGA& JitContext::getGA() { return *ga_; }

// ---------------------------------------------------------------------------
// Lobe Registry
// ---------------------------------------------------------------------------

static std::string lobeDir() {
  const char* home = std::getenv("HOME");
  if (!home) return "/tmp/.neurojit/registry";
  return std::string(home) + "/.neurojit/registry";
}

void JitContext::saveLobe(const std::string& name,
                          const std::string& ir,
                          const SimulationResult& result) {
  // L1: update immediately
  {
    std::lock_guard<std::mutex> lock(lobeMutex_);
    lobeCache_[name] = ir;
    lobeResults_[name] = result;
  }

  // L2: async write with atomic rename (prevents corrupt reads on crash)
  std::thread([name, ir, result]() {
    std::string dir = lobeDir();
    fs::create_directories(dir);
    std::string tmp  = dir + "/" + name + ".mlir.tmp";
    std::string dest = dir + "/" + name + ".mlir";
    {
      std::ofstream out(tmp);
      if (!out) {
        fprintf(stderr, "[Registry] ERROR: cannot write to %s\n", tmp.c_str());
        return;
      }
      out << ir;
    } // close before rename
    // std::rename is POSIX-atomic: either old or new, never partial
    if (std::rename(tmp.c_str(), dest.c_str()) != 0) {
      fprintf(stderr, "[Registry] ERROR: rename failed for %s\n", dest.c_str());
    } else {
      printf("[Registry] Saved: %s (score=%.2f)\n", name.c_str(), result.score());
    }
  }).detach();
}

std::string JitContext::loadLobe(const std::string& name) {
  // L1 first
  {
    std::lock_guard<std::mutex> lock(lobeMutex_);
    auto it = lobeCache_.find(name);
    if (it != lobeCache_.end()) {
      printf("[Registry] L1 hit: %s\n", name.c_str());
      return it->second;
    }
  }
  // L2 fallback
  std::string path = lobeDir() + "/" + name + ".mlir";
  std::ifstream in(path);
  if (!in.is_open()) return "";
  std::ostringstream ss;
  ss << in.rdbuf();
  std::string ir = ss.str();
  if (ir.empty()) return "";
  // Backfill L1
  {
    std::lock_guard<std::mutex> lock(lobeMutex_);
    lobeCache_[name] = ir;
  }
  printf("[Registry] L2 hit & L1 backfill: %s\n", name.c_str());
  return ir;
}

bool JitContext::hasLobe(const std::string& name) {
  {
    std::lock_guard<std::mutex> lock(lobeMutex_);
    if (lobeCache_.count(name)) return true;
  }
  return fs::exists(lobeDir() + "/" + name + ".mlir");
}

SimulationResult JitContext::loadLobeResult(const std::string& name) {
  std::lock_guard<std::mutex> lock(lobeMutex_);
  auto it = lobeResults_.find(name);
  if (it != lobeResults_.end()) return it->second;
  return {};
}

void JitContext::saveLobeStrategy(const std::string& name,
                                   const ControlStrategy& s) {
  // Format as JSON manually — nlohmann already in project via StrategyCache
  // Using manual format to avoid adding an include here
  char buf[512];
  std::snprintf(buf, sizeof(buf),
      "{\"kp\":%.6f,\"ki\":%.6f,\"kd\":%.6f,"
      "\"target_velocity\":%.6f,\"thrust_clamp_max\":%.6f}\n",
      s.kp, s.ki, s.kd, s.targetVelocity, s.thrustClampMax);
  std::string json(buf);

  std::thread([name, json]() {
    std::string dir = lobeDir();
    fs::create_directories(dir);
    std::string tmp  = dir + "/" + name + ".json.tmp";
    std::string dest = dir + "/" + name + ".json";
    {
      std::ofstream out(tmp);
      if (!out) return;
      out << json;
    }
    std::rename(tmp.c_str(), dest.c_str());
    printf("[Registry] Strategy JSON saved: %s\n", name.c_str());
  }).detach();
}

ControlStrategy JitContext::loadLobeStrategy(const std::string& name) {
  std::string path = lobeDir() + "/" + name + ".json";
  std::ifstream in(path);
  if (!in.is_open()) {
    printf("[Registry] No strategy JSON for '%s' — using defaults\n",
           name.c_str());
    return ControlStrategy{};
  }
  std::string json((std::istreambuf_iterator<char>(in)),
                    std::istreambuf_iterator<char>());

  // Simple field extraction — same pattern as StrategyGA::llmMutate
  auto extract = [&](const std::string& key) -> float {
    auto pos = json.find("\"" + key + "\"");
    if (pos == std::string::npos) return 0.0f;
    auto colon = json.find(':', pos);
    if (colon == std::string::npos) return 0.0f;
    const char* start = json.c_str() + colon + 1;
    char* end;
    float val = std::strtof(start, &end);
    if (end == start) return 0.0f;
    return val;
  };

  ControlStrategy s;
  s.kp             = extract("kp");
  s.ki             = extract("ki");
  s.kd             = extract("kd");
  s.targetVelocity = extract("target_velocity");
  s.thrustClampMax = extract("thrust_clamp_max");

  printf("[Registry] Loaded strategy: kp=%.3f ki=%.3f kd=%.3f\n",
         s.kp, s.ki, s.kd);
  return s;
}

} // namespace tensorlang
} // namespace mlir