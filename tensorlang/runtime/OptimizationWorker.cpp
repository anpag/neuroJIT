#include "tensorlang/Runtime/OptimizationWorker.h"
#include "tensorlang/Runtime/JitContext.h"
#include "tensorlang/Runtime/ModelRunner.h"
#include "tensorlang/Runtime/VerificationSandbox.h"
#include "tensorlang/Runtime/ExperienceLogger.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdio>

// Forward declaration — defined in Runtime.cpp
extern "C" {
int tensorlang_compile(const char* ir_string);
void* tensorlang_get_symbol_address(const char* name);
}

namespace mlir {
namespace tensorlang {

OptimizationWorker::OptimizationWorker(HotSwapCallback cb)
    : hotSwapCb_(std::move(cb)) {
  thread_ = std::thread(&OptimizationWorker::workerLoop, this);
}

OptimizationWorker::~OptimizationWorker() {
  {
    std::lock_guard<std::mutex> lock(queueMutex_);
    shutdown_.store(true);
  }
  cv_.notify_all();
  if (thread_.joinable()) thread_.join();
}

void OptimizationWorker::submit(OptimizationRequest req) {
  {
    std::lock_guard<std::mutex> lock(queueMutex_);
    if (!queue_.empty() && queue_.back().functionName == req.functionName) {
      printf("[Worker] Dropping duplicate request for '%s'\n", req.functionName.c_str());
      return;
    }
    queue_.push(std::move(req));
  }
  cv_.notify_one();
}

void OptimizationWorker::workerLoop() {
  while (true) {
    OptimizationRequest req;
    {
      std::unique_lock<std::mutex> lock(queueMutex_);
      cv_.wait(lock, [this] {
        return !queue_.empty() || shutdown_.load();
      });
      if (shutdown_.load() && queue_.empty()) break;
      req = std::move(queue_.front());
      queue_.pop();
    }

    busy_.store(true, std::memory_order_release);
    printf("[Worker] Processing %s for '%s'\n",
           req.errorMessage.empty() ? "optimization" : "crash repair",
           req.functionName.c_str());

    auto* runner = JitContext::getInstance().getModelRunner();
    if (!runner) {
      printf("[Worker] ERROR: No model runner available.\n");
      busy_.store(false, std::memory_order_release);
      continue;
    }

    std::string prompt = "You are an expert MLIR compiler engineer.\n";
    if (!req.errorMessage.empty()) {
      prompt += "The following MLIR module caused an assertion failure at runtime: " + req.errorMessage + "\n";
      prompt += "Rewrite the function '" + req.functionName + "' to fix the bug and prevent the crash.\n";
    } else {
      prompt += "Optimize the function '" + req.functionName + "' in the following MLIR module.\n";
    }
    prompt += "Return the FULL, syntactically valid MLIR module. Wrap it in ```mlir ... ``` tags.\n";
    prompt += "Original IR:\n```mlir\n" + req.originalIR + "\n```\n";

    std::string newIR = runner->query(prompt);

    ExperienceRecord logRecord;
    logRecord.episode = processed_.load();
    logRecord.failureType = req.errorMessage.empty() ? "optimization" : "assert_fail";
    logRecord.irBefore = req.originalIR;
    logRecord.fullPrompt = prompt;
    logRecord.generatedPatch = newIR;

    if (newIR.empty()) {
      printf("[Worker] LLM returned empty IR — skipping\n");
      busy_.store(false, std::memory_order_release);
      ExperienceLogger::logExperience(logRecord);
      continue;
    }

    if (VerificationSandbox::verifyCandidate(newIR)) {
      logRecord.sandboxPassed = true;
      logRecord.reward = 1.0;
      // It passed verification, compile it into the main context and swap!
      if (tensorlang_compile(newIR.c_str()) == 0) {
        logRecord.compiled = true;
        void* fnPtr = tensorlang_get_symbol_address(req.functionName.c_str());
        if (fnPtr) {
          hotSwapCb_(fnPtr, newIR);
        } else {
          printf("[Worker] Main context compiled but symbol '%s' not found\n", req.functionName.c_str());
        }
      } else {
        printf("[Worker] Main context compilation failed despite Sandbox success.\n");
      }
    } else {
      logRecord.sandboxPassed = false;
      logRecord.reward = -1.0;
      printf("[Worker] Verification Sandbox rejected LLM logic. Dropping patch.\n");
    }

    ExperienceLogger::logExperience(logRecord);

    processed_.fetch_add(1, std::memory_order_relaxed);
    busy_.store(false, std::memory_order_release);
  }
  printf("[Worker] Thread exiting cleanly\n");
}

} // namespace tensorlang
} // namespace mlir