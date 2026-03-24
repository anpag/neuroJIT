#include "tensorlang/ExecutionEngine/JitRunner.h"
#include "tensorlang/Runtime/JitContext.h"
#include "tensorlang/Runtime/ModelRunner.h"
#include "tensorlang/Runtime/OptimizationWorker.h"
#include "tensorlang/Runtime/MatmulSpeedEvaluator.h"
#include "llvm/Support/Error.h"
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <thread>

using namespace mlir::tensorlang;

int main(int argc, char** argv) {
  std::cout << "[NeuroJIT] Initializing Seed AI MCTS Engine..." << std::endl;

  auto runnerOrErr = JitRunner::create();
  if (!runnerOrErr) {
    std::cerr << "FAILED TO CREATE JIT RUNNER: " << llvm::toString(runnerOrErr.takeError()) << std::endl;
    return 1;
  }
  auto executionRunner = std::move(*runnerOrErr);
  std::cout << "[NeuroJIT] JIT Runner initialized." << std::endl;

  std::cout << "[NeuroJIT] Accessing JitContext..." << std::endl;
  auto& context = JitContext::getInstance();
  context.registerRunner(executionRunner.get());
  std::cout << "[NeuroJIT] JitContext accessed and Runner registered." << std::endl;

  std::cout << "[NeuroJIT] Creating AI Runner..." << std::endl;
  auto runner = ModelRunner::create("llama");
  if (runner->load("tensorlang/runtime/models/deepseek-r1-32b-q4_k_m.gguf") != 0) {
      std::cerr << "Failed to load DeepSeek model.\n";
      return 1;
  }
  context.setModelRunner(std::move(runner));

  std::string mlirPath = "tensorlang/benchmarks/matmul_pure.mlir";
  std::cout << "[NeuroJIT] Reading MLIR from: " << mlirPath << std::endl;
  std::ifstream ifs(mlirPath);
  std::string mlirContent((std::istreambuf_iterator<char>(ifs)), (std::istreambuf_iterator<char>()));
  context.setModuleIR(mlirContent);

  std::cout << "[NeuroJIT] Creating Evaluator..." << std::endl;
  MatmulSpeedEvaluator evaluator;

  std::cout << "[NeuroJIT] Getting worker from JitContext..." << std::endl;
  auto& worker = context.getWorker();

  std::cout << "[NeuroJIT] Setting up OptimizationRequest..." << std::endl;
  OptimizationRequest req;
  req.originalIR = mlirContent;
  req.functionName = "matmul";
  req.evaluator = &evaluator;

  std::cout << "[NeuroJIT] Submitting request..." << std::endl;
  worker.submit(req);
  std::cout << "[NeuroJIT] Request submitted." << std::endl;

  // Keep the main thread alive while the background MCTS threads explore
  std::cout << "[NeuroJIT] Main thread entering holding pattern. Engine is LIVE." << std::endl;
  
  // Wait until the worker starts processing
  while (!worker.isBusy()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  
  // Wait until the worker finishes
  while (worker.isBusy()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  std::cout << "[NeuroJIT] Test run complete. Shutting down." << std::endl;
  context.shutdown();
  return 0;
}
