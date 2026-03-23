#include "tensorlang/Runtime/OptimizationWorker.h"
#include "tensorlang/Runtime/JitContext.h"
#include "tensorlang/Runtime/ModelRunner.h"
#include "tensorlang/Runtime/VerificationSandbox.h"
#include "tensorlang/Runtime/ExperienceLogger.h"
#include "tensorlang/Runtime/ASTMutator.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "nlohmann/json.hpp"
#include <cstdio>
#include <future>
#include <cmath>
#include <algorithm>
#include <mutex>
#include <vector>

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

extern "C" {
int tensorlang_compile(const char* ir_string);
void* tensorlang_get_symbol_address(const char* name);
}

namespace mlir {
namespace tensorlang {

OptimizationWorker::OptimizationWorker(HotSwapCallback cb)
    : hotSwapCb_(std::move(cb)),
      queueMutex_(std::make_unique<std::mutex>()),
      cv_(std::make_unique<std::condition_variable>()) {
  thread_ = std::thread(&OptimizationWorker::workerLoop, this);
}

OptimizationWorker::~OptimizationWorker() {
  {
    std::lock_guard<std::mutex> lock(*queueMutex_);
    shutdown_.store(true);
  }
  cv_->notify_all();
  if (thread_.joinable()) thread_.join();
}

bool OptimizationWorker::submit(OptimizationRequest req) {
  printf("[Worker] submit() called for '%s'\n", req.functionName.c_str());
  auto reqPtr = std::make_unique<OptimizationRequest>(std::move(req));
  {
    std::lock_guard<std::mutex> lock(*queueMutex_);
    if (busy_.load(std::memory_order_acquire) || !queue_.empty()) {
      printf("[Worker] Busy, dropping optimization request for '%s'\n", reqPtr->functionName.c_str());
      return false;
    }
    printf("[Worker] Pushing request to queue.\n");
    queue_.push(std::move(reqPtr));
  }
  printf("[Worker] Notifying cv.\n");
  cv_->notify_one();
  printf("[Worker] submit() returning true.\n");
  return true;
}

void OptimizationWorker::workerLoop() {
  printf("[Worker] Loop started.\n");
  while (true) {
    std::unique_ptr<OptimizationRequest> req;
    {
      std::unique_lock<std::mutex> lock(*queueMutex_);
      printf("[Worker] Waiting for request...\n");
      cv_->wait(lock, [this] {
        return !queue_.empty() || shutdown_.load();
      });
      printf("[Worker] Wake up!\n");
      if (shutdown_.load() && queue_.empty()) break;
      req = std::move(queue_.front());
      queue_.pop();
    }

    busy_.store(true, std::memory_order_release);
    printf("[Worker] Processing MCTS for '%s'\n", req->functionName.c_str());

    auto* runner = JitContext::getInstance().getModelRunner();
    if (!runner) {
      printf("[Worker] ERROR: No model runner available.\n");
      busy_.store(false, std::memory_order_release);
      continue;
    }

    class DummyEvaluator : public IEvaluator {
    public:
      float evaluate(void*) override { return 0.0f; }
    };
    DummyEvaluator dummyEval;
    IEvaluator* evaluator = req->evaluator ? req->evaluator : &dummyEval;

    auto root = std::make_shared<MCTSNode>();
    root->irState = req->originalIR;
    
    printf("[Worker] Evaluating root node...\n");
    auto rootRes = VerificationSandbox::verifyCandidate(root->irState, req->functionName, evaluator);
    printf("[Worker] Root node evaluated. Status: %d, Fitness: %f\n", (int)rootRes.status, rootRes.fitnessScore);
    
    float best_score_seen_so_far = rootRes.status == VerificationSandbox::VerificationResult::Success ? rootRes.fitnessScore : -10000.0f;
    root->totalScore = best_score_seen_so_far;
    root->visits = 1;

    bool hotSwapped = false;
    const int MCTS_ITERATIONS = 5;

    for (int iter = 0; iter < MCTS_ITERATIONS && !hotSwapped && !shutdown_.load(); ++iter) {
      printf("[Worker] MCTS Iteration %d/%d\n", iter+1, MCTS_ITERATIONS);
      auto node = root;
      while (!node->children.empty()) {
        auto bestChild = node->children[0];
        float bestUCB1 = -1e9f;
        for (const auto& child : node->children) {
          if (child->visits == 0) {
            bestChild = child;
            break;
          }
          float exploitation = child->totalScore / child->visits;
          float exploration = 1.414f * std::sqrt(std::log(node->visits) / (float)child->visits);
          float ucb1 = exploitation + exploration;
          if (ucb1 > bestUCB1) {
            bestUCB1 = ucb1;
            bestChild = child;
          }
        }
        node = bestChild;
        if (node->visits == 0) break;
      }

      std::vector<std::string> newIRs;
      for (int i = 0; i < 2; ++i) {
        std::string rawJSON = runner->query(node->irState);
        if (rawJSON.empty()) continue;

        auto j = nlohmann::json::parse(rawJSON, nullptr, false);
        if (j.is_discarded()) continue;

        std::string action;
        float value = 0.0f;
        std::string new_op;
        if (j.contains("action") && j["action"].is_string()) action = j["action"];
        if (j.contains("value") && j["value"].is_number()) value = j["value"];
        if (j.contains("new_op") && j["new_op"].is_string()) new_op = j["new_op"];

        mlir::MLIRContext context;
        context.getOrLoadDialect<mlir::arith::ArithDialect>();
        context.getOrLoadDialect<mlir::func::FuncDialect>();
        context.getOrLoadDialect<mlir::scf::SCFDialect>();
        context.getOrLoadDialect<mlir::memref::MemRefDialect>();
        context.getOrLoadDialect<mlir::tensor::TensorDialect>();
        
        auto module = mlir::parseSourceString<mlir::ModuleOp>(node->irState, &context);
        if (!module) continue;

        std::optional<mlir::ModuleOp> mutatedOpt;
        if (action == "mutateConstant") {
          mutatedOpt = ASTMutator::mutateConstant(*module, req->functionName, 0, value);
        } else if (action == "swapBinaryOperator") {
          mutatedOpt = ASTMutator::swapBinaryOperator(*module, req->functionName, 0, new_op);
        }

        if (mutatedOpt) {
          std::string mutatedIR;
          llvm::raw_string_ostream os(mutatedIR);
          mutatedOpt->print(os);
          mutatedOpt->erase();
          newIRs.push_back(mutatedIR);
        }
      }

      if (newIRs.empty()) {
        auto curr = node;
        while (curr) {
          curr->visits++;
          curr->totalScore -= 10.0f;
          curr = curr->parent.lock();
        }
        continue;
      }

      std::vector<std::future<VerificationSandbox::Result>> futures;
      std::vector<std::shared_ptr<MCTSNode>> newChildren;

      for (const auto& ir : newIRs) {
        auto child = std::make_shared<MCTSNode>();
        child->irState = ir;
        child->parent = node;
        node->children.push_back(child);
        newChildren.push_back(child);
        futures.push_back(std::async(std::launch::async, 
          &VerificationSandbox::verifyCandidate, ir, req->functionName, evaluator));
      }

      for (size_t i = 0; i < futures.size(); ++i) {
        auto vRes = futures[i].get();
        auto child = newChildren[i];
        child->visits = 1;
        child->totalScore = vRes.fitnessScore;
        if (vRes.status == VerificationSandbox::VerificationResult::Success) {
          if (vRes.fitnessScore > best_score_seen_so_far) {
            best_score_seen_so_far = vRes.fitnessScore;
            if (tensorlang_compile(child->irState.c_str()) == 0) {
              void* fnPtr = tensorlang_get_symbol_address(req->functionName.c_str());
              if (fnPtr) {
                printf("[Worker] Found better fitness: %f! Hot-swapping...\n", best_score_seen_so_far);
                hotSwapCb_(fnPtr, child->irState);
                hotSwapped = true;
              }
            }
          }
        }
        auto curr = node;
        while (curr) {
          curr->visits++;
          curr->totalScore += vRes.fitnessScore;
          curr = curr->parent.lock();
        }
      }
    }
    processed_.fetch_add(1, std::memory_order_relaxed);
    busy_.store(false, std::memory_order_release);
  }
}

} // namespace tensorlang
} // namespace mlir
