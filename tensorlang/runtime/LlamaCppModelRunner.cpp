#include "tensorlang/Runtime/ModelRunner.h"
#include "llama.h"
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <chrono>
#include <fstream>
#include <mutex>
#include <sys/stat.h>

namespace mlir {
namespace tensorlang {

class LlamaCppModelRunner : public ModelRunner {
public:
  LlamaCppModelRunner() {
    ggml_backend_load_all();
  }

  ~LlamaCppModelRunner() {
    if (brainModel) llama_model_free(brainModel);
    if (muscleModel) llama_model_free(muscleModel);
  }

  bool fileExists(const std::string& name) {
    struct stat buffer;   
    return (stat (name.c_str(), &buffer) == 0); 
  }

  int load(const std::string& modelPath) override {
    // FINAL GOLDEN PAIR (Feb 2026)
    // Brain: DeepSeek-R1 (32B Distill) - Complex Logic
    // Muscle: Qwen 2.5 Coder (7B) - High Reliability MLIR Syntax
    
    std::string brainPath = "tensorlang/runtime/models/deepseek-r1-32b-q4_k_m.gguf";
    std::string musclePath = "tensorlang/runtime/models/qwen2.5-coder-7b-instruct-q4_k_m.gguf";

    std::cout << "[NeuroJIT] Initializing Golden Multi-Agent Pair..." << std::endl;
    std::cout << "  - Brain (Logic):  " << brainPath << std::endl;
    std::cout << "  - Muscle (MLIR): " << musclePath << std::endl;
    
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    
    brainModel = llama_model_load_from_file(brainPath.c_str(), mparams);
    muscleModel = llama_model_load_from_file(musclePath.c_str(), mparams);

    return (brainModel && muscleModel) ? 0 : -1;
  }

  std::string query(const std::string& prompt) override {
    std::lock_guard<std::mutex> lock(queryMutex);
    
    // Using Fresh Context Pattern for continuous evolution stability
    llama_context* muscleCtx = createContext(muscleModel);
    if (!muscleCtx) return "(error: muscle context creation failed)";

    std::string plan;
    std::string mlir_raw;

    // Fast-Repair Bypass (With Chain-of-Thought)
    if (prompt.find("SYNTAX REPAIR MODE:") != std::string::npos) {
      std::cout << "[Evolution] Fast-Repair Bypass Triggered: Routing JIT diagnostics directly to Muscle (CoT enabled)..." << std::endl;
      std::stringstream repair_ss;
      repair_ss << "<|im_start|>system\n"
                << "You are an MLIR specialist. You must REASON about the provided syntax error before fixing it.\n"
                << "1. Use scalar f32 math.\n"
                << "2. Address the specific loc() error provided.\n"
                << "3. Ensure all function arguments and constants use valid MLIR delimiters (,).\n"
                << "<|im_end|>\n"
                << "<|im_start|>user\n"
                << "ERROR AND CODE:\n" << prompt << "\n"
                << "Analyze the error and return the FULL corrected func.func @get_thrust block.\n"
                << "<|im_end|>\n"
                << "<|im_start|>assistant\n"
                << "<think>\n"; // Force the model to start thinking
      mlir_raw = runInference(muscleCtx, muscleModel, repair_ss.str(), 1024);
    } else {
      // --- STEP 1: RAPID SYNTHESIS (Qwen only for Gen 1) ---
      std::cout << "[Evolution] Rapid Synthesis Mode: Routing directly to Muscle..." << std::endl;
      std::stringstream rapid_ss;
      rapid_ss << "<|im_start|>system\n"
               << "You are an MLIR specialist. Evolve a scalar control system for 100 landers.\n"
               << "Objective: Soft landing using PD control arithmetic.\n"
               << "Return ONLY the func.func @get_thrust block.\n"
               << "RULES:\n"
               << "1. Use scalar f32 math.\n"
               << "2. Use arith.constant, arith.addf, arith.mulf.\n"
               << "<|im_end|>\n"
               << "<|im_start|>user\n"
               << "IMPLEMENT PD CONTROL:\n" << prompt << "\n"
               << "<|im_end|>\n"
               << "<|im_start|>assistant\n";
      
      mlir_raw = runInference(muscleCtx, muscleModel, rapid_ss.str(), 1024);
    }
    
    llama_free(muscleCtx);

    return extractAndWrap(mlir_raw);
  }

private:
  std::mutex queryMutex;
  llama_model* brainModel = nullptr;
  llama_model* muscleModel = nullptr;

  llama_context* createContext(llama_model* m, int n_ctx = 2048) {
    if (!m) return nullptr;
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = n_ctx;
    cparams.n_threads = 64;
    cparams.n_threads_batch = 64; // Saturation for prompt eval
    return llama_init_from_model(m, cparams);
  }

  std::string runInference(llama_context* ctx, llama_model* model, const std::string& prompt, int n_predict) {
    const llama_vocab* vocab = llama_model_get_vocab(model);
    // add_special = false, parse_special = true (Tokenization Fix)
    int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, false, true);
    std::vector<llama_token> tokens(n_prompt);
    llama_tokenize(vocab, prompt.c_str(), prompt.size(), tokens.data(), tokens.size(), false, true);
    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
    if (llama_decode(ctx, batch)) return "(error)";
    auto sparams = llama_sampler_chain_default_params();
    llama_sampler* smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    std::stringstream ss;
    llama_token id;
    for (int i = 0; i < n_predict; i++) {
      id = llama_sampler_sample(smpl, ctx, -1);
      if (llama_vocab_is_eog(vocab, id)) break;
      char buf[256];
      int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
      std::string piece(buf, n);
      ss << piece;
      
      // PROACTIVE TERMINATION: If we see the closing brace of the main function, stop.
      if (piece.find("}") != std::string::npos && ss.str().find("func.func") != std::string::npos) {
          break; 
      }

      batch = llama_batch_get_one(&id, 1);
      if (llama_decode(ctx, batch)) break;
    }
    llama_sampler_free(smpl);
    return ss.str();
  }

  std::string extractAndWrap(std::string result) {
    // 1. Strip Thought Blocks
    size_t r1_end = result.find("</think>");
    if (r1_end != std::string::npos) result = result.substr(r1_end + 8);
    
    // 2. Extract FIRST func.func block aggressively
    size_t pos = result.find("func.func");
    if (pos != std::string::npos) {
        size_t end_brace = result.find("}", pos);
        if (end_brace != std::string::npos) {
            std::stringstream ss;
            ss << "module {\n" << result.substr(pos, end_brace - pos + 1) << "\n}\n";
            return ss.str();
        }
    }
    
    return result; // Fallback to raw if no structure found
  }
};

std::unique_ptr<ModelRunner> createLlamaCppModelRunner() {
  return std::make_unique<LlamaCppModelRunner>();
}

} // namespace tensorlang
} // namespace mlir
