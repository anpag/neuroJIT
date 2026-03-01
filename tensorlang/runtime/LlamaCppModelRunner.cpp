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
    llama_context* brainCtx = createContext(brainModel);
    llama_context* muscleCtx = createContext(muscleModel);
    if (!brainCtx || !muscleCtx) return "(error: context creation failed)";

    // --- STEP 1: THE BRAIN (Swarm Architecture & Cross-Breeding) ---
    std::cout << "[Evolution] Brain (R1) analyzing swarm results for cross-breeding..." << std::endl;
    std::stringstream brain_ss;
    brain_ss << "<｜begin▁of▁sentence｜><｜User｜><think>\n"
             << "I am evolving a swarm of 100 landers. Environment: Gravitational Turbulence (-0.5 to 0.5). "
             << "Objective: Cross-breed the most efficient PD/PID control parameters. "
             << "I will synthesize a 'Super Lobe' that is noise-resilient.\n"
             << "</think>\n"
             << "SWARM EVOLUTION PLAN:\n"
             << "1. Synthesize a noise-robust control architecture (Lobe).\n"
             << "2. Use memory tensor to filter turbulence.\n"
             << "Current IR: " << prompt << "\n"
             << "<｜Assistant｜>";
    
    std::string plan = runInference(brainCtx, brainModel, brain_ss.str(), 512);
    
    // --- STEP 2: THE MUSCLE ---
    std::cout << "[Evolution] Muscle synthesizing noise-resilient Super Lobe..." << std::endl;
    std::stringstream muscle_ss;
    muscle_ss << "<|im_start|>system\n"
              << "You are an MLIR Swarm Architect. Implement the noise-resilient Super Lobe.\n"
              << "STRICT: Return ONLY the func.func blocks. Use multiple lobes if needed.\n"
              << "<|im_end|>\n"
              << "<|im_start|>user\n"
              << "IMPLEMENT SWARM PLAN:\n" << plan << "\n"
              << "<|im_end|>\n"
              << "<|im_start|>assistant\n";

    std::string mlir_raw = runInference(muscleCtx, muscleModel, muscle_ss.str(), 1024);
    
    llama_free(brainCtx);
    llama_free(muscleCtx);

    return extractAndWrap(mlir_raw);
  }

private:
  std::mutex queryMutex;
  llama_model* brainModel = nullptr;
  llama_model* muscleModel = nullptr;

  llama_context* createContext(llama_model* m) {
    if (!m) return nullptr;
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 4096;
    cparams.n_threads = 64;
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
      ss << std::string(buf, n);
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
    
    // 2. Strip Markdown
    size_t start_ticks = result.find("```");
    if (start_ticks != std::string::npos) {
        size_t next_line = result.find("\n", start_ticks);
        size_t end_ticks = result.find("```", next_line);
        if (next_line != std::string::npos && end_ticks != std::string::npos)
            result = result.substr(next_line + 1, end_ticks - next_line - 1);
    }
    
    // 3. Extract ALL func.func blocks and wrap in module
    std::stringstream ss;
    ss << "module {\n";
    size_t pos = result.find("func.func");
    while (pos != std::string::npos) {
        size_t end_brace = result.find("}", pos);
        if (end_brace != std::string::npos) {
            // Find the nested scope depth if any (simple brace count)
            ss << result.substr(pos, end_brace - pos + 1) << "\n";
            pos = result.find("func.func", end_brace);
        } else {
            break;
        }
    }
    ss << "}\n";
    
    std::string wrapped = ss.str();
    // Only return if we actually found functions
    return (wrapped.size() > 20) ? wrapped : result;
  }
};

std::unique_ptr<ModelRunner> createLlamaCppModelRunner() {
  return std::make_unique<LlamaCppModelRunner>();
}

} // namespace tensorlang
} // namespace mlir
