#include "tensorlang/Runtime/ModelRunner.h"
#include "llama.h"
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <chrono>
#include <fstream>
#include <mutex>

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

  int load(const std::string& modelPath) override {
    std::string brainPath = "tensorlang/runtime/models/gemma-3-12b-it-q4_k_m.gguf";
    std::string musclePath = "tensorlang/runtime/models/qwen2.5-coder-7b-instruct-q4_k_m.gguf";

    std::cout << "[NeuroJIT] Initializing Multi-Agent AI System..." << std::endl;
    
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    
    brainModel = llama_model_load_from_file(brainPath.c_str(), mparams);
    muscleModel = llama_model_load_from_file(musclePath.c_str(), mparams);

    return (brainModel && muscleModel) ? 0 : -1;
  }

  std::string query(const std::string& prompt) override {
    std::lock_guard<std::mutex> lock(queryMutex);
    
    llama_context* brainCtx = createContext(brainModel);
    llama_context* muscleCtx = createContext(muscleModel);
    if (!brainCtx || !muscleCtx) return "(error: context creation failed)";

    // --- STEP 1: THE BRAIN (Gemma 3) ---
    std::cout << "[Evolution] Brain is searching for a more fit architecture..." << std::endl;
    std::stringstream brain_ss;
    brain_ss << "<|begin_of_thought|>\n"
             << "I am performing Recursive Architecture Optimization. "
             << "The current simple proportional controller is stable, but I should explore more advanced techniques. "
             << "I will propose a specific algorithmic change to improve fitness. "
             << "<|end_of_thought|>\n"
             << "<|im_start|>user\n"
             << "CURIOSITY DRIVE: The current code is functional but sub-optimal. "
             << "Synthesize a NEW ARCHITECTURE for @get_thrust. "
             << "Consider using PID logic, state-integration, or non-linear damping.\n"
             << "Provide a REASONING PLAN and specific constants.\n"
             << "<|im_end|>\n"
             << "<|im_start|>assistant\n";
    
    std::string plan = runInference(brainCtx, brainModel, brain_ss.str(), 512);
    std::cout << "[Evolution Plan] " << plan << std::endl;

    // --- STEP 2: THE MUSCLE (Qwen 2.5) ---
    std::cout << "[Evolution] Muscle is synthesizing the new DNA..." << std::endl;
    std::stringstream muscle_ss;
    muscle_ss << "<|im_start|>system\n"
              << "You are an MLIR Genetic Synthesizer. You implement the Brain's evolutionary plan into valid MLIR code.\n"
              << "STRICT RULES: Return ONLY the func.func @get_thrust block. Use unique SSA names. Declare all constants.\n"
              << "<|im_end|>\n"
              << "<|im_start|>user\n"
              << "IMPLEMENT THIS EVOLUTIONARY PLAN:\n" << plan << "\n"
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
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 2048;
    cparams.n_threads = 64; 
    cparams.n_threads_batch = 64;
    return llama_init_from_model(m, cparams);
  }

  std::string runInference(llama_context* ctx, llama_model* model, const std::string& prompt, int n_predict) {
    const llama_vocab* vocab = llama_model_get_vocab(model);
    int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, true, true);
    std::vector<llama_token> tokens(n_prompt);
    llama_tokenize(vocab, prompt.c_str(), prompt.size(), tokens.data(), tokens.size(), true, true);

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
    size_t thought_end = result.find("<|end_of_thought|>");
    if (thought_end != std::string::npos) result = result.substr(thought_end + 18);
    size_t start_ticks = result.find("```");
    if (start_ticks != std::string::npos) {
        size_t next_line = result.find("\n", start_ticks);
        size_t end_ticks = result.find("```", next_line);
        if (next_line != std::string::npos && end_ticks != std::string::npos)
            result = result.substr(next_line + 1, end_ticks - next_line - 1);
    }
    size_t func_pos = result.find("func.func");
    if (func_pos != std::string::npos) {
        size_t last_brace = result.rfind("}");
        if (last_brace != std::string::npos) {
            std::string func_code = result.substr(func_pos, last_brace - func_pos + 1);
            return "module {\n" + func_code + "\n}";
        }
    }
    return result;
  }
};

std::unique_ptr<ModelRunner> createLlamaCppModelRunner() {
  return std::make_unique<LlamaCppModelRunner>();
}

} // namespace tensorlang
} // namespace mlir
