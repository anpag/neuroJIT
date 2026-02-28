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
    cleanupModel(brainModel, brainCtx);
    cleanupModel(muscleModel, muscleCtx);
  }

  int load(const std::string& modelPath) override {
    // We use hardcoded paths for this experiment to ensure we load BOTH
    std::string brainPath = "tensorlang/runtime/models/gemma-3-12b-it-q4_k_m.gguf";
    std::string musclePath = "tensorlang/runtime/models/qwen2.5-coder-7b-instruct-q4_k_m.gguf";

    std::cout << "[NeuroJIT] Initializing Multi-Agent AI System..." << std::endl;
    
    if (loadModel(brainPath, brainModel, brainCtx) != 0) return -1;
    if (loadModel(musclePath, muscleModel, muscleCtx) != 0) return -1;

    return 0;
  }

  std::string query(const std::string& prompt) override {
    std::lock_guard<std::mutex> lock(queryMutex);
    if (!brainModel || !muscleModel) return "(error: models not loaded)";

    // --- STEP 1: THE BRAIN (Gemma 3) ---
    std::cout << "[Evolution] Brain is searching for a more fit architecture..." << std::endl;
    
    std::stringstream brain_ss;
    brain_ss << "<|begin_of_thought|>\n"
             << "I am evolving a digital brain for a lunar lander. "
             << "Current Objective: Survival (Soft landing) AND Resource Efficiency (Minimum fuel). "
             << "I should consider using a PID-style control or a fuzzy logic approach. "
             << "I must specify exact constants for the Muscle to implement. "
             << "<|end_of_thought|>\n"
             << "<|im_start|>user\n"
             << "EVOLUTIONARY PROMPT:\n"
             << "Current Code: " << prompt << "\n"
             << "GOAL: Create a new version of @get_thrust(%arg0, %arg1, %arg2) that:\n"
             << "1. Lands at EXACTLY -0.5 m/s velocity.\n"
             << "2. Minimizes fuel consumption (arg2).\n"
             << "3. Explains the logic (Proportional gain, Target velocity).\n"
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
    
    return extractAndWrap(mlir_raw);
  }

private:
  std::mutex queryMutex;
  llama_model* brainModel = nullptr;
  llama_context* brainCtx = nullptr;
  
  llama_model* muscleModel = nullptr;
  llama_context* muscleCtx = nullptr;

  int loadModel(const std::string& path, llama_model*& model, llama_context*& ctx) {
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    model = llama_model_load_from_file(path.c_str(), mparams);
    if (!model) return -1;

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 2048;
    cparams.n_threads = 64; // Use full 64 cores
    cparams.n_threads_batch = 64;
    ctx = llama_init_from_model(model, cparams);
    return ctx ? 0 : -1;
  }

  void cleanupModel(llama_model* model, llama_context* ctx) {
    if (ctx) llama_free(ctx);
    if (model) llama_model_free(model);
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
    // 1. Strip Thought Blocks
    size_t thought_end = result.find("<|end_of_thought|>");
    if (thought_end != std::string::npos) result = result.substr(thought_end + 18);

    // 2. Strip Markdown
    size_t start_ticks = result.find("```");
    if (start_ticks != std::string::npos) {
        size_t next_line = result.find("\n", start_ticks);
        size_t end_ticks = result.find("```", next_line);
        if (next_line != std::string::npos && end_ticks != std::string::npos)
            result = result.substr(next_line + 1, end_ticks - next_line - 1);
    }

    // 3. Extract func and wrap
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
