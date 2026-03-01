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
    // FEB 2026: The "Elite Suite"
    // Brain: DeepSeek-R1-32B (Chain of Thought logic)
    // Muscle: Qwen3-Coder-Next (Advanced MLIR Synthesis)
    
    std::string brainPath = "tensorlang/runtime/models/deepseek-r1-32b-q4_k_m.gguf";
    std::string musclePath = "tensorlang/runtime/models/qwen3-coder-next-ud-q4_k_xl.gguf";
    
    // Auto-detect and fallback
    if (!fileExists(brainPath)) {
        std::cout << "[NeuroJIT] R1-32B not found, falling back to Gemma 3" << std::endl;
        brainPath = "tensorlang/runtime/models/gemma-3-12b-it-q4_k_m.gguf";
    }
    
    if (!fileExists(musclePath)) {
        std::cout << "[NeuroJIT] Qwen3 not found, falling back to Qwen 2.5 32B" << std::endl;
        musclePath = "tensorlang/runtime/models/qwen2.5-coder-32b-instruct-q4_k_m.gguf";
    }
    
    if (!fileExists(musclePath)) {
        std::cout << "[NeuroJIT] Qwen 32B not found, falling back to Qwen 2.5 7B" << std::endl;
        musclePath = "tensorlang/runtime/models/qwen2.5-coder-7b-instruct-q4_k_m.gguf";
    }

    std::cout << "[NeuroJIT] Elite Configuration Active:" << std::endl;
    std::cout << "  - Brain:  " << brainPath << std::endl;
    std::cout << "  - Muscle: " << musclePath << std::endl;
    
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    
    brainModel = llama_model_load_from_file(brainPath.c_str(), mparams);
    muscleModel = llama_model_load_from_file(musclePath.c_str(), mparams);

    return (brainModel && muscleModel) ? 0 : -1;
  }

  std::string query(const std::string& prompt) override {
    std::lock_guard<std::mutex> lock(queryMutex);
    
    // Use the "Fresh Context" pattern to avoid memory slots failure (MoE requirement)
    llama_context* brainCtx = createContext(brainModel);
    llama_context* muscleCtx = createContext(muscleModel);
    if (!brainCtx || !muscleCtx) return "(error: context creation failed)";

    // --- STEP 1: THE BRAIN (DeepSeek-R1 / Gemma 3) ---
    std::cout << "[Evolution] Brain reasoning cycle starting..." << std::endl;
    std::stringstream brain_ss;
    
    // R1 Style Prompt - Constrained for speed
    brain_ss << "<｜begin▁of▁sentence｜><｜User｜><think>\n"
             << "I need a concise mathematical derivation for lunar landing control. "
             << "Target: -0.5 m/s. Minimize fuel. Be extremely brief.\n"
             << "</think>\n"
             << "Generate a precise implementation plan for @get_thrust. "
             << "Focus on GAIN and TARGET constants. Max 50 words logic explanation. "
             << "Telemetry: " << prompt << "\n"
             << "<｜Assistant｜>";
    
    // Reduce n_predict to 512 for the Brain to force a quick conclusion
    std::string plan = runInference(brainCtx, brainModel, brain_ss.str(), 512);
    std::cout << "[Evolution Plan] Plan synthesized." << std::endl;

    // --- STEP 2: THE MUSCLE (Qwen3-Coder-Next) ---
    std::cout << "[Evolution] Muscle synthesis cycle starting..." << std::endl;
    std::stringstream muscle_ss;
    muscle_ss << "<|im_start|>system\n"
              << "You are an MLIR specialist. You implement logical plans into @get_thrust.\n"
              << "STRICT: Return ONLY the func.func block. No markdown. No thought tokens.\n"
              << "<|im_end|>\n"
              << "<|im_start|>user\n"
              << "Plan: " << plan << "\n"
              << "Implement this in MLIR for the NeuroJIT compiler.\n"
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
    // Strip R1 Thought from implementation if it leaked
    size_t r1_end = result.find("</think>");
    if (r1_end != std::string::npos) result = result.substr(r1_end + 8);

    // Markdown stripping
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
