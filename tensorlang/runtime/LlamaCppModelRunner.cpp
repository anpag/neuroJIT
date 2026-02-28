#include "tensorlang/Runtime/ModelRunner.h"
#include "llama.h"
#include <iostream>
#include <vector>
#include <string>
#include <sstream>

namespace mlir {
namespace tensorlang {

class LlamaCppModelRunner : public ModelRunner {
public:
  LlamaCppModelRunner() {
    // Initializing backends
    ggml_backend_load_all();
  }

  ~LlamaCppModelRunner() {
    if (ctx) {
      llama_free(ctx);
    }
    if (model) {
      llama_model_free(model);
    }
    if (smpl) {
      llama_sampler_free(smpl);
    }
  }

  int load(const std::string& modelPath) override {
    std::cout << "[LlamaCpp] Loading model: " << modelPath << std::endl;
    
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0; // CPU only as requested

    model = llama_model_load_from_file(modelPath.c_str(), model_params);
    if (!model) {
      std::cerr << "[LlamaCpp] Error: Failed to load model from " << modelPath << std::endl;
      return -1;
    }

    vocab = llama_model_get_vocab(model);
    return 0;
  }

  std::string query(const std::string& prompt) override {
    if (!model) return "(error: model not loaded)";

    std::cout << "[LlamaCpp] Querying model..." << std::endl;

    // Construct full prompt with ChatML formatting for Qwen2.5-Coder
    std::stringstream prompt_ss;
    prompt_ss << "<|im_start|>system\n"
              << "You are an MLIR compiler engineer. The following MLIR code is crashing because @get_thrust is returning 0.0. "
              << "You MUST change the @get_thrust function to calculate a thrust value based on %h (altitude) and %v (velocity) to ensure a soft landing. "
              << "Example of fixed logic: %thrust = -0.5 * %v. "
              << "Return ONLY the complete MLIR module. No prose, no backticks, no semicolons.\n"
              << "<|im_end|>\n"
              << "<|im_start|>user\n"
              << "Optimize the following MLIR code for soft landing:\n"
              << "module {\n"
              << "  func.func @get_thrust(%h: f32, %v: f32) -> f32 {\n"
              << "    %c0 = arith.constant 0.0 : f32\n"
              << "    return %c0 : f32\n"
              << "  }\n"
              << "}\n"
              << "<|im_end|>\n"
              << "<|im_start|>assistant\n"
              << "module {\n"
              << "  func.func @get_thrust(%h: f32, %v: f32) -> f32 {\n"
              << "    %target_v = arith.constant -2.0 : f32\n"
              << "    %diff = arith.subf %target_v, %v : f32\n"
              << "    %kp = arith.constant 1.5 : f32\n"
              << "    %thrust = arith.mulf %diff, %kp : f32\n"
              << "    return %thrust : f32\n"
              << "  }\n"
              << "}\n"
              << "<|im_end|>\n"
              << "<|im_start|>user\n"
              << "Optimize the following MLIR code:\n"
              << prompt
              << "<|im_end|>\n"
              << "<|im_start|>assistant\n";
    
    std::string full_prompt = prompt_ss.str();

    // Tokenize
    int n_prompt = -llama_tokenize(vocab, full_prompt.c_str(), full_prompt.size(), NULL, 0, true, true);
    std::vector<llama_token> prompt_tokens(n_prompt);
    if (llama_tokenize(vocab, full_prompt.c_str(), full_prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
      return "(error: tokenization failed)";
    }

    // Context params
    int n_predict = 1024; // Max tokens to generate
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_prompt + n_predict;
    ctx_params.n_batch = n_prompt;
    ctx_params.n_threads = 64;       // Use all cores
    ctx_params.n_threads_batch = 64; // Use all cores
    
    // Re-init context for each query to keep it simple and stateless for now
    if (ctx) llama_free(ctx);
    ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) return "(error: context initialization failed)";

    // Sampler
    if (smpl) llama_sampler_free(smpl);
    auto sparams = llama_sampler_chain_default_params();
    smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    // Decode prompt
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
    if (llama_decode(ctx, batch)) {
      return "(error: prompt decoding failed)";
    }

    std::stringstream ss;
    llama_token new_token_id;
    int n_decode = 0;

    // Main inference loop
    while (n_decode < n_predict) {
      new_token_id = llama_sampler_sample(smpl, ctx, -1);
      
      if (llama_vocab_is_eog(vocab, new_token_id)) {
        break;
      }

      char buf[256];
      int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
      if (n < 0) break;
      ss << std::string(buf, n);

      batch = llama_batch_get_one(&new_token_id, 1);
      if (llama_decode(ctx, batch)) {
        break;
      }
      n_decode++;
    }

    std::string result = ss.str();
    std::cerr << "[LlamaCpp] Raw model output:\n" << result << "\n[LlamaCpp] End of raw output." << std::endl;
    
    // Simple cleanup of the result (same as GeminiModelRunner)
    size_t module_pos = result.find("module");
    if (module_pos != std::string::npos) {
        size_t brace_pos = result.find("{", module_pos);
        if (brace_pos != std::string::npos) {
            std::string code = result.substr(module_pos);
            size_t last_brace = code.rfind("}");
            if (last_brace != std::string::npos) {
                code = code.substr(0, last_brace + 1);
                std::cerr << "[LlamaCpp] Extracted MLIR:\n" << code << std::endl;
                return code;
            }
        }
    }

    return result;
  }

private:
  llama_model* model = nullptr;
  const llama_vocab* vocab = nullptr;
  llama_context* ctx = nullptr;
  llama_sampler* smpl = nullptr;
};

std::unique_ptr<ModelRunner> createLlamaCppModelRunner() {
  return std::make_unique<LlamaCppModelRunner>();
}

} // namespace tensorlang
} // namespace mlir
