#include "tensorlang/Runtime/ModelRunner.h"
#include "tensorlang/Runtime/JitContext.h"
#include "llama.h"
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <mutex>

namespace mlir {
namespace tensorlang {

class LlamaCppModelRunner : public ModelRunner {
public:
  LlamaCppModelRunner() { ggml_backend_load_all(); }

  ~LlamaCppModelRunner() {
    if (muscleModel_) llama_model_free(muscleModel_);
  }

  int load(const std::string& modelPath) override {
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0; // CPU only

    printf("[LLM] Loading model: %s\n", modelPath.c_str());
    muscleModel_ = llama_model_load_from_file(modelPath.c_str(), mparams);

    if (!muscleModel_) {
      fprintf(stderr, "[LLM] ERROR: Failed to load muscle model\n");
      return -1;
    }

    printf("[LLM] Ready.\n");
    return 0;
  }

  std::string query(const std::string& prompt) override {
    std::lock_guard<std::mutex> lock(queryMutex_);

    if (!muscleModel_) return "";

    // A large context window is needed to hold the full MLIR module
    llama_context* ctx = createContext(muscleModel_, 4096);
    if (!ctx) return "";

    std::string formatted = formatPrompt(prompt);
    std::string raw = runInference(ctx, muscleModel_, formatted, 2048);
    llama_free(ctx);

    return extractMLIR(raw);
  }

private:
  std::mutex queryMutex_;
  llama_model* muscleModel_ = nullptr;

  llama_context* createContext(llama_model* m, int n_ctx) {
    if (!m) return nullptr;
    llama_context_params p = llama_context_default_params();
    p.n_ctx           = n_ctx;
    p.n_threads       = 64;
    p.n_threads_batch = 64;
    return llama_init_from_model(m, p);
  }

  std::string formatPrompt(const std::string& user_content) {
    std::ostringstream ss;
    ss << "<|im_start|>system\n"
       << "You are an expert compiler optimization engineer. "
       << "You will be provided with an MLIR module and an execution state context.\n"
       << "Your task is to rewrite the failing MLIR function to prevent assertions or physics violations.\n\n"
       << "CRITICAL DIALECT RULES:\n"
       << "1. The 'tensorlang.assert' operation takes one operand and NO parenthesis:\n"
       << "   VALID:   func.call @tensorlang_assert_fail(%cond) : (i64) -> ()\n"
       << "   INVALID: assert(%cond)\n"
       << "2. All variables must be strictly typed (e.g. : f32 or : i64).\n"
       << "3. Use standard arith operations (arith.addf, arith.mulf, arith.cmpf olt).\n\n"
       << "Return ONLY a valid MLIR module starting with `module {` and ending with `}`.\n"
       << "No markdown. No explanation. No comments outside the module.\n"
       << "<|im_end|>\n"
       << "<|im_start|>user\n"
       << user_content << "\n"
       << "<|im_end|>\n"
       << "<|im_start|>assistant\n"
       << "```mlir\nmodule {\n";
    return ss.str();
  }

  std::string runInference(llama_context* ctx,
                            llama_model* model,
                            const std::string& prompt,
                            int n_predict) {
    const llama_vocab* vocab = llama_model_get_vocab(model);

    int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(),
                                   nullptr, 0, false, true);
    std::vector<llama_token> tokens(n_prompt);
    llama_tokenize(vocab, prompt.c_str(), prompt.size(),
                   tokens.data(), tokens.size(), false, true);

    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
    if (llama_decode(ctx, batch)) return "";

    auto sparams = llama_sampler_chain_default_params();
    llama_sampler* smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    std::ostringstream ss;
    ss << "```mlir\n"; // Account for the prompt priming
    llama_token id;

    for (int i = 0; i < n_predict; i++) {
      id = llama_sampler_sample(smpl, ctx, -1);
      if (llama_vocab_is_eog(vocab, id)) break;

      char buf[256];
      int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
      std::string piece(buf, n);
      ss << piece;

      batch = llama_batch_get_one(&id, 1);
      if (llama_decode(ctx, batch)) break;
    }

    llama_sampler_free(smpl);
    return ss.str();
  }

  std::string extractMLIR(const std::string& raw) {
    std::string clean = "module {\n" + raw;
    size_t end = clean.find("```");
    if (end != std::string::npos) {
      return clean.substr(0, end);
    }
    return clean; 
  }
};

std::unique_ptr<ModelRunner> createLlamaCppModelRunner() {
  return std::make_unique<LlamaCppModelRunner>();
}

} // namespace tensorlang
} // namespace mlir