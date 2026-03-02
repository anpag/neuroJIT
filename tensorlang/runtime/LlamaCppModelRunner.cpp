#include "tensorlang/Runtime/ModelRunner.h"
#include "tensorlang/Runtime/JitContext.h"
#include "llama.h"
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <mutex>
#include <sys/stat.h>

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
    std::lock_guard<std::mutex> lock(inferenceMutex_);

    if (!muscleModel_) return "";

    struct stat st;
    if (stat("adapter_latest.bin", &st) == 0) {
      if (st.st_mtime > lastAdapterMtime_) {
        printf("[LLM] Hot-reloading new LoRA adapter...\n");
        currentAdapter_ = llama_adapter_lora_init(muscleModel_, "adapter_latest.bin");
        if (!currentAdapter_) {
          fprintf(stderr, "[LLM] Failed to load adapter from adapter_latest.bin\n");
        } else {
          lastAdapterMtime_ = st.st_mtime;
        }
      }
    }

    // A large context window is needed to hold the full MLIR module
    llama_context* ctx = createContext(muscleModel_, 4096);
    if (!ctx) return "";

    if (currentAdapter_) {
      float scale = 1.0f;
      llama_set_adapters_lora(ctx, &currentAdapter_, 1, &scale);
    }

    std::string formatted = formatPrompt(prompt);
    std::string raw = runInference(ctx, muscleModel_, formatted, 2048);
    llama_free(ctx);

    return extractMLIR(raw);
  }

private:
  mutable std::mutex inferenceMutex_;
  llama_model* muscleModel_ = nullptr;
  llama_adapter_lora* currentAdapter_ = nullptr;
  time_t lastAdapterMtime_ = 0;

  llama_context* createContext(llama_model* m, int n_ctx) {
    if (!m) return nullptr;
    llama_context_params p = llama_context_default_params();
    p.n_ctx           = n_ctx;
    unsigned int nCores = std::max(1u, std::thread::hardware_concurrency() / 2);
    p.n_threads       = nCores;
    p.n_threads_batch = nCores;
    fprintf(stderr, "[Llama] Using %u threads\n", nCores);
    return llama_init_from_model(m, p);
  }

  std::string formatPrompt(const std::string& user_content) {
    std::ostringstream ss;
    ss << "<|im_start|>system\n"
       << "You are an expert compiler optimization engineer.\n"
       << "Your task is to rewrite ONLY the failing 'get_thrust' function to prevent assertions or physics violations.\n\n"
       << "CRITICAL RULES:\n"
       << "1. The function MUST have the 'llvm.emit_c_interface' attribute.\n"
       << "2. ONLY return the modified get_thrust function inside a module, DO NOT return the entire original file.\n\n"
       << "EXAMPLE OF A VALID PATCH:\n"
       << "module {\n"
       << "  func.func @get_thrust(%h: f32, %v: f32) -> f32 attributes { llvm.emit_c_interface } {\n"
       << "    %gravity = arith.constant 1.62 : f32\n"
       << "    %kp      = arith.constant 0.5  : f32\n"
       << "    %neg_v   = arith.negf %v : f32\n"
       << "    %ctrl    = arith.mulf %neg_v, %kp : f32\n"
       << "    %thrust  = arith.addf %gravity, %ctrl : f32\n"
       << "    return %thrust : f32\n"
       << "  }\n"
       << "}\n\n"
       << "Return ONLY a valid MLIR module starting with `module {` and ending with `}`.\n"
       << "<|im_end|>\n"
       << "<|im_start|>user\n"
       << user_content << "\n"
       << "<|im_end|>\n"
       << "<|im_start|>assistant\n"
       << "module {\n";
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
    ss << "module {\n"; // Account for the prompt priming
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
    size_t start = raw.find("module {");
    if (start == std::string::npos) return raw; // Fallback

    int depth = 0;
    for (size_t i = start; i < raw.size(); i++) {
      if (raw[i] == '{') depth++;
      else if (raw[i] == '}') {
        depth--;
        if (depth == 0) {
          return raw.substr(start, i - start + 1);
        }
      }
    }
    
    // Fallback if the model didn't perfectly close braces but wrote some code
    size_t end = raw.find("```");
    if (end != std::string::npos) {
      return raw.substr(start, end - start);
    }

    return raw; 
  }
};

std::unique_ptr<ModelRunner> createLlamaCppModelRunner() {
  return std::make_unique<LlamaCppModelRunner>();
}

} // namespace tensorlang
} // namespace mlir