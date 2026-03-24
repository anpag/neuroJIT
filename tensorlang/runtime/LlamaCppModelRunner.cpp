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
#include <random>

namespace mlir {
namespace tensorlang {

class LlamaCppModelRunner : public ModelRunner {
public:
  LlamaCppModelRunner() { 
    ggml_backend_load_all();
    std::random_device rd;
    rng_.seed(rd());
  }

  ~LlamaCppModelRunner() {
    if (muscleModel_) llama_model_free(muscleModel_);
  }

  int load(const std::string& modelPath) override {
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0; // CPU only
    mparams.use_mmap = false; // Force the model entirely into RAM

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

    llama_context* ctx = createContext(muscleModel_, 8192); // Increase context for large AST
    if (!ctx) return "";

    if (currentAdapter_) {
      float scale = 1.0f;
      llama_set_adapters_lora(ctx, &currentAdapter_, 1, &scale);
    }

    std::string history = getHistory();
    std::string formatted = formatPrompt(prompt, history);
    
    std::string rawJSON = runInference(ctx, muscleModel_, formatted, 1024);
    llama_free(ctx);

    return rawJSON;
  }

private:
  mutable std::mutex inferenceMutex_;
  llama_model* muscleModel_ = nullptr;
  llama_adapter_lora* currentAdapter_ = nullptr;
  time_t lastAdapterMtime_ = 0;
  std::mt19937 rng_;

  llama_context* createContext(llama_model* m, int n_ctx) {
    if (!m) return nullptr;
    llama_context_params p = llama_context_default_params();
    p.n_ctx           = 4096; // Increase context
    p.n_batch         = 4096; // Increase batch size to match context
    p.n_threads       = 16;
    p.n_threads_batch = 16;
    fprintf(stderr, "[Llama] Using 16 threads, ctx/batch=4096\n");
    return llama_init_from_model(m, p);
  }

  std::string getHistory() {
    // Phase 3: history management will be handled by MCTS later.
    return "[]\n";
  }

  std::string formatPrompt(const std::string& ir_before, const std::string& history) {
    std::ostringstream ss;
    ss << "<|im_start|>system\n"
       << "You are an AI compiler optimization agent. Your task is to analyze the provided MLIR AST and propose a discrete mutation to improve its fitness score. You must output a strictly formatted JSON object.\n\n"
       << "Output Schema:\n"
       << "{\n"
       << "  \"target_function\": \"matmul\",\n"
       << "  \"mutations\": [\n"
       << "    { \"type\": \"unroll\", \"loop_depth\": 1, \"factor\": 4 },\n"
       << "    { \"type\": \"tile\", \"sizes\": [32, 32, 32] }\n"
       << "  ]\n"
       << "}\n\n"
       << "Previous Actions (History):\n"
       << history << "\n"
       << "Return ONLY the JSON. No explanation. No markdown.\n"
       << "<|im_end|>\n"
       << "<|im_start|>user\n"
       << "CURRENT MLIR AST:\n"
       << ir_before << "\n\n"
       << "Output your JSON mutation now.\n"
       << "<|im_end|>\n"
       << "<|im_start|>assistant\n";
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
    
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.8f));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.95f, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(smpl, llama_sampler_init_penalties(
        64, 1.15f, 0.0f, 0.0f
    ));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(0));

    std::string grammarStr;
    std::ifstream grammarFile("tensorlang/runtime/models/mutation.gbnf");
    if (grammarFile.is_open()) {
      std::ostringstream ss_gram;
      ss_gram << grammarFile.rdbuf();
      grammarStr = ss_gram.str();
    } else {
      fprintf(stderr, "[Llama] WARNING: Could not open grammar file.\n");
    }

    // Disable grammar sampler for now as it causes empty stack crashes in llama.cpp 
    // with the DeepSeek 32B model tokenizer.
    // The regex in ASTMutator is robust enough to extract the mutation anyway.
    /*
    llama_sampler* grammar_sampler = nullptr;
    if (!grammarStr.empty()) {
      grammar_sampler = llama_sampler_init_grammar(vocab, grammarStr.c_str(), "root");
    }
    if (grammar_sampler) {
        llama_sampler_chain_add(smpl, grammar_sampler);
    } else {
        fprintf(stderr, "[Llama] WARNING: Failed to initialize grammar sampler.\n");
    }
    */

    std::ostringstream ss;
    llama_token id;

    for (int i = 0; i < n_predict; i++) {
      id = llama_sampler_sample(smpl, ctx, -1);
      if (llama_vocab_is_eog(vocab, id)) break;

      char buf[256];
      int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
      std::string piece(buf, n);
      ss << piece;

      llama_sampler_accept(smpl, id);

      batch = llama_batch_get_one(&id, 1);
      if (llama_decode(ctx, batch)) break;
    }

    llama_sampler_free(smpl);
    return ss.str();
  }
};

std::unique_ptr<ModelRunner> createLlamaCppModelRunner() {
  return std::make_unique<LlamaCppModelRunner>();
}

} // namespace tensorlang
} // namespace mlir
