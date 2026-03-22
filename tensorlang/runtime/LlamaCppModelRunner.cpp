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

    std::string history = getHistory();
    std::string constraint = getSeedConstraint();
    std::string formatted = formatPrompt(prompt, history, constraint);
    
    std::string raw = runInference(ctx, muscleModel_, formatted, 2048);
    llama_free(ctx);

    return extractGetThrust(raw);
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
    p.n_ctx           = n_ctx;
    unsigned int nCores = std::max(1u, std::thread::hardware_concurrency() / 2);
    p.n_threads       = nCores;
    p.n_threads_batch = nCores;
    fprintf(stderr, "[Llama] Using %u threads\n", nCores);
    return llama_init_from_model(m, p);
  }

  std::string getHistory() {
    std::ifstream file("tensorlang_training_data.jsonl");
    if (!file.is_open()) return "No previous attempts.\n";

    std::vector<std::string> lines;
    std::string line;
    while (std::getline(file, line)) {
      if (!line.empty()) lines.push_back(line);
    }

    std::ostringstream ss;
    int count = 0;
    for (int i = lines.size() - 1; i >= 0 && count < 3; --i, ++count) {
      // Very simple JSON extraction for speed (avoiding extra deps)
      size_t patchStart = lines[i].find("\"generated_patch\":\"");
      if (patchStart == std::string::npos) continue;
      patchStart += 19;
      size_t patchEnd = lines[i].find("\",\"compiled\"", patchStart);
      if (patchEnd == std::string::npos) continue;
      
      std::string patch = lines[i].substr(patchStart, patchEnd - patchStart);
      // Unescape newlines
      size_t pos = 0;
      while ((pos = patch.find("\\n", pos)) != std::string::npos) {
        patch.replace(pos, 2, "\n");
        pos += 1;
      }

      size_t rewardStart = lines[i].find("\"reward\":");
      std::string reward = "unknown";
      if (rewardStart != std::string::npos) {
        rewardStart += 9;
        size_t rewardEnd = lines[i].find(",", rewardStart);
        if (rewardEnd == std::string::npos) rewardEnd = lines[i].find("}", rewardStart);
        if (rewardEnd != std::string::npos) reward = lines[i].substr(rewardStart, rewardEnd - rewardStart);
      }

      ss << "Attempt " << (count + 1) << " (Reward: " << reward << "):\n" << patch << "\n\n";
    }
    return ss.str();
  }

  std::string getSeedConstraint() {
    static const std::vector<std::string> constraints = {
      "The base thrust must be between 0.5 and 1.0",
      "The velocity coefficient must be negative",
      "Use at least 3 arithmetic operations",
      "The thrust must decrease as height increases",
      "Use a coefficient greater than 1.0 for velocity scaling",
    };
    std::uniform_int_distribution<int> dist(0, constraints.size() - 1);
    return constraints[dist(rng_)];
  }

  std::string formatPrompt(const std::string& ir_before, const std::string& history, const std::string& constraint) {
    std::ostringstream ss;
    ss << "<|im_start|>system\n"
       << "You are an MLIR compiler engineer. You rewrite broken MLIR functions.\n"
       << "Write a NEW function @get_thrust(%h: f32, %v: f32) -> f32 that uses %h (height) and %v (velocity) to compute thrust.\n\n"
       << "Rules:\n"
       << "- Use ONLY these ops: arith.constant, arith.addf, arith.subf, arith.mulf, arith.negf, arith.select, arith.cmpf\n"
       << "- Return a single f32 value inside a module block.\n"
       << "- No loops, no branches.\n"
       << "- Use DIFFERENT constants than previous attempts.\n"
       << "- Additional constraint this round: " << constraint << "\n\n"
       << "Previous attempts and their scores:\n"
       << history << "\n"
       << "Return ONLY the module block starting with 'module {'. No explanation. No markdown.\n"
       << "<|im_end|>\n"
       << "<|im_start|>user\n"
       << "The following get_thrust function is broken. Fix it.\n\n"
       << "BROKEN FUNCTION:\n"
       << ir_before << "\n\n"
       << "Output the fixed module now.\n"
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
    
    // Apply new sampling logic for diversity
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.8f));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.95f, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(smpl, llama_sampler_init_penalties(
        /*penalty_last_n*/ 64,
        /*penalty_repeat*/ 1.15f,
        /*penalty_freq*/ 0.0f,
        /*penalty_present*/ 0.0f
    ));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(0));

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

  std::string extractGetThrust(const std::string& raw) {
    // Find get_thrust function start
    size_t fnStart = raw.find("func.func @get_thrust");
    if (fnStart == std::string::npos) return "";
    
    // Find the module start before it
    size_t modStart = raw.rfind("module {", fnStart);
    if (modStart == std::string::npos) modStart = fnStart;
    
    // Walk forward to find matching closing brace of get_thrust
    size_t pos = fnStart;
    int depth = 0;
    bool started = false;
    while (pos < raw.size()) {
        if (raw[pos] == '{') { depth++; started = true; }
        if (raw[pos] == '}') { depth--; }
        if (started && depth == 0) {
            // Build isolated module with only get_thrust
            return "module {\n" + 
                   raw.substr(fnStart, pos - fnStart + 1) + 
                   "\n}";
        }
        pos++;
    }
    return "";
  }
};

std::unique_ptr<ModelRunner> createLlamaCppModelRunner() {
  return std::make_unique<LlamaCppModelRunner>();
}

} // namespace tensorlang
} // namespace mlir