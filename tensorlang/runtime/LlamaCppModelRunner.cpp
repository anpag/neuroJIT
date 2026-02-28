#include "tensorlang/Runtime/ModelRunner.h"
#include "llama.h"
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <chrono>
#include <fstream>

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
    currentModelPath = modelPath;
    
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

    auto start_time = std::chrono::high_resolution_clock::now();
    std::cout << "[LlamaCpp] Querying model..." << std::endl;

    // Determine prompt format based on model name
    std::stringstream prompt_ss;
    if (currentModelPath.find("gemma-3") != std::string::npos) {
        prompt_ss << "<|begin_of_thought|>\n"
                  << "I need to rewrite the @get_thrust function in MLIR to prevent the lander from crashing. "
                  << "The current code returns 0.0, but I need to use the velocity %arg1 to calculate thrust. "
                  << "Thrust logic: thrust = ( -1.0 - velocity ) * 4.0. "
                  << "<|end_of_thought|>\n"
                  << "<|im_start|>system\n"
                  << "You are an MLIR compiler engineer. Return ONLY the code for @get_thrust inside the module.\n"
                  << "<|im_end|>\n"
                  << "<|im_start|>user\n"
                  << "Rewrite @get_thrust for safety:\n"
                  << "  func.func @get_thrust(%arg0: f32, %arg1: f32) -> f32 {\n"
                  << "    %c0 = arith.constant 0.0 : f32\n"
                  << "    return %c0 : f32\n"
                  << "  }\n"
                  << "<|im_end|>\n"
                  << "<|im_start|>assistant\n";
    } else {
        // Default ChatML for Qwen/Llama/DeepSeek
        prompt_ss << "<|im_start|>system\n"
                  << "You are an MLIR compiler engineer. Return ONLY the code for @get_thrust.\n"
                  << "STRICT RULES:\n"
                  << "1. FULL FUNCTION: Return the code inside `func.func @get_thrust(%arg0: f32, %arg1: f32) -> f32 { ... }`.\n"
                  << "2. NO INLINE LITERALS: You CANNOT use numbers directly in operations. You MUST declare a constant first.\n"
                  << "3. SYNTAX: %res = arith.subf %v1, %v2 : f32\n"
                  << "4. NO MARKDOWN: Do not use ``` or backticks.\n"
                  << "5. LOGIC: Use thrust = ( -1.0 - velocity ) * 4.0.\n"
                  << "<|im_end|>\n"
                  << "<|im_start|>user\n"
                  << "Rewrite this function to land safely:\n"
                  << "  func.func @get_thrust(%arg0: f32, %arg1: f32) -> f32 {\n"
                  << "    %c0 = arith.constant 0.0 : f32\n"
                  << "    return %c0 : f32\n"
                  << "  }\n"
                  << "<|im_end|>\n"
                  << "<|im_start|>assistant\n";
    }
    
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

    // Load and add grammar
    /* Grammar disabled due to llama.cpp parsing crashes with MLIR whitespace
    std::string grammar_path = "tensorlang/runtime/mlir_thrust.gbnf";
    std::ifstream grammar_file(grammar_path);
    if (grammar_file.is_open()) {
        std::stringstream buffer;
        buffer << grammar_file.rdbuf();
        std::string grammar_str = buffer.str();
        auto* grammar_sampler = llama_sampler_init_grammar(vocab, grammar_str.c_str(), "root");
        if (grammar_sampler) {
            llama_sampler_chain_add(smpl, grammar_sampler);
            std::cout << "[LlamaCpp] Applied grammar constraints from " << grammar_path << std::endl;
        } else {
            std::cerr << "[LlamaCpp] Error: Failed to parse grammar. Continuing without constraints." << std::endl;
        }
    } else {
        std::cerr << "[LlamaCpp] Warning: Could not load grammar file: " << grammar_path << std::endl;
    }
    */

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

    auto end_time = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(end_time - start_time).count();
    
    std::cout << "[LlamaCpp] Inference finished in " << duration << "s (" 
              << (n_decode / (duration > 0 ? duration : 1.0)) << " tokens/sec)" << std::endl;

    std::string result = ss.str();
    std::cout << "[LlamaCpp] Raw model output:\n" << result << "\n[LlamaCpp] End of raw output." << std::endl;

    // 1. Strip Thought Blocks (Gemma 3)
    size_t thought_start = result.find("<|begin_of_thought|>");
    if (thought_start != std::string::npos) {
        size_t thought_end = result.find("<|end_of_thought|>", thought_start);
        if (thought_end != std::string::npos) {
            result = result.substr(thought_end + 18); // Skip thought content
        }
    }

    // 2. Strip Markdown Code Blocks
    size_t start_ticks = result.find("```");
    if (start_ticks != std::string::npos) {
        size_t next_line = result.find("\n", start_ticks);
        if (next_line != std::string::npos) {
            size_t end_ticks = result.find("```", next_line);
            if (end_ticks != std::string::npos) {
                result = result.substr(next_line + 1, end_ticks - next_line - 1);
            }
        }
    }

    // 2. Try to extract module
    size_t module_pos = result.find("module");
    if (module_pos != std::string::npos) {
        size_t brace_pos = result.find("{", module_pos);
        if (brace_pos != std::string::npos) {
            std::string code = result.substr(module_pos);
            size_t last_brace = code.rfind("}");
            if (last_brace != std::string::npos) {
                code = code.substr(0, last_brace + 1);
                std::cout << "[LlamaCpp] Extracted Module:\n" << code << std::endl;
                return code;
            }
        }
    }

    // 3. Try to extract function
    size_t func_pos = result.find("func.func");
    if (func_pos != std::string::npos) {
        size_t last_brace = result.rfind("}");
        if (last_brace != std::string::npos) {
            std::string func_code = result.substr(func_pos, last_brace - func_pos + 1);
            std::string wrapped = "module {\n" + func_code + "\n}";
            std::cout << "[LlamaCpp] Extracted Function and Wrapped in module:\n" << wrapped << std::endl;
            return wrapped;
        }
    }

    // 4. Fallback: If it's just raw body, wrap it in a function and module
    std::string fallback = "module {\n  func.func @get_thrust(%arg0: f32, %arg1: f32) -> f32 {\n" + result + "\n  }\n}";
    std::cout << "[LlamaCpp] Fallback Wrapping:\n" << fallback << std::endl;
    return fallback;

    return result;
  }

private:
  std::string currentModelPath;
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
