#include "tensorlang/Runtime/ModelRunner.h"
#include <iostream>
#include <cstdlib>
#include <string>
#include <fstream>
#include <sstream>
#include <regex>
#include <curl/curl.h>

#define JSON_NOEXCEPTION 1
#include "nlohmann/json.hpp"

using json = nlohmann::json;

namespace mlir {
namespace tensorlang {

// Helper for CURL response
size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}

class GeminiModelRunner : public ModelRunner {
public:
  GeminiModelRunner() {
    curl_global_init(CURL_GLOBAL_ALL);
    
    // Try loading from .env file FIRST (to override potentially bad env vars)
    std::ifstream envFile(".env");
    if (envFile.is_open()) {
      std::string line;
      while (std::getline(envFile, line)) {
        if (line.find("GEMINI_API_KEY=") == 0) {
          apiKey_ = line.substr(15);
          // Trim whitespace/newlines
          apiKey_.erase(apiKey_.find_last_not_of(" \n\r\t") + 1);
          valid_ = true;
          break;
        }
      }
    }

    // If not found in .env, try environment variable
    if (!valid_) {
      const char* env_key = std::getenv("GEMINI_API_KEY");
      if (env_key) {
        apiKey_ = env_key;
        valid_ = true;
      }
    }
    
    if (!valid_) {
      std::cerr << "[GeminiRunner] Error: GEMINI_API_KEY not found in env or .env file." << std::endl;
    } else {
      std::cout << "[GeminiRunner] API Key loaded." << std::endl;
    }
  }

  ~GeminiModelRunner() {
    curl_global_cleanup();
  }

  int load(const std::string& modelPath) override {
    if (!valid_) return -1;
    return 0;
  }

  std::string query(const std::string& input_code) override {
    if (!valid_) return "(error: no api key)";

    std::cerr << "[GeminiRunner] Querying model for optimization..." << std::endl;

    CURL* curl = curl_easy_init();
    std::string response_string;
    
    if(curl) {
      std::string url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContent";
      
      // Construct the dynamic prompt using nlohmann/json for safe escaping
      std::string system_instruction = "You are an MLIR compiler engineer. Optimize the following MLIR code. Return ONLY the optimized MLIR code inside a module. Do not use semicolons for separators; use newlines. Do not include markdown backticks.";
      
      json payload = {
        {"contents", {{
          {"parts", {{
            {"text", system_instruction + "\n\nCODE TO OPTIMIZE:\n" + input_code}
          }}}
        }}}
      };

      std::string json_payload = payload.dump();

      struct curl_slist* headers = NULL;
      headers = curl_slist_append(headers, "Content-Type: application/json");
      std::string auth_header = "X-goog-api-key: " + apiKey_;
      headers = curl_slist_append(headers, auth_header.c_str());

      curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
      curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_payload.c_str());
      curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
      curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
      curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_string);

      CURLcode res = curl_easy_perform(curl);
      if(res != CURLE_OK) {
        std::cerr << "[GeminiRunner] curl_easy_perform() failed: " << curl_easy_strerror(res) << std::endl;
      }
      
      curl_easy_cleanup(curl);
      curl_slist_free_all(headers);
    }

    // Parse Response using nlohmann/json (no exceptions)
    json res_json = json::parse(response_string, nullptr, false);
    if (res_json.is_discarded()) {
        std::cerr << "[GeminiRunner] Failed to parse JSON response." << std::endl;
        return "(error: invalid json)";
    }

    std::string code;
    if (res_json.contains("candidates") && res_json["candidates"].is_array() && !res_json["candidates"].empty()) {
        auto& candidate = res_json["candidates"][0];
        if (candidate.contains("content") && candidate["content"].contains("parts") && 
            candidate["content"]["parts"].is_array() && !candidate["content"]["parts"].empty()) {
            auto& part = candidate["content"]["parts"][0];
            if (part.contains("text") && part["text"].is_string()) {
                code = part["text"].get<std::string>();
            }
        }
    }

    if (code.empty()) {
        std::cerr << "[GeminiRunner] Empty response or invalid structure from model." << std::endl;
        return "(error: extraction failed)";
    }
    
    // Clean up markdown and non-MLIR text
    size_t module_pos = code.find("module");
    if (module_pos != std::string::npos) {
        size_t brace_pos = code.find("{", module_pos);
        if (brace_pos != std::string::npos) {
            code = code.substr(module_pos);
            size_t last_brace = code.rfind("}");
            if (last_brace != std::string::npos) {
                code = code.substr(0, last_brace + 1);
            }
        }
    } else {
        // Fallback to markdown cleanup if "module {" not found
        size_t md_start = code.find("```mlir");
        if (md_start == std::string::npos) md_start = code.find("```");
        
        if (md_start != std::string::npos) {
            size_t start_skip = (code.substr(md_start, 7) == "```mlir") ? 7 : 3;
            code = code.substr(md_start + start_skip);
            size_t md_end = code.find("```");
            if (md_end != std::string::npos) code = code.substr(0, md_end);
        }
    }
    
    // Remove trailing whitespace
    code.erase(code.find_last_not_of(" \n\r\t") + 1);
    
    std::cerr << "[GeminiRunner] Successfully parsed " << code.length() << " bytes of MLIR." << std::endl;
    return code;
  }

private:
  std::string apiKey_;
  bool valid_ = false;
};

// Update the factory to use this runner
std::unique_ptr<ModelRunner> ModelRunner::create(const std::string& type) {
  if (type == "gemini") {
    return std::make_unique<GeminiModelRunner>();
  }
  return std::make_unique<MockModelRunner>();
}

} // namespace tensorlang
} // namespace mlir
