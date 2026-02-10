#include "tensorlang/Runtime/ModelRunner.h"
#include <iostream>
#include <cstdlib>
#include <string>
#include <fstream>
#include <sstream>
#include <regex>
#include <curl/curl.h>

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

    std::cerr << "[GeminiRunner] Constructing generic optimization prompt..." << std::endl;

    CURL* curl = curl_easy_init();
    std::string response_string;
    
    if(curl) {
      // Use 'gemini-2.5-pro' via v1beta endpoint (confirmed working).
      std::string url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent?key=" + apiKey_;
      
      std::cerr << "[GeminiRunner] Using URL: " << url << std::endl;
      std::cerr << "[GeminiRunner] API Key Length: " << apiKey_.length() << std::endl;
      
      // Construct the dynamic prompt
      std::string system_instruction = "You are an MLIR compiler engineer. Optimize the following MLIR code. Return ONLY the optimized MLIR code inside a module.";
      
      // Escape for JSON
      std::string escaped_code = "";
      for(char c : input_code) {
        if (c == '"') escaped_code += "\\\"";
        else if (c == '\n') escaped_code += "\\n";
        else if (c == '\\') escaped_code += "\\\\";
        else escaped_code += c;
      }

      std::string json_payload = "{\"contents\":[{\"parts\":[{\"text\":\"" + system_instruction + "\\n\\nCODE TO OPTIMIZE:\\n" + escaped_code + "\"}]}]}";

      struct curl_slist* headers = NULL;
      headers = curl_slist_append(headers, "Content-Type: application/json");

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

    std::cerr << "[GeminiRunner] Raw Response: " << response_string << std::endl;

    std::string code;
    size_t text_pos = response_string.find("\"text\": \"");
    if (text_pos != std::string::npos) {
        size_t start = text_pos + 9;
        size_t end = response_string.find("\"", start);
        std::string raw_code = response_string.substr(start, end - start);
        
        // Unescape
        for (size_t i = 0; i < raw_code.length(); ++i) {
            if (raw_code[i] == '\\') {
                if (i + 1 < raw_code.length()) {
                    if (raw_code[i+1] == 'n') { code += '\n'; i++; }
                    else if (raw_code[i+1] == '\"') { code += '\"'; i++; }
                    else if (raw_code[i+1] == '\\') { code += '\\'; i++; }
                    else if (raw_code[i+1] == 'u' && i + 5 < raw_code.length()) {
                        // Handle simple unicode escapes like \u003e (>) and \u003c (<)
                        std::string hex = raw_code.substr(i+2, 4);
                        int char_code = std::stoi(hex, nullptr, 16);
                        if (char_code < 128) {
                            code += (char)char_code;
                        } else {
                            // Simple hack for non-ascii: append as is or ignore.
                            // For MLIR code, we expect mostly ASCII.
                            // To be safe, we could use a library, but let's just drop high unicode or keep raw.
                            // Actually, let's just skip it to avoid breaking compilation if it's comment.
                        }
                        i += 5;
                    }
                    else { code += raw_code[i]; }
                }
            } else {
                code += raw_code[i];
            }
        }
    } else {
        std::cerr << "[GeminiRunner] Failed to parse JSON. Check raw response above." << std::endl;
        return "(error: api failed)";
    }
    
    // Clean up markdown
    size_t md_start = code.find("```mlir");
    if (md_start != std::string::npos) {
        code = code.substr(md_start + 7);
        size_t md_end = code.find("```");
        if (md_end != std::string::npos) code = code.substr(0, md_end);
    }
    // ... generic markdown cleanup ...
    
    // Sanitize semicolons (Gemini sometimes uses them as separators)
    for (size_t i = 0; i < code.length(); ++i) {
        if (code[i] == ';') {
            code[i] = '\n';
        }
    }
    
    std::cerr << "[GeminiRunner] Parsed Code Length: " << code.length() << std::endl;
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
