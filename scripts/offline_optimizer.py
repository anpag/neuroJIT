import argparse
import subprocess
import json
import os
import hashlib
import time
import requests

CACHE_FILE = os.path.expanduser("~/.neurojit/cache.json")

def load_config():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key and os.path.exists(".env"):
        with open(".env", "r") as f:
            for line in f:
                if line.startswith("GEMINI_API_KEY="):
                    api_key = line.strip().split("=", 1)[1]
                    break
    return api_key

def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_cache(cache):
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)

def normalize_ir(ir_text):
    import re
    # Remove line comments
    lines = [line for line in ir_text.split('\n') if not line.strip().startswith('//')]
    text = '\n'.join(lines)
    # Replace all whitespace with single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def hash_ir(normalized_ir):
    return hashlib.sha256(normalized_ir.encode('utf-8')).hexdigest()

def evaluate_ir(compiler_path, mlir_content):
    temp_file = "/tmp/test_opt.mlir"
    with open(temp_file, "w") as f:
        f.write(mlir_content)
    
    start = time.time()
    try:
        result = subprocess.run([compiler_path, temp_file], capture_output=True, text=True, timeout=15)
        end = time.time()
        
        if result.returncode == 0:
            return True, (end - start) * 1000
        else:
            print(f"  [Error] Compiler failed with code {result.returncode}")
            print(f"  [STDOUT] {result.stdout}")
            print(f"  [STDERR] {result.stderr}")
            return False, 0
    except subprocess.TimeoutExpired:
        return False, 0

def query_gemini(api_key, ir_text, model="gemini-1.5-flash"):
    if not api_key:
        print("  [Error] No API key found.")
        return None

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    prompt = f"You are an MLIR compiler engineer. Optimize the following MLIR code for performance. Return ONLY the optimized MLIR module. Do not use semicolons. Do not include markdown backticks.\n\nCODE:\n{ir_text}"
    
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }
    
    headers = {
        'Content-Type': 'application/json'
    }
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code == 200:
        res_json = response.json()
        try:
            code = res_json['candidates'][0]['content']['parts'][0]['text']
            # Basic cleanup
            if "module {" in code:
                code = code[code.find("module {"):]
                code = code[:code.rfind("}")+1]
            return code
        except:
            print(f"  [Error] Failed to parse response: {response.text}")
            return None
    else:
        print(f"  [Error] API Call failed ({response.status_code}): {response.text}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Offline MLIR Optimizer")
    parser.add_argument("files", nargs="+", help="MLIR files to optimize")
    parser.add_argument("--compiler", default="./build/tools/tensorlang-run/tensorlang-run", help="Path to compiler")
    parser.add_argument("--model", default="gemini-1.5-flash", help="Gemini model to use")
    args = parser.parse_args()

    api_key = load_config()
    cache = load_cache()

    for file in args.files:
        print(f"Processing {file}...")
        if not os.path.exists(file):
            print(f"  [Error] File not found: {file}")
            continue

        with open(file, 'r') as f:
            content = f.read()
        
        success, baseline_time = evaluate_ir(args.compiler, content)
        if not success:
            print(f"  [Error] Baseline failed to compile/run for {file}")
            continue
            
        print(f"  [Baseline] {baseline_time:.2f} ms")
        
        optimized_ir = query_gemini(api_key, content, args.model)
        if not optimized_ir:
            continue

        success, opt_time = evaluate_ir(args.compiler, optimized_ir)
        if success:
            print(f"  [Optimized] {opt_time:.2f} ms")
            if opt_time < baseline_time:
                print(f"  [Success] Improved performance!")
            
            norm_ir = normalize_ir(content)
            h = hash_ir(norm_ir)
            cache[h] = optimized_ir
            save_cache(cache)
            print(f"  [Cache] Saved strategy (Hash: {h[:8]}...)")
        else:
            print("  [Failed] Optimized IR failed to compile or run.")

if __name__ == "__main__":
    main()
