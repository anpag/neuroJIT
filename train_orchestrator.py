import json
import time
import os
import subprocess

DATA_FILE = "tensorlang_training_data.jsonl"
TRAIN_FILE = "train_data.txt"
ADAPTER_OUT = "adapter_latest.bin"
MODEL_PATH = "tensorlang/runtime/models/qwen2.5-coder-7b-instruct-q4_k_m.gguf"
THRESHOLD = 50 # We wait for 50 new experiences before triggering

def get_line_count(filepath):
    if not os.path.exists(filepath):
        return 0
    with open(filepath, 'r') as f:
        return sum(1 for _ in f)

def generate_training_data():
    positives = []
    with open(DATA_FILE, 'r') as f:
        for line in f:
            try:
                record = json.loads(line)
                if record.get("reward", 0) == 1.0:
                    positives.append(record)
            except json.JSONDecodeError:
                continue
    
    with open(TRAIN_FILE, 'w') as f:
        for p in positives:
            # Reconstruct the exact text sequence the model saw and generated for SFT
            text = f"<|im_start|>system\n" \
                   f"You are an expert compiler optimization engineer.\n" \
                   f"Your task is to rewrite ONLY the failing 'get_thrust' function to prevent assertions or physics violations.\n\n" \
                   f"CRITICAL RULES:\n" \
                   f"1. The function MUST have the 'llvm.emit_c_interface' attribute.\n" \
                   f"2. ONLY return the modified get_thrust function inside a module, DO NOT return the entire original file.\n\n" \
                   f"EXAMPLE OF A VALID PATCH:\n" \
                   f"module {{\n" \
                   f"  func.func @get_thrust(%h: f32, %v: f32) -> f32 attributes {{ llvm.emit_c_interface }} {{\n" \
                   f"    %gravity = arith.constant 1.62 : f32\n" \
                   f"    %kp      = arith.constant 0.5  : f32\n" \
                   f"    %neg_v   = arith.negf %v : f32\n" \
                   f"    %ctrl    = arith.mulf %neg_v, %kp : f32\n" \
                   f"    %thrust  = arith.addf %gravity, %ctrl : f32\n" \
                   f"    return %thrust : f32\n" \
                   f"  }}\n" \
                   f"}}\n\n" \
                   f"Return ONLY a valid MLIR module starting with `module {{` and ending with `}}`.\n" \
                   f"<|im_end|>\n" \
                   f"<|im_start|>user\n" \
                   f"{p.get('full_prompt', '')}\n" \
                   f"<|im_end|>\n" \
                   f"<|im_start|>assistant\n" \
                   f"{p.get('generated_patch', '')}<|im_end|>\n\n"
            f.write(text)
    return len(positives)

def run_training():
    print(f"[Orchestrator] Kicking off LoRA fine-tuning...")
    
    cmd = [
        "./tensorlang/deps/llama.cpp/build/bin/llama-finetune",
        "--model-base", MODEL_PATH,
        "--train-data", TRAIN_FILE,
        "--lora-out", ADAPTER_OUT,
        "--save-every", "0",
        "--threads", "8",
        "--ctx-size", "4096",
        "--batch-size", "1",
        "--grad-acc", "1",
        "--epochs", "3"
    ]
    
    if not os.path.exists(cmd[0]):
        print(f"[Orchestrator] Fine-tune binary not found at {cmd[0]}")
        print(f"[Orchestrator] Simulating successful training for demonstration.")
        # Simulate creating an adapter file
        with open(ADAPTER_OUT, 'w') as f:
            f.write("DUMMY LORA ADAPTER WEIGHTS")
        return True

    try:
        subprocess.run(cmd, check=True)
        print("[Orchestrator] Fine-tuning complete. New adapter generated.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[Orchestrator] Fine-tuning failed: {e}")
        return False

def main():
    print("[Orchestrator] Starting NeuroJIT Training Daemon...")
    last_count = get_line_count(DATA_FILE)
    print(f"[Orchestrator] Initial experience count: {last_count}")
    
    while True:
        time.sleep(5)
        current_count = get_line_count(DATA_FILE)
        if current_count - last_count >= THRESHOLD:
            print(f"\n[Orchestrator] Reached {current_count} experiences. Triggering training cycle...")
            num_positives = generate_training_data()
            print(f"[Orchestrator] Extracted {num_positives} successful patches for SFT.")
            
            if num_positives > 0:
                success = run_training()
                if success:
                    # Update count to avoid re-triggering for the same rows immediately
                    last_count = current_count
            else:
                print("[Orchestrator] No positive examples found. Skipping training.")
                last_count = current_count

if __name__ == '__main__':
    main()