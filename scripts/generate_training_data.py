import subprocess
import os
import random
import json
import re

# Template for the pilot function
PILOT_TEMPLATE = """
  func.func @get_thrust(%arg0: f32, %arg1: f32) -> f32 {{
    %target_v = arith.constant {target_v:.2f} : f32
    %diff = arith.subf %target_v, %arg1 : f32
    %kp = arith.constant {kp:.2f} : f32
    %thrust = arith.mulf %diff, %kp : f32
    return %thrust : f32
  }}
"""

def generate_full_mlir(pilot_code):
    return f"""module {{
  func.func private @tensorlang_assert_fail(i64)
  func.func private @tensorlang_print_status(f32, f32)

  {pilot_code}

  func.func @main() -> i32 {{
    %dt = arith.constant 0.1 : f32
    %g = arith.constant -1.62 : f32
    %h0 = arith.constant 100.0 : f32
    %v0 = arith.constant -10.0 : f32
    
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c200 = arith.constant 200 : index
    %c0_i32 = arith.constant 0 : i32
    
    %h_fail = arith.constant 0.0 : f32
    %v_fail = arith.constant -5.0 : f32
    %true = arith.constant true
    
    %res:2 = scf.for %i = %c0 to %c200 step %c1 iter_args(%h = %h0, %v = %v0) -> (f32, f32) {{
        %thrust = func.call @get_thrust(%h, %v) : (f32, f32) -> f32
        
        %accel = arith.addf %g, %thrust : f32
        %dv = arith.mulf %accel, %dt : f32
        %v_new = arith.addf %v, %dv : f32
        
        %dh = arith.mulf %v_new, %dt : f32
        %h_new = arith.addf %h, %dh : f32
        
        func.call @tensorlang_print_status(%h_new, %v_new) : (f32, f32) -> ()
        
        %is_crashed = arith.cmpf olt, %h_new, %h_fail : f32
        %is_too_fast = arith.cmpf olt, %v_new, %v_fail : f32
        %crash_cond = arith.andi %is_crashed, %is_too_fast : i1
        
        %is_safe = arith.xori %crash_cond, %true : i1
        tensorlang.assert %is_safe
        
        scf.yield %h_new, %v_new : f32, f32
    }}
    
    return %c0_i32 : i32
  }}
}}
"""

def run_sim(mlir_content):
    with open("temp_sim.mlir", "w") as f:
        f.write(mlir_content)
    
    env = os.environ.copy()
    build_dir = os.path.join(os.getcwd(), "build")
    env["LD_LIBRARY_PATH"] = f"{build_dir}/lib:{build_dir}/runtime:{os.getcwd()}/tensorlang/deps/llama.cpp/build/bin"
    
    try:
        # Use mock runner to avoid AI overhead during data generation
        cmd = [f"{build_dir}/tools/tensorlang-run/tensorlang-run", "--runner=mock", "temp_sim.mlir"]
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return True, result.stdout
        else:
            return False, result.stdout
    except Exception as e:
        return False, str(e)

print("Starting physics search for training data...")
successful_strategies = []

for _ in range(500):
    kp = random.uniform(0.5, 5.0)
    target_v = random.uniform(-3.0, -0.5)
    
    pilot = PILOT_TEMPLATE.format(kp=kp, target_v=target_v)
    full_mlir = generate_full_mlir(pilot)
    
    success, log = run_sim(full_mlir)
    if success:
        print(f"SAFE LANDING FOUND! Kp={kp:.2f}, TargetV={target_v:.2f}")
        # Extract only the relevant part of the MLIR for training
        strategy = {
            "instruction": "Fix the MLIR pilot to land safely.",
            "input": "module { func.func @get_thrust(%h: f32, %v: f32) -> f32 { %c0 = arith.constant 0.0 : f32 return %c0 : f32 } }",
            "output": pilot.strip()
        }
        successful_strategies.append(strategy)
        if len(successful_strategies) >= 50:
            break

with open("tensorlang_training_data.jsonl", "w") as f:
    for s in successful_strategies:
        f.write(json.dumps(s) + "\n")

print(f"Generated {len(successful_strategies)} high-quality training examples.")
