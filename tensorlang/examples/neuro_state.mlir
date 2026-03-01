module {
  // Runtime Hooks
  func.func private @tensorlang_assert_fail(i64)
  func.func private @tensorlang_print_status(f32, f32)
  func.func private @tensorlang_start_timer()
  func.func private @tensorlang_stop_timer(f32)

  // --- THE EVOLVING NERVOUS SYSTEM ---
  // %arg0: altitude, %arg1: velocity, %arg2: fuel
  // %arg3: STATE TENSOR (Memory of the system)
  // Returns: %thrust, %NEW_STATE_TENSOR
  func.func @get_thrust(%arg0: f32, %arg1: f32, %arg2: f32, %arg3: tensor<4xf32>) -> (f32, tensor<4xf32>) {
    %c0 = arith.constant 0.0 : f32
    // For now, it just returns 0 thrust and unchanged state.
    // The Evolution Engine will rewrite this into a "thinking" function.
    return %c0, %arg3 : f32, tensor<4xf32>
  }

  // --- THE EVOLUTIONARY ENVIRONMENT ---
  func.func @main() -> i32 {
    %c0_gen = arith.constant 0 : index
    %c1_gen = arith.constant 1 : index
    %c3_gen = arith.constant 3 : index 

    scf.for %gen = %c0_gen to %c3_gen step %c1_gen {
        %dt = arith.constant 0.1 : f32
        %gravity = arith.constant -1.62 : f32
        
        // Initial Physical State
        %h0 = arith.constant 100.0 : f32
        %v0 = arith.constant -10.0 : f32
        %f0 = arith.constant 500.0 : f32 
        
        // --- INITIAL MENTAL STATE (Memory) ---
        %m0 = arith.constant dense<0.0> : tensor<4xf32>
        
        %c0_idx = arith.constant 0 : index
        %c1_idx = arith.constant 1 : index
        %c100_idx = arith.constant 100 : index
        %c0_f32 = arith.constant 0.0 : f32
        %c_safe_speed = arith.constant -5.0 : f32
        %true = arith.constant 1 : i1

        func.call @tensorlang_start_timer() : () -> ()

        // Survival Loop with Stateful Memory
        %res:4 = scf.for %i = %c0_idx to %c100_idx step %c1_idx 
            iter_args(%h=%h0, %v=%v0, %fuel=%f0, %mem=%m0) -> (f32, f32, f32, tensor<4xf32>) {
            
            // 1. Recursive Processing (Inputs + Memory)
            %raw_thrust, %mem_new = func.call @get_thrust(%h, %v, %fuel, %mem) : (f32, f32, f32, tensor<4xf32>) -> (f32, tensor<4xf32>)
            
            // 2. Physics & Constraints
            %has_fuel = arith.cmpf ogt, %fuel, %c0_f32 : f32
            %thrust = arith.select %has_fuel, %raw_thrust, %c0_f32 : f32
            
            %accel = arith.addf %gravity, %thrust : f32
            %dv = arith.mulf %accel, %dt : f32
            %v_new = arith.addf %v, %dv : f32
            %dh = arith.mulf %v_new, %dt : f32
            %h_new = arith.addf %h, %dh : f32
            
            // 3. Survival Check
            %is_ground = arith.cmpf olt, %h_new, %c0_f32 : f32
            %is_fast   = arith.cmpf olt, %v_new, %c_safe_speed : f32
            %crash_cond = arith.andi %is_ground, %is_fast : i1
            %is_safe = arith.xori %crash_cond, %true : i1
            tensorlang.assert %is_safe
            
            func.call @tensorlang_print_status(%h_new, %v_new) : (f32, f32) -> ()
            scf.yield %h_new, %v_new, %fuel, %mem_new : f32, f32, f32, tensor<4xf32>
        }

        func.call @tensorlang_stop_timer(%res#1) : (f32) -> ()
    }

    %c0_i32 = arith.constant 0 : i32
    return %c0_i32 : i32
  }
}
