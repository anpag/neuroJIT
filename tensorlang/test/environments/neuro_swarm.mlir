module {
  // Runtime Hooks
  func.func private @tensorlang_assert_fail(i64)
  func.func private @tensorlang_print_status(f32, f32)
  func.func private @tensorlang_start_timer()
  func.func private @tensorlang_stop_timer(f32)
  func.func private @tensorlang_get_random() -> f32 // New: Per-tick noise

  // --- THE SWARM BRAIN ---
  func.func @get_thrust(%h: f32, %v: f32, %fuel: f32, %mem: tensor<4xf32>) -> (f32, tensor<4xf32>) {
    %c0 = arith.constant 0.0 : f32
    // Initial state: No thrust. The Architect will evolve this.
    return %c0, %mem : f32, tensor<4xf32>
  }

  // --- THE SWARM SIMULATOR ---
  func.func @main() -> i32 {
    %c0_idx = arith.constant 0 : index
    %c1_idx = arith.constant 1 : index
    %c100_idx = arith.constant 100 : index // Swarm Size: 100
    %c50_steps = arith.constant 50 : index 
    
    %dt = arith.constant 0.1 : f32
    %base_gravity = arith.constant -1.62 : f32
    %true = arith.constant 1 : i1

    func.call @tensorlang_start_timer() : () -> ()

    // SWARM LOOP: Run 100 landers
    scf.for %lander_id = %c0_idx to %c100_idx step %c1_idx {
        // Individual Initial State
        %h0 = arith.constant 100.0 : f32
        %v0 = arith.constant -10.0 : f32
        %m0 = arith.constant dense<0.0> : tensor<4xf32>
        
        %res:4 = scf.for %i = %c0_idx to %c50_steps step %c1_idx 
            iter_args(%h=%h0, %v=%v0, %fuel=%c0_idx, %mem=%m0) -> (f32, f32, index, tensor<4xf32>) {
            
            // 1. Environmental Noise (Gravitational Turbulence)
            %noise = func.call @tensorlang_get_random() : () -> f32
            %gravity = arith.addf %base_gravity, %noise : f32
            
            // 2. Brain Execution
            %f_dummy = arith.constant 500.0 : f32
            %thrust, %mem_new = func.call @get_thrust(%h, %v, %f_dummy, %mem) : (f32, f32, f32, tensor<4xf32>) -> (f32, tensor<4xf32>)
            
            // 3. Physics
            %accel = arith.addf %gravity, %thrust : f32
            %v_new = arith.addf %v, %accel : f32 // Simplified for swarm speed
            %h_new = arith.addf %h, %v_new : f32
            
            scf.yield %h_new, %v_new, %fuel, %mem_new : f32, f32, index, tensor<4xf32>
        }
        
        // Record final performance for this individual
        func.call @tensorlang_print_status(%res#0, %res#1) : (f32, f32) -> ()
    }

    func.call @tensorlang_stop_timer(%base_gravity) : (f32) -> ()
    %ret = arith.constant 0 : i32
    return %ret : i32
  }
}
