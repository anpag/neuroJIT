module {
  // Runtime Hooks
  func.func private @tensorlang_assert_fail(i64)
  func.func private @tensorlang_print_status(f32, f32)
  func.func private @tensorlang_start_timer()
  func.func private @tensorlang_stop_timer(f32)
  func.func private @tensorlang_get_random() -> f32

  // Scalar Brain (Stable)
  func.func @get_thrust(%h: f32, %v: f32) -> f32 {
    %c0 = arith.constant 0.0 : f32
    return %c0 : f32
  }

  // --- THE SWARM SIMULATOR ---
  func.func @main() -> i32 {
    %c0_idx = arith.constant 0 : index
    %c1_idx = arith.constant 1 : index
    %c100_idx = arith.constant 100 : index 
    %c50_steps = arith.constant 50 : index 
    
    %dt = arith.constant 0.1 : f32
    %base_gravity = arith.constant -1.62 : f32

    func.call @tensorlang_start_timer() : () -> ()

    // SWARM LOOP
    scf.for %lander_id = %c0_idx to %c100_idx step %c1_idx {
        %h0 = arith.constant 100.0 : f32
        %v0 = arith.constant -10.0 : f32
        
        %res:2 = scf.for %i = %c0_idx to %c50_steps step %c1_idx 
            iter_args(%h=%h0, %v=%v0) -> (f32, f32) {
            
            %thrust = func.call @get_thrust(%h, %v) : (f32, f32) -> f32
            
            %accel = arith.addf %base_gravity, %thrust : f32
            %dv = arith.mulf %accel, %dt : f32
            %v_new = arith.addf %v, %dv : f32
            %dh = arith.mulf %v_new, %dt : f32
            %h_new = arith.addf %h, %dh : f32
            
            scf.yield %h_new, %v_new : f32, f32
        }
        
        func.call @tensorlang_print_status(%res#0, %res#1) : (f32, f32) -> ()
    }

    %grav_scalar = arith.constant -1.62 : f32
    func.call @tensorlang_stop_timer(%grav_scalar) : (f32) -> ()
    %ret = arith.constant 0 : i32
    return %ret : i32
  }
}
