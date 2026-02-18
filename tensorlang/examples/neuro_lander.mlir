module {
  // Runtime Hooks
  func.func private @tensorlang_assert_fail(i64)
  func.func private @tensorlang_print_status(f32, f32)

  // --- THE PILOT (System 1) ---
  // Currently: Terrible pilot. Does nothing.
  func.func @get_thrust(%h: f32, %v: f32) -> f32 {
    %c0 = arith.constant 0.0 : f32
    return %c0 : f32
  }

  // --- THE SIMULATION ---
  func.func @main() -> i32 {
    %dt = arith.constant 0.1 : f32
    %gravity = arith.constant -1.62 : f32 // Moon Gravity
    
    // Initial State: 100m up, falling at -10m/s
    %h0 = arith.constant 100.0 : f32
    %v0 = arith.constant -10.0 : f32
    
    %c0_idx = arith.constant 0 : index
    %c1_idx = arith.constant 1 : index
    %c200_idx = arith.constant 200 : index
    %c0_i32 = arith.constant 0 : i32
    %c0_f32 = arith.constant 0.0 : f32
    %c_safe_speed = arith.constant -5.0 : f32
    %true = arith.constant 1 : i1

    // Simulation Loop (200 ticks)
    %final_res:2 = scf.for %i = %c0_idx to %c200_idx step %c1_idx iter_args(%h=%h0, %v=%v0) -> (f32, f32) {
        
        // 1. Get Control Signal
        %thrust = func.call @get_thrust(%h, %v) : (f32, f32) -> f32
        
        // 2. Physics Update
        %accel = arith.addf %gravity, %thrust : f32
        %dv = arith.mulf %accel, %dt : f32
        %v_new = arith.addf %v, %dv : f32
        %dh = arith.mulf %v_new, %dt : f32
        %h_new = arith.addf %h, %dh : f32
        
        // 3. Render
        func.call @tensorlang_print_status(%h_new, %v_new) : (f32, f32) -> ()
        
        // 4. THE SAFETY CHECK
        // If height < 0 (Impact) AND velocity < -5.0 (Fast), we die.
        %is_ground = arith.cmpf olt, %h_new, %c0_f32 : f32
        %is_fast   = arith.cmpf olt, %v_new, %c_safe_speed : f32
        %crash_cond = arith.andi %is_ground, %is_fast : i1
        
        // "Assert that we are NOT crashing"
        %is_safe = arith.xori %crash_cond, %true : i1
        tensorlang.assert %is_safe
        
        scf.yield %h_new, %v_new : f32, f32
    }
    
    return %c0_i32 : i32
  }
}
