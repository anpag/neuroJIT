module {
  // Runtime Hooks
  func.func private @tensorlang_assert_fail(i64)
  func.func private @tensorlang_print_status(f32, f32)
  func.func private @tensorlang_start_timer()
  func.func private @tensorlang_stop_timer(f32)

  // --- THE EVOLVING BRAIN (The "Nervous System") ---
  // %arg0: altitude, %arg1: velocity, %arg2: current_fuel
  // Returns: %thrust, %new_memory
  func.func @get_thrust(%arg0: f32, %arg1: f32, %arg2: f32) -> f32 {
    %c1 = arith.constant -1.0 : f32
    %c4 = arith.constant 4.0 : f32
    %error = arith.subf %c1, %arg1 : f32
    %thrust = arith.mulf %error, %c4 : f32
    return %thrust : f32
  }

  // --- THE SIMULATION ENGINE ---
  func.func @main() -> i32 {
    %dt = arith.constant 0.1 : f32
    %gravity = arith.constant -1.62 : f32
    
    // Initial State
    %h0 = arith.constant 100.0 : f32
    %v0 = arith.constant -10.0 : f32
    %f0 = arith.constant 500.0 : f32 // Starting Fuel
    
    %c0_idx = arith.constant 0 : index
    %c1_idx = arith.constant 1 : index
    %c200_idx = arith.constant 200 : index
    %c0_f32 = arith.constant 0.0 : f32
    %c_safe_speed = arith.constant -5.0 : f32
    %true = arith.constant 1 : i1

    func.call @tensorlang_start_timer() : () -> ()

    // Survival Loop: We loop until we hit the ground or run out of fuel.
    %res:3 = scf.for %i = %c0_idx to %c200_idx step %c1_idx 
        iter_args(%h=%h0, %v=%v0, %fuel=%f0) -> (f32, f32, f32) {
        
        // 1. Brain Processing
        %raw_thrust = func.call @get_thrust(%h, %v, %fuel) : (f32, f32, f32) -> f32
        
        // 2. Fuel Constraint: Cannot thrust more than we have fuel for.
        %has_fuel = arith.cmpf ogt, %fuel, %c0_f32 : f32
        %thrust = arith.select %has_fuel, %raw_thrust, %c0_f32 : f32
        
        // 3. Physics Update
        %accel = arith.addf %gravity, %thrust : f32
        %dv = arith.mulf %accel, %dt : f32
        %v_new = arith.addf %v, %dv : f32
        %dh = arith.mulf %v_new, %dt : f32
        %h_new = arith.addf %h, %dh : f32
        
        // 4. Consumption: Thrust costs fuel.
        %c_zero = arith.constant 0.0 : f32
        %is_neg = arith.cmpf olt, %thrust, %c_zero : f32
        %neg_thrust = arith.negf %thrust : f32
        %abs_thrust = arith.select %is_neg, %neg_thrust, %thrust : f32
        
        %fuel_used = arith.mulf %abs_thrust, %dt : f32
        %fuel_new = arith.subf %fuel, %fuel_used : f32
        
        func.call @tensorlang_print_status(%h_new, %v_new) : (f32, f32) -> ()
        
        // 5. Survival Check
        %is_ground = arith.cmpf olt, %h_new, %c0_f32 : f32
        %is_fast   = arith.cmpf olt, %v_new, %c_safe_speed : f32
        %crash_cond = arith.andi %is_ground, %is_fast : i1
        %is_safe = arith.xori %crash_cond, %true : i1
        tensorlang.assert %is_safe
        
        // Break loop if grounded
        %not_ground = arith.xori %is_ground, %true : i1
        %continue = arith.andi %not_ground, %true : i1
        
        scf.yield %h_new, %v_new, %fuel_new : f32, f32, f32
    }

    func.call @tensorlang_stop_timer(%res#1) : (f32) -> ()
    %c0_i32 = arith.constant 0 : i32
    return %c0_i32 : i32
  }
}
