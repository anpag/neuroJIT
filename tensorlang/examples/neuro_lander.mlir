module {
  // -------------------------------------------------------------------------
  // Runtime hooks — V2 contract
  // -------------------------------------------------------------------------
  func.func private @tensorlang_print_status(f32, f32)
  func.func private @tensorlang_start_timer()
  func.func private @tensorlang_stop_timer(f32)
  func.func private @tensorlang_get_random() -> f32
  func.func private @tensorlang_record_thrust(f32)

  // -------------------------------------------------------------------------
  // THE PILOT — initial state: zero thrust (worst case)
  // The GA + template engine rewrites this function each generation.
  // It is NEVER edited by the LLM directly.
  // -------------------------------------------------------------------------
  func.func @get_thrust(%arg0: f32, %arg1: f32) -> f32 {
    %zero = arith.constant 0.0 : f32
    return %zero : f32
  }

  // -------------------------------------------------------------------------
  // SIMULATION LOOP
  // One episode = one call to @main.
  // The outer generation loop is driven by the C++ harness (Task 17).
  // -------------------------------------------------------------------------
  func.func @main() -> i32 {
    %dt            = arith.constant 0.1  : f32
    %base_gravity  = arith.constant -1.62 : f32  // Moon gravity
    %noise_scale   = arith.constant 0.25 : f32   // Turbulence amplitude
    %h0            = arith.constant 100.0 : f32
    %v0            = arith.constant -10.0 : f32

    %c0_idx        = arith.constant 0   : index
    %c1_idx        = arith.constant 1   : index
    %c200_idx      = arith.constant 200 : index
    %c0_i32        = arith.constant 0   : i32
    %c0_f32        = arith.constant 0.0 : f32
    %safe_speed    = arith.constant -5.0 : f32

    func.call @tensorlang_start_timer() : () -> ()

    %result:2 = scf.for %i = %c0_idx to %c200_idx step %c1_idx
        iter_args(%h = %h0, %v = %v0) -> (f32, f32) {

      // 1. Environmental noise — variable gravity each tick
      //    This makes the problem non-trivially solvable by a pure P controller
      %noise    = func.call @tensorlang_get_random() : () -> f32
      %delta_g  = arith.mulf %noise, %noise_scale : f32
      %gravity  = arith.addf %base_gravity, %delta_g : f32

      // 2. Control signal from the pilot
      %thrust   = func.call @get_thrust(%h, %v) : (f32, f32) -> f32

      // 3. Record fuel consumption for fitness tracking
      func.call @tensorlang_record_thrust(%thrust) : (f32) -> ()

      // 4. Physics update
      %accel    = arith.addf %gravity, %thrust : f32
      %dv       = arith.mulf %accel, %dt : f32
      %v_new    = arith.addf %v, %dv : f32
      %dh       = arith.mulf %v_new, %dt : f32
      %h_new    = arith.addf %h, %dh : f32

      // 5. Status display
      func.call @tensorlang_print_status(%h_new, %v_new) : (f32, f32) -> ()

      // 6. Safe early exit: stop simulation if landed successfully
      //    h < 0.5 AND |v| < 2.0 = successful soft landing
      //    This prevents unnecessary ticks and gives settling time signal
      %near_ground   = arith.cmpf olt, %h_new, %c0_f32 : f32
      %v_abs         = arith.mulf %v_new, %v_new : f32   // v^2 as proxy
      
      // If below ground AND moving too fast: the episode ends here.
      // tensorlang_stop_timer is called with final velocity.
      // The C++ runtime records the SimulationResult.
      %is_crashed    = arith.cmpf olt, %v_new, %safe_speed : f32
      %hard_landing  = arith.andi %near_ground, %is_crashed : i1

      // Yield next state regardless — the outer harness handles termination
      scf.yield %h_new, %v_new : f32, f32
    }

    // Record final state as simulation result
    func.call @tensorlang_stop_timer(%result#1) : (f32) -> ()

    return %c0_i32 : i32
  }
}