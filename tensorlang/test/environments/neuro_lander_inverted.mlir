module {
  // -------------------------------------------------------------------------
  // Runtime hooks — Async Auto-Healing API
  // -------------------------------------------------------------------------
  func.func private @tensorlang_print_status(f32, f32)
  func.func private @tensorlang_start_timer()
  func.func private @tensorlang_stop_timer(f32)
  func.func private @tensorlang_get_random() -> f32
  func.func private @tensorlang_record_thrust(f32)
  func.func private @tensorlang_assert_fail(i64)

  // -------------------------------------------------------------------------
  // THE PILOT — initial state: zero thrust (worst case)
  // -------------------------------------------------------------------------
  func.func @get_thrust(%arg0: f32, %arg1: f32) -> f32 {
    %c_neg = arith.constant -1.5 : f32
    %thrust = arith.mulf %arg1, %c_neg : f32
    return %thrust : f32
  }

  func.func @sim_main() -> i32 {
    %dt            = arith.constant 0.1  : f32
    %base_gravity  = arith.constant -1.62 : f32
    %noise_scale   = arith.constant 0.0 : f32 // DETERMINISTIC
    %h0            = arith.constant 100.0 : f32
    %v0            = arith.constant -10.0 : f32

    %c0_idx        = arith.constant 0   : index
    %c1_idx        = arith.constant 1   : index
    %c200_idx      = arith.constant 200 : index
    %c0_i32        = arith.constant 0   : i32
    %c0_f32        = arith.constant 0.0 : f32
    %safe_speed    = arith.constant -5.0 : f32

    func.call @tensorlang_start_timer() : () -> ()

    %c1_i1 = arith.constant 1 : i1
    %c0_i1 = arith.constant 0 : i1
    %result:3 = scf.for %i = %c0_idx to %c200_idx step %c1_idx
        iter_args(%h = %h0, %v = %v0, %already_crashed = %c0_i1) -> (f32, f32, i1) {

      // Skip physics if already crashed
      %res_h, %res_v, %res_crashed = scf.if %already_crashed -> (f32, f32, i1) {
        scf.yield %h, %v, %c1_i1 : f32, f32, i1
      } else {
        %noise    = func.call @tensorlang_get_random() : () -> f32
        %delta_g  = arith.mulf %noise, %noise_scale : f32
        %gravity  = arith.addf %base_gravity, %delta_g : f32

        %thrust   = func.call @get_thrust(%h, %v) : (f32, f32) -> f32
        func.call @tensorlang_record_thrust(%thrust) : (f32) -> ()

        %accel    = arith.addf %gravity, %thrust : f32
        %dv       = arith.mulf %accel, %dt : f32
        %v_new    = arith.addf %v, %dv : f32
        %dh       = arith.mulf %v_new, %dt : f32
        %h_new    = arith.addf %h, %dh : f32

        func.call @tensorlang_print_status(%h_new, %v_new) : (f32, f32) -> ()

        %near_ground   = arith.cmpf olt, %h_new, %c0_f32 : f32
        %is_crashed    = arith.cmpf olt, %v_new, %safe_speed : f32
        %hard_landing  = arith.andi %near_ground, %is_crashed : i1

        scf.if %hard_landing {
          %loc_code = arith.constant 42 : i64
          func.call @tensorlang_assert_fail(%loc_code) : (i64) -> ()
        }
        
        scf.yield %h_new, %v_new, %hard_landing : f32, f32, i1
      }

      scf.yield %res_h, %res_v, %res_crashed : f32, f32, i1
    }

    func.call @tensorlang_stop_timer(%result#1) : (f32) -> ()
    return %c0_i32 : i32
  }
}