module {
  // -------------------------------------------------------------------------
  // Runtime hooks
  // -------------------------------------------------------------------------
  func.func private @tensorlang_print_status(f32, f32)
  func.func private @tensorlang_start_timer()
  func.func private @tensorlang_stop_timer(f32)
  func.func private @tensorlang_get_random() -> f32
  func.func private @tensorlang_record_thrust(f32)

  // -------------------------------------------------------------------------
  // PILOT: takes altitude, vertical velocity, horizontal velocity
  // Returns: vertical_thrust (horizontal thrust is 0 in initial state)
  // A pure P controller on vertical velocity CANNOT stabilize lateral drift.
  // A PD controller using altitude for derivative IS needed here.
  // -------------------------------------------------------------------------
  func.func @get_thrust(%arg0: f32, %arg1: f32, %arg2: f32) -> f32 {
    %zero = arith.constant 0.0 : f32
    return %zero : f32
  }

  func.func @main() -> i32 {
    %dt           = arith.constant 0.1   : f32
    %g_vert       = arith.constant -1.62 : f32
    %noise_vert   = arith.constant 0.3   : f32
    %noise_horiz  = arith.constant 0.15  : f32

    // Initial conditions — 2D state
    %h0    = arith.constant 100.0  : f32  // altitude
    %vv0   = arith.constant -10.0  : f32  // vertical velocity
    %x0    = arith.constant 0.0    : f32  // horizontal position
    %vh0   = arith.constant 2.0    : f32  // horizontal drift (harder problem)

    %c0_idx   = arith.constant 0   : index
    %c1_idx   = arith.constant 1   : index
    %c200_idx = arith.constant 200 : index
    %c0_i32   = arith.constant 0   : i32
    %c0_f32   = arith.constant 0.0 : f32
    %safe_v   = arith.constant -4.0 : f32 // tighter safety constraint
    %safe_x   = arith.constant 20.0 : f32 // must land within 20m of center

    func.call @tensorlang_start_timer() : () -> ()

    %res:4 = scf.for %i = %c0_idx to %c200_idx step %c1_idx
        iter_args(%h = %h0, %vv = %vv0, %x = %x0, %vh = %vh0)
        -> (f32, f32, f32, f32) {

      // Variable gravity
      %gnoise = func.call @tensorlang_get_random() : () -> f32
      %dg     = arith.mulf %gnoise, %noise_vert : f32
      %g      = arith.addf %g_vert, %dg : f32

      // Horizontal wind
      %wnoise = func.call @tensorlang_get_random() : () -> f32
      %wind   = arith.mulf %wnoise, %noise_horiz : f32

      // Thrust control
      %thrust = func.call @get_thrust(%h, %vv, %vh) : (f32, f32, f32) -> f32
      func.call @tensorlang_record_thrust(%thrust) : (f32) -> ()

      // Vertical physics
      %a_vert  = arith.addf %g, %thrust : f32
      %dvv     = arith.mulf %a_vert, %dt : f32
      %vv_new  = arith.addf %vv, %dvv : f32
      %dh      = arith.mulf %vv_new, %dt : f32
      %h_new   = arith.addf %h, %dh : f32

      // Horizontal physics (wind only, no lateral thrust yet)
      %dvh   = arith.mulf %wind, %dt : f32
      %vh_new = arith.addf %vh, %dvh : f32
      %dx    = arith.mulf %vh_new, %dt : f32
      %x_new = arith.addf %x, %dx : f32

      func.call @tensorlang_print_status(%h_new, %vv_new) : (f32, f32) -> ()
      scf.yield %h_new, %vv_new, %x_new, %vh_new : f32, f32, f32, f32
    }

    func.call @tensorlang_stop_timer(%res#1) : (f32) -> ()
    return %c0_i32 : i32
  }
}