module {
  func.func private @tensorlang_assert_fail(i64)
  func.func private @tensorlang_print_status(f32, f32)

  
  func.func @get_thrust(%arg0: f32, %arg1: f32) -> f32 {
    %target_v = arith.constant -0.53 : f32
    %diff = arith.subf %target_v, %arg1 : f32
    %kp = arith.constant 4.58 : f32
    %thrust = arith.mulf %diff, %kp : f32
    return %thrust : f32
  }


  func.func @main() -> i32 {
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
    
    %res:2 = scf.for %i = %c0 to %c200 step %c1 iter_args(%h = %h0, %v = %v0) -> (f32, f32) {
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
    }
    
    return %c0_i32 : i32
  }
}
