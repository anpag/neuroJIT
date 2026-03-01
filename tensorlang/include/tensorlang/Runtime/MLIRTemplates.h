#ifndef TENSORLANG_RUNTIME_MLIRTEMPLATES_H
#define TENSORLANG_RUNTIME_MLIRTEMPLATES_H

#include "tensorlang/Runtime/OptimizationStrategy.h"
#include "llvm/Support/FormatVariadic.h"
#include <string>

namespace mlir {
namespace tensorlang {

/// Generates syntactically guaranteed-valid MLIR for a PID/PD controller.
/// Parameters come from ControlStrategy; the template structure is hardcoded.
/// The LLM never writes MLIR directly — it only produces the parameter values.
inline std::string instantiateControllerMLIR(const ControlStrategy& s) {
  return llvm::formatv(R"mlir(
module {{
  func.func private @tensorlang_assert_fail(i64)
  func.func private @tensorlang_print_status(f32, f32)
  func.func private @tensorlang_start_timer()
  func.func private @tensorlang_stop_timer(f32)
  func.func private @tensorlang_get_random() -> f32

  func.func @get_thrust(%arg0: f32, %arg1: f32) -> f32 {{
    // PID Controller - parameters from ControlStrategy
    // arg0 = altitude, arg1 = velocity
    %kp          = arith.constant {0:f} : f32
    %ki          = arith.constant {1:f} : f32
    %kd          = arith.constant {2:f} : f32
    %target_v    = arith.constant {3:f} : f32
    %clamp_max   = arith.constant {4:f} : f32
    %clamp_min   = arith.constant 0.0 : f32

    // P term: proportional to velocity error
    %v_error     = arith.subf %target_v, %arg1 : f32
    %p_term      = arith.mulf %kp, %v_error : f32

    // D term: derivative of altitude (damp oscillation)
    %d_term      = arith.mulf %kd, %arg0 : f32

    // Combined thrust
    %raw_thrust  = arith.addf %p_term, %d_term : f32

    // Clamp: thrust must be in [clamp_min, clamp_max]
    %above_max   = arith.cmpf ogt, %raw_thrust, %clamp_max : f32
    %clamped_hi  = arith.select %above_max, %clamp_max, %raw_thrust : f32
    %below_min   = arith.cmpf olt, %clamped_hi, %clamp_min : f32
    %result      = arith.select %below_min, %clamp_min, %clamped_hi : f32

    return %result : f32
  }}
}}
)mlir",
    s.kp, s.ki, s.kd, s.targetVelocity, s.thrustClampMax)
    .str();
}

} // namespace tensorlang
} // namespace mlir

#endif // TENSORLANG_RUNTIME_MLIRTEMPLATES_H