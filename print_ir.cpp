#include "tensorlang/Runtime/MLIRTemplates.h"
#include <iostream>
int main() {
  mlir::tensorlang::ControlStrategy s;
  s.kp = 1.0; s.ki = 0.0; s.kd = 0.0; s.targetVelocity = -1.0; s.thrustClampMax = 5.0;
  std::cout << mlir::tensorlang::instantiateControllerMLIR(s);
  return 0;
}
