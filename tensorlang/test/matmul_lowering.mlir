// RUN: tensorlang-opt %s --convert-tensorlang-to-linalg | FileCheck %s

module {
  func.func @main() -> tensor<4x4xf32> {
    // A = constant 4x4
    %A = tensorlang.constant dense<1.0> : tensor<4x4xf32> : tensor<4x4xf32>
    
    // B = constant 4x4
    %B = tensorlang.constant dense<2.0> : tensor<4x4xf32> : tensor<4x4xf32>
    
    // C = A * B
    %C = tensorlang.matmul %A, %B : tensor<4x4xf32>, tensor<4x4xf32> -> tensor<4x4xf32>
    
    return %C : tensor<4x4xf32>
  }
}

// CHECK-LABEL: func.func @main
// CHECK: %[[A:.*]] = arith.constant dense<1.000000e+00> : tensor<4x4xf32>
// CHECK: %[[B:.*]] = arith.constant dense<2.000000e+00> : tensor<4x4xf32>
// CHECK: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[INIT:.*]] = tensor.empty() : tensor<4x4xf32>
// CHECK: %[[FILLED:.*]] = linalg.fill ins(%[[ZERO]] : f32) outs(%[[INIT]] : tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK: %[[RES:.*]] = linalg.matmul ins(%[[A]], %[[B]] : tensor<4x4xf32>, tensor<4x4xf32>) outs(%[[FILLED]] : tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK: return %[[RES]] : tensor<4x4xf32>
