// RUN: tensorlang-opt %s --one-shot-bufferize="bufferize-function-boundaries"

module {
  func.func @main() -> tensor<4x4xf32> {
    %cst = arith.constant dense<1.000000e+00> : tensor<4x4xf32>
    %cst_0 = arith.constant dense<2.000000e+00> : tensor<4x4xf32>
    %cst_1 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<4x4xf32>
    %1 = linalg.fill ins(%cst_1 : f32) outs(%0 : tensor<4x4xf32>) -> tensor<4x4xf32>
    %2 = linalg.matmul ins(%cst, %cst_0 : tensor<4x4xf32>, tensor<4x4xf32>) outs(%1 : tensor<4x4xf32>) -> tensor<4x4xf32>
    return %2 : tensor<4x4xf32>
  }
}
