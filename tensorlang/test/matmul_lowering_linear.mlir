// RUN: tensorlang-opt %s --convert-tensorlang-to-linalg

module {
  func.func @main() -> !tensorlang.tensor<f32, 4, 4> {
    // A = constant 4x4
    %A = tensorlang.constant dense<1.0> : tensor<4x4xf32> : !tensorlang.tensor<f32, 4, 4>
    
    // B = constant 4x4
    %B = tensorlang.constant dense<2.0> : tensor<4x4xf32> : !tensorlang.tensor<f32, 4, 4>
    
    // C = A * B
    %C = tensorlang.matmul %A, %B : !tensorlang.tensor<f32, 4, 4>, !tensorlang.tensor<f32, 4, 4> -> !tensorlang.tensor<f32, 4, 4>
    
    return %C : !tensorlang.tensor<f32, 4, 4>
  }
}
