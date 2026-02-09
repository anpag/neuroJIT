// RUN: tensorlang-opt %s --verify-linearity -verify-diagnostics

module {
  func.func @test_all() {
    // Double use
    // expected-error @+1 {{result with linear type must be used exactly once}}
    %0 = tensorlang.constant dense<1.0> : tensor<4xf32> : !tensorlang.tensor<f32, 4>
    tensorlang.print %0 : !tensorlang.tensor<f32, 4>
    tensorlang.print %0 : !tensorlang.tensor<f32, 4>

    // Unused
    // expected-error @+1 {{result with linear type must be used exactly once}}
    %1 = tensorlang.constant dense<1.0> : tensor<4xf32> : !tensorlang.tensor<f32, 4>
    
    return
  }
}
