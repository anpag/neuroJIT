module {
  func.func @matmul(%A: memref<128x128xf32>, %B: memref<128x128xf32>, %C: memref<128x128xf32>) attributes {llvm.emit_c_interface} {
    affine.for %i = 0 to 128 {
      affine.for %j = 0 to 128 {
        %sum_initial = affine.load %C[%i, %j] : memref<128x128xf32>
        %final_sum = affine.for %k = 0 to 128 iter_args(%sum = %sum_initial) -> (f32) {
          %a = affine.load %A[%i, %k] : memref<128x128xf32>
          %b = affine.load %B[%k, %j] : memref<128x128xf32>
          %prod = arith.mulf %a, %b : f32
          %next_sum = arith.addf %sum, %prod : f32
          affine.yield %next_sum : f32
        }
        affine.store %final_sum, %C[%i, %j] : memref<128x128xf32>
      }
    }
    return
  }
}
