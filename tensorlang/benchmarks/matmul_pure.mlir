module {
  func.func @matmul(%A: memref<128x128xf32>, %B: memref<128x128xf32>, %C: memref<128x128xf32>) attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    
    scf.for %i = %c0 to %c128 step %c1 {
      scf.for %j = %c0 to %c128 step %c1 {
        %sum_initial = memref.load %C[%i, %j] : memref<128x128xf32>
        %final_sum = scf.for %k = %c0 to %c128 step %c1 iter_args(%sum = %sum_initial) -> (f32) {
          %a = memref.load %A[%i, %k] : memref<128x128xf32>
          %b = memref.load %B[%k, %j] : memref<128x128xf32>
          %prod = arith.mulf %a, %b : f32
          %next_sum = arith.addf %sum, %prod : f32
          scf.yield %next_sum : f32
        }
        memref.store %final_sum, %C[%i, %j] : memref<128x128xf32>
      }
    }
    return
  }
}
