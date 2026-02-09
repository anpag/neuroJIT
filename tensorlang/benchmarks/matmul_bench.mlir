module {
  llvm.func @printf(!llvm.ptr, ...) -> i32
  
  llvm.mlir.global internal constant @str_done("MatMul Done. Value at [0,0]: %f\0A\00")

  func.func @main() -> i32 {
    // 1. Create constants (256x256)
    %A = tensorlang.constant dense<1.0> : tensor<256x256xf32> : tensor<256x256xf32>
    %B = tensorlang.constant dense<2.0> : tensor<256x256xf32> : tensor<256x256xf32>

    // 2. Run MatMul
    %C = tensorlang.matmul %A, %B : tensor<256x256xf32>, tensor<256x256xf32> -> tensor<256x256xf32>
    
    // 3. Print one value to prevent DCE and verify
    %c0 = arith.constant 0 : index
    %val = tensor.extract %C[%c0, %c0] : tensor<256x256xf32>
    %val_f64 = arith.extf %val : f32 to f64
    
    %fmt = llvm.mlir.addressof @str_done : !llvm.ptr
    llvm.call @printf(%fmt, %val_f64) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64) -> i32

    %ret = arith.constant 0 : i32
    return %ret : i32
  }
}
