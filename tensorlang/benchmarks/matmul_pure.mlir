module {
  llvm.func @printf(!llvm.ptr, ...) -> i32
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @free(!llvm.ptr)
  llvm.func @clock_gettime(i32, !llvm.ptr) -> i32

  llvm.mlir.global internal constant @str_time("Time: %f ms\0A\00")
  llvm.mlir.global internal constant @str_header("Benchmarking MatMul (256x256)...\0A\00")

  func.func private @print_time(%start: !llvm.ptr, %end: !llvm.ptr) {
    %start_sec = llvm.load %start : !llvm.ptr -> i64
    %start_nsec_ptr = llvm.getelementptr %start[8] : (!llvm.ptr) -> !llvm.ptr, i8
    %start_nsec = llvm.load %start_nsec_ptr : !llvm.ptr -> i64
    
    %end_sec = llvm.load %end : !llvm.ptr -> i64
    %end_nsec_ptr = llvm.getelementptr %end[8] : (!llvm.ptr) -> !llvm.ptr, i8
    %end_nsec = llvm.load %end_nsec_ptr : !llvm.ptr -> i64
    
    %diff_sec = arith.subi %end_sec, %start_sec : i64
    %diff_nsec = arith.subi %end_nsec, %start_nsec : i64
    
    %f_diff_sec = arith.sitofp %diff_sec : i64 to f64
    %f_diff_nsec = arith.sitofp %diff_nsec : i64 to f64
    
    %c_1000 = arith.constant 1000.0 : f64
    %c_1M = arith.constant 1000000.0 : f64
    
    %ms_sec = arith.mulf %f_diff_sec, %c_1000 : f64
    %ms_nsec = arith.divf %f_diff_nsec, %c_1M : f64
    
    %total_ms = arith.addf %ms_sec, %ms_nsec : f64
    
    %fmt = llvm.mlir.addressof @str_time : !llvm.ptr
    llvm.call @printf(%fmt, %total_ms) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64) -> i32
    return
  }

  func.func @main() -> i32 {
    %c16 = arith.constant 16 : i64
    %ts_start = llvm.call @malloc(%c16) : (i64) -> !llvm.ptr
    %ts_end = llvm.call @malloc(%c16) : (i64) -> !llvm.ptr
    %clk_id = arith.constant 1 : i32 

    %hdr = llvm.mlir.addressof @str_header : !llvm.ptr
    llvm.call @printf(%hdr) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32

    // Warmup
    %A = tensorlang.constant dense<1.0> : tensor<256x256xf32> : tensor<256x256xf32>
    %B = tensorlang.constant dense<2.0> : tensor<256x256xf32> : tensor<256x256xf32>
    %C_warm = tensorlang.matmul %A, %B : tensor<256x256xf32>, tensor<256x256xf32> -> tensor<256x256xf32>

    // Timed Region
    llvm.call @clock_gettime(%clk_id, %ts_start) : (i32, !llvm.ptr) -> i32
    
    %C = tensorlang.matmul %A, %B : tensor<256x256xf32>, tensor<256x256xf32> -> tensor<256x256xf32>
    
    llvm.call @clock_gettime(%clk_id, %ts_end) : (i32, !llvm.ptr) -> i32
    
    func.call @print_time(%ts_start, %ts_end) : (!llvm.ptr, !llvm.ptr) -> ()

    // Verify a value to prevent DCE
    %c0 = arith.constant 0 : index
    %val = tensor.extract %C[%c0, %c0] : tensor<256x256xf32>
    
    llvm.call @free(%ts_start) : (!llvm.ptr) -> ()
    llvm.call @free(%ts_end) : (!llvm.ptr) -> ()
    
    %ret = arith.constant 0 : i32
    return %ret : i32
  }
}
