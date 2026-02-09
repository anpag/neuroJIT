module {
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @free(!llvm.ptr)
  llvm.func @clock_gettime(i32, !llvm.ptr) -> i32
  llvm.func @printf(!llvm.ptr, ...) -> i32
  
  llvm.func @tensorlang_get_ir() -> !llvm.ptr
  llvm.func @tensorlang_query_model(!llvm.ptr) -> !llvm.ptr
  llvm.func @tensorlang_compile(!llvm.ptr) -> i32
  llvm.func @tensorlang_get_symbol_address(!llvm.ptr) -> !llvm.ptr

  llvm.mlir.global internal constant @str_orig("Running Original Conv2D (64x64 input, 3x3 kernel)...\0A\00")
  llvm.mlir.global internal constant @str_opt("Running Optimized Conv2D...\0A\00")
  llvm.mlir.global internal constant @str_time("Time: %f ms\0A\00")
  llvm.mlir.global internal constant @str_sym("main_optimized\00")

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

  func.func @conv2d_naive() -> i32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c62 = arith.constant 62 : index
    %c64 = arith.constant 64 : index
    
    %f1 = arith.constant 1.0 : f32
    %f0 = arith.constant 0.0 : f32
    
    %Input = memref.alloc() : memref<64x64xf32>
    %Kernel = memref.alloc() : memref<3x3xf32>
    %Output = memref.alloc() : memref<62x62xf32>
    
    linalg.fill ins(%f1 : f32) outs(%Input : memref<64x64xf32>)
    linalg.fill ins(%f1 : f32) outs(%Kernel : memref<3x3xf32>)
    linalg.fill ins(%f0 : f32) outs(%Output : memref<62x62xf32>)
    
    scf.for %h = %c0 to %c62 step %c1 {
      scf.for %w = %c0 to %c62 step %c1 {
        scf.for %kh = %c0 to %c3 step %c1 {
          scf.for %kw = %c0 to %c3 step %c1 {
             %ih = arith.addi %h, %kh : index
             %iw = arith.addi %w, %kw : index
             
             %in_val = memref.load %Input[%ih, %iw] : memref<64x64xf32>
             %k_val = memref.load %Kernel[%kh, %kw] : memref<3x3xf32>
             %out_val = memref.load %Output[%h, %w] : memref<62x62xf32>
             
             %prod = arith.mulf %in_val, %k_val : f32
             %sum = arith.addf %out_val, %prod : f32
             
             memref.store %sum, %Output[%h, %w] : memref<62x62xf32>
          }
        }
      }
    }
    
    %val = memref.load %Output[%c0, %c0] : memref<62x62xf32>
    
    memref.dealloc %Input : memref<64x64xf32>
    memref.dealloc %Kernel : memref<3x3xf32>
    memref.dealloc %Output : memref<62x62xf32>
    
    %ret = arith.constant 0 : i32
    return %ret : i32
  }

  func.func @main() -> i32 {
    %c16 = arith.constant 16 : i64
    %ts_start = llvm.call @malloc(%c16) : (i64) -> !llvm.ptr
    %ts_end = llvm.call @malloc(%c16) : (i64) -> !llvm.ptr
    %clk_id = arith.constant 1 : i32 

    // 1. Run Original
    %fmt_orig = llvm.mlir.addressof @str_orig : !llvm.ptr
    llvm.call @printf(%fmt_orig) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
    
    llvm.call @clock_gettime(%clk_id, %ts_start) : (i32, !llvm.ptr) -> i32
    func.call @conv2d_naive() : () -> i32
    llvm.call @clock_gettime(%clk_id, %ts_end) : (i32, !llvm.ptr) -> i32
    func.call @print_time(%ts_start, %ts_end) : (!llvm.ptr, !llvm.ptr) -> ()

    // 2. Optimization Loop
    %ir_ptr = llvm.call @tensorlang_get_ir() : () -> !llvm.ptr
    %new_ir_ptr = llvm.call @tensorlang_query_model(%ir_ptr) : (!llvm.ptr) -> !llvm.ptr
    %res = llvm.call @tensorlang_compile(%new_ir_ptr) : (!llvm.ptr) -> i32
    
    %sym = llvm.mlir.addressof @str_sym : !llvm.ptr
    %fn_ptr = llvm.call @tensorlang_get_symbol_address(%sym) : (!llvm.ptr) -> !llvm.ptr
    
    // 3. Run Optimized
    %fmt_opt = llvm.mlir.addressof @str_opt : !llvm.ptr
    llvm.call @printf(%fmt_opt) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
    
    llvm.call @clock_gettime(%clk_id, %ts_start) : (i32, !llvm.ptr) -> i32
    %ret_opt = llvm.call %fn_ptr() : !llvm.ptr, () -> i32
    llvm.call @clock_gettime(%clk_id, %ts_end) : (i32, !llvm.ptr) -> i32
    func.call @print_time(%ts_start, %ts_end) : (!llvm.ptr, !llvm.ptr) -> ()

    llvm.call @free(%ts_start) : (!llvm.ptr) -> ()
    llvm.call @free(%ts_end) : (!llvm.ptr) -> ()
    
    %c0_i32 = arith.constant 0 : i32
    return %c0_i32 : i32
  }
}
