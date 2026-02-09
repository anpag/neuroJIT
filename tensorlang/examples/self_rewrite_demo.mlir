module {
  llvm.func @tensorlang_get_ir() -> !llvm.ptr
  llvm.func @tensorlang_query_model(!llvm.ptr) -> !llvm.ptr
  llvm.func @tensorlang_compile(!llvm.ptr) -> i32
  llvm.func @tensorlang_get_symbol_address(!llvm.ptr) -> !llvm.ptr
  llvm.func @printf(!llvm.ptr, ...) -> i32

  llvm.mlir.global internal constant @str_ir("Current IR: %s\0A\00")
  llvm.mlir.global internal constant @str_opt("Optimized IR: %s\0A\00")
  llvm.mlir.global internal constant @str_prompt("Optimize this code.\00")
  llvm.mlir.global internal constant @str_sym("main_optimized\00")
  llvm.mlir.global internal constant @str_exec("Executing optimized function...\0A\00")

  func.func @main() -> i32 {
    %fmt_ir = llvm.mlir.addressof @str_ir : !llvm.ptr
    %fmt_opt = llvm.mlir.addressof @str_opt : !llvm.ptr
    %prompt = llvm.mlir.addressof @str_prompt : !llvm.ptr
    %sym_name = llvm.mlir.addressof @str_sym : !llvm.ptr
    %str_exec_ptr = llvm.mlir.addressof @str_exec : !llvm.ptr

    // 1. Get IR
    %ir_ptr = llvm.call @tensorlang_get_ir() : () -> !llvm.ptr
    
    // 2. Query Model
    %new_ir_ptr = llvm.call @tensorlang_query_model(%prompt) : (!llvm.ptr) -> !llvm.ptr
    %1 = llvm.call @printf(%fmt_opt, %new_ir_ptr) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32

    // 3. Compile
    %res = llvm.call @tensorlang_compile(%new_ir_ptr) : (!llvm.ptr) -> i32
    
    // 4. Get Address of new function
    %fn_ptr = llvm.call @tensorlang_get_symbol_address(%sym_name) : (!llvm.ptr) -> !llvm.ptr
    
    // 5. Execute
    %2 = llvm.call @printf(%str_exec_ptr) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
    
    // Indirect call
    %ret = llvm.call %fn_ptr() : !llvm.ptr, () -> i32
    
    return %ret : i32
  }
}
