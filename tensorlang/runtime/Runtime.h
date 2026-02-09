#ifndef TENSORLANG_RUNTIME_RUNTIME_H
#define TENSORLANG_RUNTIME_RUNTIME_H

#include <stdint.h>

extern "C" {

//===----------------------------------------------------------------------===//
// I/O Operations
//===----------------------------------------------------------------------===//

/// Prints a float tensor (for debugging).
void tensorlang_print_f32(float* data, int64_t rank, int64_t* shape);

//===----------------------------------------------------------------------===//
// Reflection & JIT Operations
//===----------------------------------------------------------------------===//

/// Returns the current function's IR as a null-terminated string.
/// The caller is responsible for freeing the string.
char* tensorlang_get_ir();

/// Compiles and hot-swaps the current function with the new IR provided.
/// Returns 0 on success, non-zero on error.
int tensorlang_compile(const char* ir_string);

/// Returns the address of a symbol in the JIT.
void* tensorlang_get_symbol_address(const char* name);

} // extern "C"

#endif // TENSORLANG_RUNTIME_RUNTIME_H
