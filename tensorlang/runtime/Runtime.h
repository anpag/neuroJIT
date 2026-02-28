#ifndef TENSORLANG_RUNTIME_RUNTIME_H
#define TENSORLANG_RUNTIME_RUNTIME_H

#include <stdint.h>

extern "C" {

//===----------------------------------------------------------------------===//
// I/O Operations
//===----------------------------------------------------------------------===//

/// Prints a float tensor (for debugging).
void tensorlang_print_f32(float* data, int64_t rank, int64_t* shape);

/// Prints the lander's status (Alt and Vel).
void tensorlang_print_status(float h, float v);

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

/// Called when a tensorlang.assert fails.
void tensorlang_assert_fail(int64_t loc);

/// Starts a high-resolution timer.
void tensorlang_start_timer();

/// Stops the timer and records telemetry (impact velocity, latency).
void tensorlang_stop_timer(float final_v);

/// Triggers background optimization.
void tensorlang_optimize_async(const char* prompt, const char* target_name);

} // extern "C"

#endif // TENSORLANG_RUNTIME_RUNTIME_H
