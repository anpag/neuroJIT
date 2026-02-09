#ifndef TENSORLANG_EXECUTIONENGINE_JITRUNNER_H
#define TENSORLANG_EXECUTIONENGINE_JITRUNNER_H

#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/Error.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"

#include <memory>
#include <string>

namespace mlir {
namespace tensorlang {

/// A JIT runner for executing TensorLang modules.
/// This class handles the lowering of TensorLang IR to LLVM IR and executes it
/// using LLVM's ORC JIT.
class JitRunner {
public:
  /// Creates a new JIT runner.
  /// Returns an error if initialization fails.
  static llvm::Expected<std::unique_ptr<JitRunner>> create();

  /// Destructor.
  ~JitRunner();

  /// Compiles and executes the given MLIR module.
  /// The module is lowered to LLVM IR, optimized, and then added to the JIT.
  /// This function looks for a 'main' function and executes it.
  /// Returns an error if compilation or execution fails.
  llvm::Error run(ModuleOp module);

  /// Invokes a specific function by name.
  /// The function must have been previously compiled via `run` or `compile`.
  /// Returns the exit code of the function, or an error.
  llvm::Expected<int> invoke(llvm::StringRef functionName);

  /// Looks up a symbol and returns its address.
  llvm::Expected<void*> lookup(llvm::StringRef name);

  /// Compiles the module without executing it immediately.
  /// Useful for pre-compilation or library loading.
  llvm::Error compile(ModuleOp module);

  /// Compiles the MLIR source code string.
  /// This parses the MLIR, optimizes it, and adds it to the JIT.
  llvm::Error compileString(llvm::StringRef source);

private:
  JitRunner(std::unique_ptr<llvm::orc::LLJIT> jit);

  std::unique_ptr<llvm::orc::LLJIT> jit;
};

} // namespace tensorlang
} // namespace mlir

#endif // TENSORLANG_EXECUTIONENGINE_JITRUNNER_H
