#ifndef TENSORLANG_RUNTIME_MATMULSPEEDEVALUATOR_H
#define TENSORLANG_RUNTIME_MATMULSPEEDEVALUATOR_H

#include "tensorlang/Runtime/VerificationSandbox.h"
#include <cstdint>

namespace mlir {
namespace tensorlang {

/**
 * @brief IEvaluator implementation for measuring Matrix Multiplication performance.
 */
class MatmulSpeedEvaluator : public IEvaluator {
public:
  MatmulSpeedEvaluator() = default;
  virtual ~MatmulSpeedEvaluator() = default;

  float evaluate(void* functionPointer) override;

  // MLIR MemRef Descriptor for 2D float tensors
  struct MemRef2D {
    float* allocated;
    float* aligned;
    intptr_t offset;
    intptr_t sizes[2];
    intptr_t strides[2];
  };
};

} // namespace tensorlang
} // namespace mlir

#endif // TENSORLANG_RUNTIME_MATMULSPEEDEVALUATOR_H
