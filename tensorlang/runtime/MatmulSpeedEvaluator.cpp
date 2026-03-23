#include "tensorlang/Runtime/MatmulSpeedEvaluator.h"
#include <chrono>
#include <random>
#include <vector>
#include <memory>

namespace mlir {
namespace tensorlang {

float MatmulSpeedEvaluator::evaluate(void* functionPointer) {
  if (!functionPointer) return 0.0f;

  using MatmulFn = void (*)(MemRef2D*, MemRef2D*, MemRef2D*);
  MatmulFn matmul = reinterpret_cast<MatmulFn>(functionPointer);

  const int64_t N = 128;
  const size_t size = N * N;
  
  std::vector<float> dataA(size + 16);
  std::vector<float> dataB(size + 16);
  std::vector<float> dataC(size + 16, 0.0f);

  void* ptrA = dataA.data(); size_t spaceA = dataA.size() * sizeof(float);
  std::align(64, size * sizeof(float), ptrA, spaceA);
  
  void* ptrB = dataB.data(); size_t spaceB = dataB.size() * sizeof(float);
  std::align(64, size * sizeof(float), ptrB, spaceB);
  
  void* ptrC = dataC.data(); size_t spaceC = dataC.size() * sizeof(float);
  std::align(64, size * sizeof(float), ptrC, spaceC);

  float* aAlign = static_cast<float*>(ptrA);
  float* bAlign = static_cast<float*>(ptrB);
  float* cAlign = static_cast<float*>(ptrC);

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);

  for (size_t i = 0; i < size; ++i) {
    aAlign[i] = dis(gen);
    bAlign[i] = dis(gen);
    cAlign[i] = 0.0f;
  }

  MemRef2D A, B, C;
  A.allocated = A.aligned = aAlign; A.offset = 0;
  A.sizes[0] = N; A.sizes[1] = N; A.strides[0] = N; A.strides[1] = 1;

  B.allocated = B.aligned = bAlign; B.offset = 0;
  B.sizes[0] = N; B.sizes[1] = N; B.strides[0] = N; B.strides[1] = 1;

  C.allocated = C.aligned = cAlign; C.offset = 0;
  C.sizes[0] = N; C.sizes[1] = N; C.strides[0] = N; C.strides[1] = 1;

  // Warmup
  matmul(&A, &B, &C);

  auto start = std::chrono::high_resolution_clock::now();
  matmul(&A, &B, &C);
  auto end = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  float execTimeUs = static_cast<float>(duration.count());

  return 10000.0f / (execTimeUs + 1.0f);
}

} // namespace tensorlang
} // namespace mlir
