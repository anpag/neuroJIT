add_library(TensorLangTransforms
  VerifyLinearity.cpp
)

target_include_directories(TensorLangTransforms PUBLIC
  "${CMAKE_SOURCE_DIR}/include"
  "${CMAKE_BINARY_DIR}/include"
  "${CMAKE_CURRENT_BINARY_DIR}"
)

target_link_libraries(TensorLangTransforms PUBLIC
  MLIRIR
  MLIRPass
  MLIRFuncDialect
  MLIRSCFDialect
  TensorLangIR
)

set(LLVM_TARGET_DEFINITIONS ${CMAKE_SOURCE_DIR}/include/tensorlang/Dialect/TensorLang/Transforms/Passes.td)

mlir_tablegen(Passes.h.inc -gen-pass-decls)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl)
add_public_tablegen_target(TensorLangTransformsPassesIncGen)

add_dependencies(TensorLangTransforms TensorLangTransformsPassesIncGen)
