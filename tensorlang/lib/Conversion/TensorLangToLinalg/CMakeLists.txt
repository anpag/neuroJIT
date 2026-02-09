add_library(TensorLangToLinalg
  TensorLangToLinalg.cpp
)

target_include_directories(TensorLangToLinalg
  PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/../../../../include"
  "${CMAKE_CURRENT_BINARY_DIR}/../../../../include"
)

target_link_libraries(TensorLangToLinalg
  PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRFuncTransforms
  TensorLangIR
  MLIRLinalgDialect
  MLIRTensorDialect
  MLIRArithDialect
)
