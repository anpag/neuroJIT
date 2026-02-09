add_executable(tensorlang-opt
  main.cpp
)

target_link_libraries(tensorlang-opt PRIVATE
  TensorLang
  TensorLangToLinalg
  TensorLangTransforms
  
  # MLIR Core
  MLIRIR
  MLIRParser
  MLIRSupport
  MLIROptLib
  MLIRPass
  
  # Dialects
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRTensorDialect
  
  # Transforms and Interface Impls
  MLIRArithTransforms
  MLIRLinalgTransforms
  MLIRTensorTransforms
  MLIRVectorTransforms
  MLIRSCFTransforms
  MLIRFuncTransforms
  MLIRBufferizationTransforms
  MLIRMemRefTransforms
  
  # Needed for MlirOptMain
  MLIRSupport
)

add_subdirectory(tensorlang-run)
