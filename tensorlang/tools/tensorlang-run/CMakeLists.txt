add_executable(tensorlang-run
  tensorlang-run.cpp
)

target_link_libraries(tensorlang-run
  PRIVATE
  TensorLang
  TensorLangExecutionEngine
  TensorLangRuntime
  MLIRIR
  MLIRParser
  MLIRSupport
  
  # Bufferization Interface Impls
  MLIRArithTransforms
  MLIRLinalgTransforms
  MLIRTensorTransforms
  MLIRVectorTransforms
  MLIRSCFTransforms
  MLIRBufferizationTransforms
)

set_target_properties(tensorlang-run PROPERTIES ENABLE_EXPORTS 1)
