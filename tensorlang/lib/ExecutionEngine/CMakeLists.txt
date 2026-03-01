add_library(TensorLangExecutionEngine
  JitRunner.cpp
)

target_include_directories(TensorLangExecutionEngine
  PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/../../../include"
  "${CMAKE_CURRENT_BINARY_DIR}/../../../include"
)

target_link_libraries(TensorLangExecutionEngine
  PUBLIC
  MLIRIR
  MLIRExecutionEngine
  MLIRLLVMToLLVMIRTranslation
  MLIRTargetLLVMIRExport
  LLVMOrcJIT
  TensorLangToLinalg
  TensorLangTransforms
  
  # Optimization & Conversion Libraries
  MLIRPass
  MLIRTransforms
  MLIRBufferizationTransforms
  MLIRFuncToLLVM
  MLIRMemRefToLLVM
  MLIRArithToLLVM
  MLIRVectorToSCF
  MLIRVectorToLLVM
  MLIRReconcileUnrealizedCasts
  
  # Dialects
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRTensorDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRSCFToControlFlow
  MLIRControlFlowToLLVM
  MLIRIndexToLLVM
  
  # Bufferization Interface Impls
  MLIRArithTransforms
  MLIRTensorTransforms
  MLIRVectorTransforms
  MLIRSCFTransforms
  MLIRBufferizationTransforms
  MLIRAffineTransforms
  MLIRAffineToStandard
)
