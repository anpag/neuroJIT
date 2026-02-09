add_library(TensorLangIR STATIC
  TensorLangOps.cpp
  TensorLangDialect.cpp
)

target_include_directories(TensorLangIR PUBLIC
  "${CMAKE_SOURCE_DIR}/include"
  "${CMAKE_BINARY_DIR}/include"
  "${CMAKE_CURRENT_BINARY_DIR}"
)

target_link_libraries(TensorLangIR PUBLIC
  MLIRIR
  MLIRSideEffectInterfaces
)

# TableGen definitions
configure_file(
  ${CMAKE_SOURCE_DIR}/include/tensorlang/Dialect/TensorLang/IR/TensorLang.td
  ${CMAKE_CURRENT_SOURCE_DIR}/TensorLang.td
  COPYONLY
)

add_dependencies(TensorLangIR MLIRTensorLangIncGen)

add_mlir_dialect(TensorLang
  tensorlang
  -I ${CMAKE_SOURCE_DIR}/include
  -I ${CMAKE_BINARY_DIR}/include
)
