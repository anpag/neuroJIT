add_subdirectory(Dialect)

add_library(TensorLang INTERFACE)

target_include_directories(TensorLang INTERFACE
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_BINARY_DIR}"
  "${CMAKE_CURRENT_BINARY_DIR}/Dialect/TensorLang/IR"
)

target_link_libraries(TensorLang INTERFACE
  MLIRIR
  TensorLangIR
)

add_subdirectory(Conversion)
add_subdirectory(ExecutionEngine)
