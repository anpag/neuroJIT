add_library(TensorLangRuntime STATIC
  Runtime.cpp
  JitContext.cpp
  ModelRunner.cpp
)

target_include_directories(TensorLangRuntime PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_SOURCE_DIR}/include"
  "${CMAKE_BINARY_DIR}/include"
)

target_link_libraries(TensorLangRuntime PRIVATE
  TensorLangExecutionEngine
)
