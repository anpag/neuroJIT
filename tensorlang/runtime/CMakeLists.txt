add_library(TensorLangRuntime STATIC
  Runtime.cpp
  JitContext.cpp
  ModelRunner.cpp
  GeminiModelRunner.cpp
  LlamaCppModelRunner.cpp
  StrategyCache.cpp
)

# Set absolute paths to pre-built llama libraries
set(LLAMA_LIB "${CMAKE_SOURCE_DIR}/deps/llama.cpp/build/bin/libllama.so")
set(LLAMA_COMMON_LIB "${CMAKE_SOURCE_DIR}/deps/llama.cpp/build/common/libcommon.a")
set(GGML_LIB "${CMAKE_SOURCE_DIR}/deps/llama.cpp/build/bin/libggml.so")
set(GGML_CPU_LIB "${CMAKE_SOURCE_DIR}/deps/llama.cpp/build/bin/libggml-cpu.so")
set(GGML_BASE_LIB "${CMAKE_SOURCE_DIR}/deps/llama.cpp/build/bin/libggml-base.so")

target_include_directories(TensorLangRuntime PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_SOURCE_DIR}/include"
  "${CMAKE_SOURCE_DIR}/deps"
  "${CMAKE_SOURCE_DIR}/deps/llama.cpp/include"
  "${CMAKE_SOURCE_DIR}/deps/llama.cpp/common"
  "${CMAKE_SOURCE_DIR}/deps/llama.cpp/ggml/include"
  "${CMAKE_SOURCE_DIR}/deps/llama.cpp/vendor"
  "${CMAKE_BINARY_DIR}/include"
)

target_link_libraries(TensorLangRuntime PRIVATE
  TensorLangExecutionEngine
  ${LLAMA_LIB}
  ${LLAMA_COMMON_LIB}
  ${GGML_LIB}
  ${GGML_CPU_LIB}
  ${GGML_BASE_LIB}
)
