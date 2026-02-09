find_package(CURL REQUIRED)

add_library(TensorLangRuntime STATIC
  Runtime.cpp
  JitContext.cpp
  ModelRunner.cpp
  GeminiModelRunner.cpp
)

target_include_directories(TensorLangRuntime PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_SOURCE_DIR}/include"
  "${CMAKE_BINARY_DIR}/include"
  ${CURL_INCLUDE_DIRS}
)

target_link_libraries(TensorLangRuntime PRIVATE
  TensorLangExecutionEngine
  CURL::libcurl
)
