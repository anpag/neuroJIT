include_directories(
  "${CMAKE_SOURCE_DIR}/include"
  "${CMAKE_BINARY_DIR}/include"
)

add_subdirectory(IR)
add_subdirectory(Transforms)
