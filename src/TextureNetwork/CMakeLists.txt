add_executable(TextureNetwork main.cpp)
set_target_properties(TextureNetwork PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
target_link_libraries(TextureNetwork "${TORCH_LIBRARIES}")
