add_executable(StyleTransferCLI main.cpp)
set_target_properties(StyleTransferCLI PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
target_link_libraries(StyleTransferCLI "${TORCH_LIBRARIES}")
