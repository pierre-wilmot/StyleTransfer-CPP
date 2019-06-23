set(Torch_DIR ${PROJECT_SOURCE_DIR}/lib/libtorch/share/cmake/Torch/)
find_package(Torch REQUIRED)
add_executable(StyleTransferCLI main.cpp)
target_link_libraries(StyleTransferCLI "${TORCH_LIBRARIES}")
