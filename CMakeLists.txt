
project(StyleTransfer)
cmake_minimum_required(VERSION 3.9)
set(CMAKE_CXX_STANDARD 11)

set(Torch_DIR ${PROJECT_SOURCE_DIR}/lib/libtorch/share/cmake/Torch/)
find_package(Torch REQUIRED)

include_directories(include)
add_subdirectory(src)

enable_testing()
add_subdirectory(test)
