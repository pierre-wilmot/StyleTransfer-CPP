### Hunter snapshot that will be used ###
include("cmake/HunterGate.cmake")
HunterGate(
    URL "https://github.com/ruslo/hunter/archive/v0.23.194.tar.gz"
    SHA1 "6b722085f51b039e883ecb3d233390a93ed29dac"
)

project(StyleTransfer)
cmake_minimum_required(VERSION 3.9)
set(CMAKE_CXX_STANDARD 17)

set(Torch_DIR ${PROJECT_SOURCE_DIR}/lib/libtorch/share/cmake/Torch/)
find_package(Torch REQUIRED)

include_directories(include)
add_subdirectory(src)

enable_testing()
add_subdirectory(test)
