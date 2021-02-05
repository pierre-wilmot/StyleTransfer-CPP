
hunter_add_package(SDL2)
find_package(SDL2 CONFIG REQUIRED)

add_executable(StyleTransferGUI main.cpp)
set_target_properties(StyleTransferGUI PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
target_link_libraries(StyleTransferGUI "${TORCH_LIBRARIES}" SDL2::SDL2 SDL2::SDL2main)
