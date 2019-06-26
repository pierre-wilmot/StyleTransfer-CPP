
hunter_add_package(SDL2)
find_package(SDL2 CONFIG REQUIRED)

add_executable(StyleTransferGUI main.cpp)
target_link_libraries(StyleTransferGUI "${TORCH_LIBRARIES}" SDL2::SDL2 SDL2::SDL2main)
