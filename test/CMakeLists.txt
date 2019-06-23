add_executable (StyleTransferTest StyleTransfer.cpp)
target_link_libraries(StyleTransferTest "${TORCH_LIBRARIES}")
add_test(StyleTransferTest StyleTransferTest)
