cmake_minimum_required(VERSION 3.31.6)
project(LeNet5_Example LANGUAGES CXX)

# Add the LeNet-5 executable
add_executable(lenet5 lenet5.cpp)

# Link against MTensor
target_link_libraries(lenet5 PRIVATE MTensor)

# Use C++17
set_target_properties(lenet5 PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Set output directory
set_target_properties(lenet5 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/examples
)
