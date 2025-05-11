cmake_minimum_required(VERSION 3.10)

project(Demo1)

set(CMAKE_CXX_STANDARD 14)

add_executable(Demo test.cpp)

# 设置可执行文件的输出目录

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

# 设置库文件的输出目录

set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

