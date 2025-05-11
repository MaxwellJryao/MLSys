cmake_minimum_required(VERSION 3.10)

project(MultiDirs)

set(CMAKE_CXX_STANDARD 14)

aux_source_directory(. SRC_LIST)

add_subdirectory(math)

add_executable(MultiDirs ${SRC_LIST})

# 设置可执行文件的输出目录

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

target_link_libraries(MultiDirs MathFunctions)