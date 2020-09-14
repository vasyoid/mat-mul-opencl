cmake_minimum_required(VERSION 3.10)
project(matmul)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCL REQUIRED)
message(${OpenCL_INCLUDE_DIRS})
include_directories(${OpenCL_INCLUDE_DIRS})
include_directories(include)
link_directories(${OpenCL_LIBRARY})
add_executable(matmul src/main.cpp src/matmul.cpp src/test.cpp matmul.cl)
target_link_libraries (matmul ${OpenCL_LIBRARY})

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
