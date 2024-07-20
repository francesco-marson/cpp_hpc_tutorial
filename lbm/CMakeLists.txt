cmake_minimum_required(VERSION 3.18)
project(lbm LANGUAGES CXX)

# Find MPI
find_package(MPI REQUIRED)

# Set compiler and flags
set(CMAKE_CXX_COMPILER nvc++)
message("Using compiler: ${CMAKE_CXX_COMPILER}")

set(CMAKE_CXX_FLAGS "-Ofast -march=native -std=c++20 -stdpar=gpu -cudalib=cublas -gpu=managed,fastmath,fma --experimental-stdpar")


# Add executable
add_executable(lbm lbm.cpp)

# Link MPI
target_include_directories(lbm PRIVATE ${MPI_CXX_INCLUDE_PATH})
target_link_libraries(lbm PRIVATE MPI::MPI_CXX)
