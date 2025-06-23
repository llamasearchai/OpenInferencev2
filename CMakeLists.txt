7. CMakeLists.txt
cmake_minimum_required(VERSION 3.18)
project(OpenInferencev2 LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
# Find packages
find_package(CUDA REQUIRED)
find_package(PkgConfig REQUIRED)
# CUDA settings
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 87 89 90)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -use_fast_math --expt-relaxed_constexpr")
# Include directories
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(src/core)
# Source files
set(SOURCES
    src/core/inference_engine.cpp
    src/core/gpu_manager.cu
    src/core/kv_cache.cu
    src/core/custom_kernels.cu
    src/core/distributed_manager.cpp
)
# Create shared library
add_library(openinferencev2_core SHARED ${SOURCES})
# Link libraries
target_link_libraries(openinferencev2_core 
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
    ${CUDA_CURAND_LIBRARIES}
    ${CUDA_CUSPARSE_LIBRARIES}
    nccl
    mpi
)
# Python bindings
find_package(pybind11 REQUIRED)
pybind11_add_module(openinferencev2_cpp src/core/python_bindings.cpp)
target_link_libraries(openinferencev2_cpp PRIVATE openinferencev2_core)
# Installation
install(TARGETS openinferencev2_core DESTINATION lib)
install(TARGETS openinferencev2_cpp DESTINATION .)