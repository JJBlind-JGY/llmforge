cmake_minimum_required(VERSION 3.24)

project(
    llmforge_cuda_kernels
    LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

add_executable(
    llmforge_vector_add
    vector_add.cu
)

target_compile_options(
    llmforge_vector_add
    PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

target_compile_definitions(
    llmforge_vector_add
    PRIVATE
    LLMFORGE_CUDA_ARCH="${CMAKE_CUDA_ARCHITECTURES}"
)
