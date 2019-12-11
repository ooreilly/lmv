cmake_minimum_required(VERSION 3.10)
project(LMV VERSION 1.0 LANGUAGES CXX CUDA)
 
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
if (DEFINED ENV{ARCH})
        set(ARCH $ENV{ARCH})
else()
        set(ARCH sm_75)
endif()
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=${ARCH} -Xptxas=-v -lineinfo -use_fast_math -lcublas -lcusparse")

set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -D${ARCH}")

include(CTest)
enable_testing()

add_subdirectory(src)
add_subdirectory(tests)

