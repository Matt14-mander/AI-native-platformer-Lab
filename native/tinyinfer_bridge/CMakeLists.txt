cmake_minimum_required(VERSION 3.20)
project(PlatformerTinyInferBridge LANGUAGES CXX)

# Some Command Line Tools installations keep libc++ headers only in the SDK.
# This affects TinyInfer's compilation too, so apply the SDK include directory
# before adding its target, without changing the dependency checkout.
if(APPLE)
    execute_process(COMMAND xcrun --show-sdk-path OUTPUT_VARIABLE PLATFORMER_SDK
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(EXISTS "${PLATFORMER_SDK}/usr/include/c++/v1/cstddef")
        include_directories(SYSTEM "${PLATFORMER_SDK}/usr/include/c++/v1")
    endif()
endif()

set(TINYINFER_SOURCE_DIR "" CACHE PATH "Path to the TinyInfer source checkout")
if(NOT EXISTS "${TINYINFER_SOURCE_DIR}/include/tinyinfer/tinyinfer.h")
    message(FATAL_ERROR "Set TINYINFER_SOURCE_DIR to a TinyInfer checkout")
endif()
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Link TinyInfer statically into the bridge" FORCE)
set(TINYINFER_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(TINYINFER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(TINYINFER_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
add_subdirectory("${TINYINFER_SOURCE_DIR}" tinyinfer)
add_library(platformer_tinyinfer SHARED bridge.cpp)
target_compile_features(platformer_tinyinfer PRIVATE cxx_std_17)
target_link_libraries(platformer_tinyinfer PRIVATE TinyInfer::tinyinfer)
set_target_properties(platformer_tinyinfer PROPERTIES
    CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN YES)
