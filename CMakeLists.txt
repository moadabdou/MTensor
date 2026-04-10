
cmake_minimum_required(VERSION 3.31.6)

project(MTensor VERSION 2.0 LANGUAGES CXX)

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)

include(FetchContent)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/config)

include(CompilerFlags)
include(GoogletestConfig)
include(OpenMPConfig)

find_package(DNNL CONFIG REQUIRED)
find_package(ZLIB REQUIRED)
find_package(PNG REQUIRED)

add_subdirectory(src)
#add_subdirectory(tests)
#add_subdirectory(examples)





