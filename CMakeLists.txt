cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
cmake_policy(VERSION 3.18)

include(external/madrona/cmake/build_type.cmake)

project(MadronaGPURearrange LANGUAGES C CXX)

include(external/madrona/cmake/setup.cmake)

add_subdirectory(external)
add_subdirectory(src)
