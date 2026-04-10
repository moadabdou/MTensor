
message(STATUS "Including CompilerFlags.cmake - setting compiler options")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Linux/GCC/Clang compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
  
  # Position-independent code for shared libraries
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  
  # Enable AVX512 for fused kernels (if supported)
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512)
  if(COMPILER_SUPPORTS_AVX512)
    add_compile_options(-mavx512f -mavx512bw -mavx512dq -mavx512vl)
  endif()
  
  # OpenMP support for GCC/Clang
  find_package(OpenMP REQUIRED)
endif()

# MSVC-specific flags
if(MSVC)
  add_compile_options(/W4)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
endif()
