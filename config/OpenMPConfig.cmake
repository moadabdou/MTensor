
message(STATUS "Including OpenMPConfig.cmake ")

# OpenMP is already configured in CompilerFlags.cmake for GCC/Clang
# This file is kept for compatibility and future OpenMP-specific settings
if(NOT OpenMP_FOUND)
  find_package(OpenMP REQUIRED)
endif()

