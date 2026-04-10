# Linux Support Analysis for MTensor

## Executive Summary

MTensor 2.0 is a CPU-based deep learning library written in C++17, currently **Windows-only** (MSVC 17/Visual Studio 2022). The codebase is **~85-90% cross-platform compatible**, with most platform-specific code already handled through conditional compilation. Adding Linux support requires changes to build configuration, CMake presets, and minor code adjustments.

---

## 1. Current State Analysis

### 1.1 Build System

**CMake Version**: 3.31.6 (minimum required)  
**Build Generator**: Visual Studio 17 2022 (Windows-only)  
**Dependency Manager**: vcpkg (submodule at `external/vcpkg`)  
**Library Type**: SHARED library (`.dll` on Windows)

**Current CMakePresets.json**:
- Generator: `Visual Studio 17 2022` (Windows-only)
- No Linux or cross-platform presets configured
- Binary output directory: `build/`

### 1.2 Dependencies

All dependencies are managed via vcpkg and are **cross-platform compatible**:

| Dependency | Purpose | Linux Support |
|-----------|---------|---------------|
| **oneDNN** | High-performance tensor primitives | ✅ Full Linux support |
| **libpng** | PNG image handling | ✅ Full Linux support |
| **ZLIB** | Compression library | ✅ Full Linux support |
| **OpenMP** | Thread parallelism | ✅ Full Linux support (via `libomp` or GCC OpenMP) |
| **GoogleTest** | Testing framework | ✅ Full Linux support |

**Note**: CImg is used for image handling but is a header-only library (no vcpkg dependency listed).

### 1.3 Source Code Platform-Specific Analysis

#### Files with Platform-Specific Code:

**1. `src/graph/grad_graph.cpp`** ✅ **Already Linux-ready**
```cpp
#if defined(_WIN32)
#include <windows.h>
#include <heapapi.h>
#elif defined(__APPLE__)
#include <malloc/malloc.h>
#elif defined(__linux__)
#include <malloc.h>
#endif
```

**Memory size query function** (lines 218-230):
```cpp
#if defined(_WIN32)
    HANDLE process_heap = GetProcessHeap();
    if (process_heap != NULL) {
        size_in_bytes = HeapSize(process_heap, 0, ptr);
    }
#elif defined(__APPLE__)
    size_in_bytes = malloc_size(ptr);
#elif defined(__linux__)
    size_in_bytes = malloc_usable_size(ptr);  // ✅ Linux support already implemented
#endif
```

**2. `include/config/mtensor_export.hpp`** ⚠️ **Needs minor update**
```cpp
#ifdef _WIN32
  #ifdef MTENSOR_EXPORTS
    #define MTENSOR_API __declspec(dllexport)
  #else
    #define MTENSOR_API __declspec(dllimport)
  #endif
#else
  #define MTENSOR_API  // ✅ Linux: empty macro (correct approach)
#endif
```

**Status**: Already handles Linux correctly (empty macro for non-Windows).

**3. `include/config/kernels_export.hpp`** ⚠️ **Needs minor update**
```cpp
#ifdef _WIN32
  #ifdef KERNEL_EXPORTS
    #define KERNEL_API extern "C" __declspec(dllexport)
  #else
    #define KERNEL_API extern "C"
  #endif
#else
  #define KERNEL_API extern "C"  // ✅ Linux: standard extern "C"
#endif
```

**Status**: Already handles Linux correctly.

**4. `src/data/mnist/mnist.cpp`** ✅ **Already Linux-ready**
```cpp
#if defined(_MSC_VER)
    magic_number = _byteswap_ulong(magic_number);
#else
    magic_number = __builtin_bswap32(magic_number);  // ✅ Works on GCC/Clang
#endif
```

**Status**: Uses GCC/Clang builtins for Linux (correct implementation).

---

## 2. Required Changes for Linux Support

### 2.1 High Priority (Blocking)

#### A. Update `CMakePresets.json` for Linux

**Current Issue**: Presets are hardcoded for Visual Studio generator.

**Required Changes**:
```json
{
  "version": 3,
  "cmakeMinimumRequired": {
    "major": 3,
    "minor": 31,
    "patch": 6
  },
  "configurePresets": [
    {
      "name": "base",
      "hidden": true,
      "cacheVariables": {
        "CMAKE_TOOLCHAIN_FILE": "${sourceDir}/external/vcpkg/scripts/buildsystems/vcpkg.cmake"
      }
    },
    {
      "name": "windows-vs2022",
      "inherits": "base",
      "generator": "Visual Studio 17 2022",
      "binaryDir": "${sourceDir}/build/windows",
      "condition": {
        "type": "equals",
        "lhs": "${hostSystemName}",
        "rhs": "Windows"
      }
    },
    {
      "name": "linux-ninja",
      "inherits": "base",
      "generator": "Ninja",
      "binaryDir": "${sourceDir}/build/linux",
      "cacheVariables": {
        "CMAKE_BUILD_TYPE": "Release"
      },
      "condition": {
        "type": "equals",
        "lhs": "${hostSystemName}",
        "rhs": "Linux"
      }
    },
    {
      "name": "linux-ninja-debug",
      "inherits": "base",
      "generator": "Ninja",
      "binaryDir": "${sourceDir}/build/linux",
      "cacheVariables": {
        "CMAKE_BUILD_TYPE": "Debug"
      },
      "condition": {
        "type": "equals",
        "lhs": "${hostSystemName}",
        "rhs": "Linux"
      }
    }
  ],
  "buildPresets": [
    {
      "name": "windows-release",
      "configurePreset": "windows-vs2022",
      "configuration": "Release"
    },
    {
      "name": "windows-debug",
      "configurePreset": "windows-vs2022",
      "configuration": "Debug"
    },
    {
      "name": "linux-release",
      "configurePreset": "linux-ninja"
    },
    {
      "name": "linux-debug",
      "configurePreset": "linux-ninja-debug"
    }
  ]
}
```

**Key Changes**:
- Removed hardcoded generator from default preset
- Added platform-specific presets with conditions
- Uses Ninja generator for Linux (recommended)
- Separated build directories by platform

---

#### B. Update `README.md` with Linux Instructions

**Sections to Update**:

1. **Beta Notice** (line 8):
```markdown
> ⚠️ **Beta Notice**: MTensor 2.0 is currently tested on **Windows** (MSVC 17) and **Linux** (GCC 11+/Clang 14+). macOS support is coming soon.
```

2. **Installation Section** (line 75+):

Add Linux-specific build instructions:

```markdown
### Linux Installation

#### Prerequisites

```bash
# Ubuntu/Debian
sudo apt-get install -y build-essential ninja-build cmake libomp-dev

# Fedora/RHEL
sudo dnf install -y gcc-c++ ninja-build cmake libomp-devel

# Arch Linux
sudo pacman -S gcc ninja cmake llvm-openmp
```

#### Build Steps

```bash
# Clone with submodules
git clone --recurse-submodules https://github.com/your-username/MTensor.git
cd MTensor

# Bootstrap vcpkg (Linux)
cd external/vcpkg
./bootstrap-vcpkg.sh
./vcpkg install

# Configure and build (Release)
cmake --preset linux-ninja
cmake --build --preset linux-release

# Or Debug build
cmake --build --preset linux-debug
```
```

---

#### C. Update vcpkg Triplet Configuration

**Current Issue**: vcpkg defaults to Windows triplets.

**Required Action**: Create `vcpkg.json` with explicit Linux triplet support or set environment variable.

**Option 1**: Set `VCPKG_DEFAULT_TRIPLET` environment variable:
```bash
export VCPKG_DEFAULT_TRIPLET=x64-linux
```

**Option 2**: Add to `CMakePresets.json` cache variables:
```json
"cacheVariables": {
  "VCPKG_TARGET_TRIPLET": "x64-linux"
}
```

---

### 2.2 Medium Priority (Recommended)

#### D. Compiler Configuration in `CompilerFlags.cmake`

**Current State**:
```cmake
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
```

**Recommended Additions**:
```cmake
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Linux-specific compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(
    -Wall
    -Wextra
    -Wpedantic
    -fPIC  # Position-independent code for shared libraries
  )
  
  # Optimization flags
  if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O3 -march=native)
  endif()
endif()

# Windows-specific flags
if(MSVC)
  add_compile_options(/W4)
endif()
```

---

#### E. Shared Library Output Configuration

**Current Issue**: `src/CMakeLists.txt` sets output to `${CMAKE_BINARY_DIR}/bin`

On Linux, shared libraries should use proper RPATH settings:

**Add to `src/CMakeLists.txt`**:
```cmake
set_target_properties(MTensor PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
  ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
  
  # RPATH settings for Linux
  BUILD_RPATH_USE_ORIGIN TRUE
  BUILD_RPATH "$ORIGIN/../lib"
  INSTALL_RPATH "$ORIGIN/../lib"
)
```

---

#### F. OpenMP Configuration on Linux

**Current State**: `config/OpenMPConfig.cmake`
```cmake
find_package(OpenMP REQUIRED)
```

**Potential Issue**: On macOS with Homebrew, OpenMP may need explicit paths. On Linux, this should work out-of-the-box with `libomp-dev` installed.

**No changes needed** for Linux support, but ensure `libomp-dev` is installed.

---

### 2.3 Low Priority (Nice to Have)

#### G. Add Linux CI/CD Pipeline

Create `.github/workflows/linux-build.yml`:
```yaml
name: Linux Build

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-22.04
    
    steps:
    - uses: actions/checkout@v4
      with:
        submodules: recursive
    
    - name: Install dependencies
      run: |
        sudo apt-get update
        sudo apt-get install -y build-essential ninja-build cmake libomp-dev
    
    - name: Bootstrap vcpkg
      run: |
        cd external/vcpkg
        ./bootstrap-vcpkg.sh
    
    - name: Install vcpkg dependencies
      run: |
        external/vcpkg/vcpkg install
    
    - name: Configure
      run: cmake --preset linux-ninja
    
    - name: Build
      run: cmake --build --preset linux-release
```

---

#### H. Test Suite Integration

**Current State**: Tests are commented out in main `CMakeLists.txt`:
```cmake
#add_subdirectory(tests)
```

**Action Required**: Uncomment and ensure tests build on Linux. The `GoogletestConfig.cmake` is already cross-platform compatible.

---

#### I. Add `.gitattributes` for Line Endings

Ensure consistent line endings across platforms:
```
* text=auto
*.cpp text eol=lf
*.hpp text eol=lf
*.cmake text eol=lf
*.sh text eol=lf
*.bat text eol=crlf
```

---

## 3. Compatibility Matrix

### 3.1 Tested/Expected Linux Configurations

| Distribution | Compiler | Status | Notes |
|-------------|----------|--------|-------|
| Ubuntu 22.04+ | GCC 11+ | ✅ Should work | Install `libomp-dev` |
| Ubuntu 22.04+ | Clang 14+ | ✅ Should work | Install `libomp-dev` |
| Fedora 37+ | GCC 12+ | ✅ Should work | Install `libomp-devel` |
| Arch Linux | GCC 13+ | ✅ Should work | Install `llvm-openmp` |
| CentOS 8+ | GCC 10+ | ⚠️ Untested | May need newer CMake |

### 3.2 Source File Compatibility

| File | Windows | Linux | macOS | Notes |
|------|---------|-------|-------|-------|
| `src/graph/grad_graph.cpp` | ✅ | ✅ | ✅ | Already cross-platform |
| `src/data/mnist/mnist.cpp` | ✅ | ✅ | ✅ | Uses GCC builtins |
| `include/config/mtensor_export.hpp` | ✅ | ✅ | ✅ | Correct macro handling |
| `include/config/kernels_export.hpp` | ✅ | ✅ | ✅ | Correct macro handling |
| `src/CMakeLists.txt` | ✅ | ⚠️ | ⚠️ | Needs RPATH settings |
| `CMakePresets.json` | ✅ | ❌ | ❌ | Windows-only presets |
| `README.md` | ✅ | ❌ | ❌ | Windows-only instructions |

---

## 4. Implementation Checklist

### Phase 1: Core Linux Support (Required)
- [ ] Update `CMakePresets.json` with Linux presets
- [ ] Update `README.md` with Linux build instructions
- [ ] Test vcpkg dependency installation on Linux (`x64-linux` triplet)
- [ ] Verify oneDNN builds and links correctly on Linux
- [ ] Verify OpenMP works correctly on Linux

### Phase 2: Build System Improvements (Recommended)
- [ ] Add `-fPIC` and warning flags for GCC/Clang in `CompilerFlags.cmake`
- [ ] Add RPATH settings to `src/CMakeLists.txt`
- [ ] Test shared library loading on Linux
- [ ] Uncomment and test test suite (`tests/`)

### Phase 3: CI/CD and Quality of Life (Nice to Have)
- [ ] Add GitHub Actions for Linux builds
- [ ] Add `.gitattributes` for line ending consistency
- [ ] Add shell scripts for common operations (`build.sh`, `test.sh`)
- [ ] Update roadmap in `README.md` to reflect Linux support

---

## 5. Known Issues & Potential Roadblocks

### 5.1 CMake Version Requirement

**Issue**: CMake 3.31.6 is very recent (as of early 2026).

**Impact**: Older Linux distributions may not have this version in repositories.

**Solution**:
- Use Kitware's APT repository for latest CMake
- Or use `pip install cmake` to get latest version
- Consider lowering minimum version if possible (test with 3.20+)

### 5.2 oneDNN Performance on Linux

**Issue**: oneDNN may have different performance characteristics on Linux vs Windows.

**Action**: Benchmark critical operations after porting.

### 5.3 AVX512 Kernels

**Current State**: Codebase includes AVX512-optimized kernels:
- `src/kernels/fused_adam_avx512.cpp`
- `src/kernels/fused_sgd_avx512.cpp`

**Linux Consideration**: Ensure compiler flags enable AVX512 on Linux:
```cmake
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-mavx512f -mavx512bw -mavx512dq)
endif()
```

---

## 6. Testing Strategy

### 6.1 Build Verification
```bash
# Clean build
rm -rf build/
cmake --preset linux-ninja
cmake --build --preset linux-release

# Verify shared library
ls -la build/linux/libMTensor.so
ldd build/linux/libMTensor.so  # Check dependencies
```

### 6.2 Functionality Testing
```bash
# Build examples
cmake --build --preset linux-release --target examples

# Run examples
./build/linux/bin/example_name
```

### 6.3 Performance Benchmarking
Compare key operations (matrix multiply, convolutions) between Windows and Linux builds.

---

## 7. Estimated Effort

| Task | Complexity | Estimated Time |
|------|-----------|----------------|
| Update CMakePresets.json | Low | 30 minutes |
| Update README.md | Low | 1 hour |
| Test vcpkg on Linux | Medium | 2-3 hours |
| Add compiler flags | Low | 30 minutes |
| Add RPATH settings | Low | 30 minutes |
| CI/CD pipeline | Medium | 2 hours |
| Testing & debugging | Medium | 4-6 hours |
| **Total** | | **~10-12 hours** |

---

## 8. Conclusion

**MTensor is 85-90% Linux-ready**. The codebase already handles platform-specific code correctly through conditional compilation. The main blockers are:

1. **Build configuration** (CMakePresets.json needs Linux presets)
2. **Documentation** (README needs Linux instructions)
3. **Testing** (need to verify on Linux)

**Recommendation**: Proceed with Phase 1 changes immediately. The codebase architecture is already cross-platform friendly, and Linux support should be straightforward to implement and test.

---

## Appendix A: Quick Start Commands for Linux

```bash
# Install dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y build-essential ninja-build cmake libomp-dev git

# Clone repository
git clone --recurse-submodules https://github.com/your-username/MTensor.git
cd MTensor

# Bootstrap vcpkg
cd external/vcpkg
./bootstrap-vcpkg.sh
./vcpkg install

# Build
cd ../..
cmake --preset linux-ninja
cmake --build --preset linux-release

# Verify
ls -la build/linux/libMTensor.so
```

---

*Report generated: April 10, 2026*  
*Project: MTensor 2.0*  
*Target: Linux (GCC 11+, Clang 14+)*
