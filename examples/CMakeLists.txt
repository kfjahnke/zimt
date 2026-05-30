# CMakeLists.txt for zimt examples

cmake_minimum_required(VERSION 3.16)
project(ZimtExamples CXX)

# Enforce C++17 with GNU extensions
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

# --- Pixi / Conda Environment Integration ---
# When building inside a Pixi/Conda shell, we want to prioritize libraries 
# located within the environment prefix over system-wide installations.
# --- Pixi / Conda Environment Integration ---
if(DEFINED ENV{CONDA_PREFIX})
    message(STATUS "Pixi/Conda environment detected: $ENV{CONDA_PREFIX}")
    
    set(CMAKE_FIND_ROOT_PATH $ENV{CONDA_PREFIX})
    
    # Change 'ONLY' to 'BOTH' to allow fallback to macOS System Frameworks
    set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE BOTH)
    set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY BOTH)
    set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE BOTH)
    
    if(APPLE)
        # Prioritize Frameworks so it finds the native OpenGL 
        set(CMAKE_FIND_FRAMEWORK FIRST)
    endif()
endif()

# Add the parent directory to the include search path
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/..")

# Look for packages natively supported by modern CMake targets
find_package(Threads REQUIRED)
find_package(OpenImageIO CONFIG REQUIRED)
find_package(Vc REQUIRED)
find_package(HWY CONFIG REQUIRED)
find_package(ZLIB REQUIRED)
find_package(OpenEXR CONFIG REQUIRED)
find_package(Vigra CONFIG REQUIRED)
if(HWY_FOUND)
  message(STATUS "Highway found: HWY_FOUND=TRUE")
else()
  message(STATUS "Highway not found")
endif()

# Hard-force -pthread compiler flag on Unix-like systems.
# This bypasses the fragile CMake thread-detection which often fails 
# with Clang inside Conda environments.
if(UNIX)
  add_compile_options(-pthread)
  add_link_options(-pthread) # can be omitted on macOS
endif()

# Define the common SIMD flags
set(COMMON_SIMD_FLAGS
  "-O3"
  "-msse2"
  "-mssse3"
  "-msse4.1"
  "-msse4.2"
  "-mpclmul"
  "-maes"
  "-mavx"
  "-mavx2"
  "-mbmi"
  "-mbmi2"
  "-mfma"
  "-mf16c"
)

# Base flags for multi-SIMD-ISA compilation
set(MSA_FLAGS "-O3")

# Find all .cc files in the current directory
file(GLOB EXAMPLES "*.cc")

foreach(SOURCE_FILE IN LISTS EXAMPLES)
  # Initialize link libraries with the abstract Threads target
  # This ensures -pthread is correctly appended for both compiler and linker stages
  set(LINK_LIBS Threads::Threads)

  get_filename_component(BASE_NAME "${SOURCE_FILE}" NAME_WE)
  file(READ "${SOURCE_FILE}" FILE_CONTENT)

  # 1. Determine linking requirements based on file content.
  set(LINK_LIBS "")
  if(FILE_CONTENT MATCHES "OpenImageIO")
    if(OpenImageIO_FOUND)
      # Modern CMake target: automatically handles dependencies like internal fmt paths
      list(APPEND LINK_LIBS OpenImageIO::OpenImageIO)
    else()
      # Fallback to raw names if find_package didn't resolve
      list(APPEND LINK_LIBS "OpenImageIO" "OpenImageIO_Util")
    endif()
  endif()
  
  if(FILE_CONTENT MATCHES "vigra")
    list(APPEND LINK_LIBS "vigraimpex")
  endif()

  # 2. Check if the std::simd backend should be skipped.
  # getting std::simd requires the use of the gnu c++ library, which would
  # have to be enforced explicitly on macOS. It can be done, but I omit
  # it for now.
  if(APPLE)
    set(BUILD_STD_SIMD FALSE)
  else()
    set(BUILD_STD_SIMD TRUE)
  endif()
  if(FILE_CONTENT MATCHES "no-std-simd")
    set(BUILD_STD_SIMD FALSE)
  endif()

  # 3. Check if we need to build multi-SIMD-ISA variants.
  set(BUILD_MULTI_SIMD FALSE)
  if(FILE_CONTENT MATCHES "dispatch_base")
    set(BUILD_MULTI_SIMD TRUE)
  endif()

  # -------------------------------------------------------------------
  # Target Generation
  # -------------------------------------------------------------------

  # --- Target: Goading (gd_) ---
  set(TARGET_GD "gd_${BASE_NAME}")
  if(NOT FILE_CONTENT MATCHES "vigra")
    add_executable("${TARGET_GD}" "${SOURCE_FILE}")
    target_compile_options("${TARGET_GD}" PRIVATE ${COMMON_SIMD_FLAGS})
    if(LINK_LIBS)
      target_link_libraries("${TARGET_GD}" PRIVATE ${LINK_LIBS})
    endif()
  endif()

  # --- Target: Vc (vc_) ---
  if(Vc_FOUND)
    set(TARGET_VC "vc_${BASE_NAME}")
    add_executable("${TARGET_VC}" "${SOURCE_FILE}")
    target_compile_definitions("${TARGET_VC}" PRIVATE USE_VC)
    target_compile_options("${TARGET_VC}" PRIVATE ${COMMON_SIMD_FLAGS})
    target_link_libraries("${TARGET_VC}" PRIVATE Vc::Vc)
    if(LINK_LIBS)
      target_link_libraries("${TARGET_VC}" PRIVATE ${LINK_LIBS})
    endif()
    target_include_directories("${TARGET_VC}" PRIVATE ${Vc_INCLUDE_DIR})
  endif()

  # --- Target: Highway (hwy_) ---
  set(TARGET_HWY "hwy_${BASE_NAME}")
  add_executable("${TARGET_HWY}" "${SOURCE_FILE}")
  target_link_libraries("${TARGET_HWY}" PRIVATE hwy::hwy)
  target_include_directories("${TARGET_HWY}" PRIVATE ${hwy_INCLUDE_DIR})
  target_compile_definitions("${TARGET_HWY}" PRIVATE USE_HWY)
  target_compile_options("${TARGET_HWY}" PRIVATE ${COMMON_SIMD_FLAGS})
  if(LINK_LIBS)
    target_link_libraries("${TARGET_HWY}" PRIVATE ${LINK_LIBS})
  endif()

  # --- Target: std::simd (stds_) ---
  if(BUILD_STD_SIMD)
    set(TARGET_STDS "stds_${BASE_NAME}")
    add_executable("${TARGET_STDS}" "${SOURCE_FILE}")
    target_compile_definitions("${TARGET_STDS}" PRIVATE USE_STDSIMD)
    target_compile_options("${TARGET_STDS}" PRIVATE ${COMMON_SIMD_FLAGS})
    if(LINK_LIBS)
      target_link_libraries("${TARGET_STDS}" PRIVATE ${LINK_LIBS})
    endif()
  endif()

  # --- Targets: Multi-SIMD-ISA (hwy_msa_ and gd_msa_) ---
  if(BUILD_MULTI_SIMD_)
    set(TARGET_HWY_MSA "hwy_msa_${BASE_NAME}")
    add_executable("${TARGET_HWY_MSA}" "${SOURCE_FILE}")
    target_compile_definitions("${TARGET_HWY_MSA}" PRIVATE USE_HWY MULTI_SIMD_ISA)
    target_compile_options("${TARGET_HWY_MSA}" PRIVATE ${MSA_FLAGS})
    target_include_directories("${TARGET_HWY_MSA}" PRIVATE ${hwy_INCLUDE_DIR})
    target_include_directories("${TARGET_HWY_MSA}" PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries("${TARGET_HWY_MSA}" PRIVATE hwy)
    if(LINK_LIBS)
      target_link_libraries("${TARGET_HWY_MSA}" PRIVATE ${LINK_LIBS})
    endif()

    set(TARGET_GD_MSA "gd_msa_${BASE_NAME}")
    add_executable("${TARGET_GD_MSA}" "${SOURCE_FILE}")
    target_compile_definitions("${TARGET_GD_MSA}" PRIVATE MULTI_SIMD_ISA)
    target_compile_options("${TARGET_GD_MSA}" PRIVATE ${MSA_FLAGS})
    target_include_directories("${TARGET_GD_MSA}" PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries("${TARGET_GD_MSA}" PRIVATE hwy)
    if(LINK_LIBS)
      target_link_libraries("${TARGET_GD_MSA}" PRIVATE ${LINK_LIBS})
    endif()
  endif()

endforeach()
