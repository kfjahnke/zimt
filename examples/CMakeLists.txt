# CMakeLists.txt for zimt examples

cmake_minimum_required(VERSION 3.16)
project(ZimtExamples CXX)

# Enforce C++17 with GNU extensions
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

# Add the parent directory to the include search path
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/..")

# Look for packages natively supported by modern CMake targets
find_package(Threads REQUIRED)
find_package(OpenImageIO QUIET)
find_package(Vc QUIET)

# Hard-force -pthread compiler flag on Unix-like systems.
# This bypasses the fragile CMake thread-detection which often fails 
# with Clang inside Conda environments.
if(UNIX)
  add_compile_options(-pthread)
  # add_link_options(-pthread)
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
  # This ensures -pthread is correctly appended for both compiler
  # and linker stages
  set(LINK_LIBS Threads::Threads)

  get_filename_component(BASE_NAME "${SOURCE_FILE}" NAME_WE)
  file(READ "${SOURCE_FILE}" FILE_CONTENT)

  # 1. Determine linking requirements based on file content.
  # set(LINK_LIBS "")
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
  set(BUILD_STD_SIMD TRUE)
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
  add_executable("${TARGET_GD}" "${SOURCE_FILE}")
  target_compile_options("${TARGET_GD}" PRIVATE ${COMMON_SIMD_FLAGS})
  if(LINK_LIBS)
    target_link_libraries("${TARGET_GD}" PRIVATE ${LINK_LIBS})
  endif()

  # --- Target: Vc (vc_) ---
  if(Vc_FOUND)
    set(TARGET_VC "vc_${BASE_NAME}")
    add_executable("${TARGET_VC}" "${SOURCE_FILE}")
    target_compile_definitions("${TARGET_VC}" PRIVATE USE_VC)
    target_compile_options("${TARGET_VC}" PRIVATE ${COMMON_SIMD_FLAGS})
    target_link_libraries("${TARGET_VC}" PRIVATE Vc)
    if(LINK_LIBS)
      target_link_libraries("${TARGET_VC}" PRIVATE ${LINK_LIBS})
    endif()
  endif()

  # --- Target: Highway (hwy_) ---
  set(TARGET_HWY "hwy_${BASE_NAME}")
  add_executable("${TARGET_HWY}" "${SOURCE_FILE}")
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
  if(BUILD_MULTI_SIMD)
    set(TARGET_HWY_MSA "hwy_msa_${BASE_NAME}")
    add_executable("${TARGET_HWY_MSA}" "${SOURCE_FILE}")
    target_compile_definitions("${TARGET_HWY_MSA}" PRIVATE USE_HWY MULTI_SIMD_ISA)
    target_compile_options("${TARGET_HWY_MSA}" PRIVATE ${MSA_FLAGS})
    target_include_directories("${TARGET_HWY_MSA}" PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries("${TARGET_HWY_MSA}" PRIVATE hwy)
    if(LINK_LIBS)
      target_link_libraries("${TARGET_HWY_MSA}" PRIVATE ${LINK_LIBS})
    endif()

    set(TARGET_STDS_MSA "stds_msa_${BASE_NAME}")
    add_executable("${TARGET_STDS_MSA}" "${SOURCE_FILE}")
    target_compile_definitions("${TARGET_STDS_MSA}" PRIVATE USE_STDSIMD MULTI_SIMD_ISA)
    target_compile_options("${TARGET_STDS_MSA}" PRIVATE ${MSA_FLAGS})
    target_include_directories("${TARGET_STDS_MSA}" PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries("${TARGET_STDS_MSA}" PRIVATE hwy)
    if(LINK_LIBS)
      target_link_libraries("${TARGET_STDS_MSA}" PRIVATE ${LINK_LIBS})
    endif()

    set(TARGET_VC_MSA "vc_msa_${BASE_NAME}")
    add_executable("${TARGET_VC_MSA}" "${SOURCE_FILE}")
    target_compile_definitions("${TARGET_VC_MSA}" PRIVATE USE_VC MULTI_SIMD_ISA)
    target_compile_options("${TARGET_VC_MSA}" PRIVATE ${MSA_FLAGS})
    target_include_directories("${TARGET_VC_MSA}" PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries("${TARGET_VC_MSA}" PRIVATE hwy Vc)
    if(LINK_LIBS)
      target_link_libraries("${TARGET_VC_MSA}" PRIVATE ${LINK_LIBS})
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
