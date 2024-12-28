# cmake script to make the executable 'multi_tu' from the sources
# in this directory. To build, do this:
#   mkdir build
#   cd build
#   cmake ..
#   make

cmake_minimum_required(VERSION 3.31)
project ( multi_tu )

# we first need to figure out 'where we are' CPU-wise. This section
# is incomplete, but it should work on x86 and ARM targets.

if (     ${CMAKE_SYSTEM_PROCESSOR} STREQUAL x86_64
      OR ${CMAKE_SYSTEM_PROCESSOR} STREQUAL AMD64 )
  set(x86_64 TRUE)
  message(STATUS "***** setting i86 TRUE for an intel/AMD target")
else()
  set(x86_64 FALSE)
  message(STATUS "***** setting i86 FALSE; not an intel/AMD target")
  if ( ${CMAKE_SYSTEM_PROCESSOR} STREQUAL arm64 )
    message(STATUS "***** setting arm64 TRUE for an ARM target")
    set ( arm64 TRUE )
  else()
    message(STATUS "***** setting arm64 FALSE, not an ARM target")
    set ( arm64 FALSE )
    if ( ${CMAKE_SYSTEM_PROCESSOR} STREQUAL aarch64 )
      message(STATUS "***** setting aarch64 TRUE for an SVE target")
      set ( aarch64 TRUE )
    else()
      message(STATUS "***** setting aarch64 FALSE, not an SVE target")
      set ( aarch64 FALSE )
    endif()
  endif()
endif()

# assuming we have correctly detected the architecture, we list ISAs
# which can occur on that architecture

if ( x86_64 )
  list ( APPEND isa_l SSE2 SSSE3 SSE4 AVX2 AVX3 AVX3_ZEN4 AVX3_SPR )
# the NEON_BF16 ISA seems not to be built by default. We've used this
# code to build envutil on a macBook pro with M1 processor:
elseif ( arm64 )
  list ( APPEND isa_l NEON_WITHOUT_AES NEON ) # NEON_BF16
# tentative, we have no SVE systems
elseif ( aarch64 )
  list ( APPEND isa_l ALL_SVE SVE SVE2 SVE_256 SVE2_128 )
else()
  # tentative catch-all ISA if we're 'somewhere else'
  list ( APPEND isa_l EMU128 )
endif()

# for the test program composed of several ISA-specific TUs,
# we have the main program disp_to_tu.cc, which does the
# dispatching, and basic.cc, which has ISA-independent code

add_executable(multi_tu driver.cc dispatch.cc)

# we need to link with libhwy

target_link_libraries(multi_tu hwy)

# the main program needs specific compile options:

set_source_files_properties ( driver.cc PROPERTIES COMPILE_FLAGS "-O3" )
set_source_files_properties ( dispatch.cc PROPERTIES COMPILE_FLAGS "-O3 -I.." )

# for the ISA-specific object files holding 'payload' code,
# we use cmake 'object libraries'. This places the ISA-specific
# object files in separate directories, for which we use the
# same name as the ISA. For this program, each of the object
# libraries will only contain a single object file made from
# inset.cc with ISA-specific compilation instructions. Since
# we're already running a loop over the ISAs, we add a line
# to tell cmake to link the object file in.

foreach ( isa IN LISTS isa_l )

    add_library ( ${isa} OBJECT payload.cc )

    target_compile_options ( ${isa} PUBLIC -DTG_ISA=HWY_${isa} -O3 )

    target_link_libraries(multi_tu $<TARGET_OBJECTS:${isa}>)

endforeach()

