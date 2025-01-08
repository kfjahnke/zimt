#          cmake file for 'multi_tu' example program
#          
#          Copyright Kay F. Jahnke <kfjahnke+zimt@gmail.com> 2025
#
#  The git repository for this software is the zimt repository:
#
#  https://github.com/kfjahnke/zimt
#
#  Please direct questions, bug reports, and contributions to
#
#  kfjahnke+zimt@gmail.com
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# This cmake file is tentative, but it should enable builds of the test
# program 'multi_tu' with automatic ISA dispatch on x86_64 and ARM machines.
# You may want to read the accompanying text:
# https://github.com/kfjahnke/zimt/blob/main/examples/multi_tu_example/multi_tu.md
# This text gives you the gist of the program - the code in this directory
# adds more comments and embellishments.

cmake_minimum_required(VERSION 3.31)
project ( multi_tu )

# we first need to figure out 'where we are' CPU-wise. This section
# is incomplete, but it should work on x86_64 and ARM targets.

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
# which can occur on that architecture. For now, we provide the list
# of ISAs explicitly, but it would be preferable to obtain them
# directly from highway, which has this information handy: highway
# builds the hwy_list_targets helper program, which emits just the
# information which is needed here. e.g. on my system, this utility
# produces this output:

# $ ./hwy_list_targets
# Config: emu128:0 scalar:0 static:0 all_attain:0 is_test:0
# Compiled HWY_TARGETS:   AVX3_SPR AVX3_ZEN4 AVX3 AVX2 SSE4 SSSE3 SSE2
# HWY_ATTAINABLE_TARGETS: AVX3_SPR AVX3_ZEN4 AVX3 AVX2 SSE4 SSSE3 SSE2 EMU128
# HWY_BASELINE_TARGETS:   SSE2 EMU128
# HWY_STATIC_TARGET:      SSE2
# HWY_BROKEN_TARGETS:    
# HWY_DISABLED_TARGETS:  
# Current CPU supports:   AVX2 SSE4 SSSE3 SSE2 EMU128 SCALAR

# What we need here is the first list, 'compiled HWY_TARGETS'. I think
# that cmake can be made to extract such information from the invocation
# of an external command, but it would be preferable if the information
# could be gleaned via cmake's find_package, where highway could deploy
# it during it's own installation process. Then, the information would
# automatically be up-to-date with the current cmake installation and
# ISA additions/removals with new highway versions would be reflected
# immediately. I have added EMU128 to the list - it's not used be code
# compiled with foreach_target, but I add a dispatch option 'manually'
# in the example program. The list defined here decides which object
# files are made - if object files are made which the program doesn't
# use, no harm is done.

if ( x86_64 )
  list ( APPEND isa_l AVX3_SPR AVX3_ZEN4 AVX3 AVX2 SSE4 SSSE3 SSE2 EMU128 )
# the NEON_BF16 ISA seems not to be built by default. We've used this
# code to build envutil on a macBook pro with M1 processor:
elseif ( arm64 )
  list ( APPEND isa_l NEON_WITHOUT_AES NEON NEON_BF16 )
# tentative, we have no SVE systems
elseif ( aarch64 )
  list ( APPEND isa_l ALL_SVE SVE SVE2 SVE_256 SVE2_128 )
else()
  # tentative catch-all ISA if we're 'somewhere else'
  list ( APPEND isa_l EMU128 )
endif()

# for the test program composed of several ISA-specific TUs,
# we have the main program disp_to_tu.cc, which does the
# dispatching, and dispatch.cc, which has ISA-independent code

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
# payload.cc with ISA-specific compilation instructions. Since
# we're already running a loop over the ISAs, we add a line
# to tell cmake to link the object file in.

foreach ( isa IN LISTS isa_l )

    add_library ( ${isa} OBJECT payload.cc )
    target_compile_options ( ${isa} PUBLIC -DTG_ISA=HWY_${isa} -O3 )
    target_link_libraries(multi_tu $<TARGET_OBJECTS:${isa}>)

endforeach()

# To demonstrate that we can also add dispatch options to payload code
# which is not managed by highway, we add 'e_payload.cc'. Note that we
# might use arbitrary compile options here - beyond the -O3. This is
# the route which you may want to take for multi-ISA programs without
# highway support.

add_library ( epl_test OBJECT e_payload.cc )
target_compile_options ( epl_test PUBLIC -O3 )
target_link_libraries(multi_tu $<TARGET_OBJECTS:epl_test>)

add_library(payload_lib SHARED l_payload.cc)
target_compile_options ( payload_lib PUBLIC -O3 )
target_link_libraries(multi_tu payload_lib)

add_library(payload_plugin SHARED p_payload.cc)
target_compile_options ( payload_plugin PUBLIC -O3 )
