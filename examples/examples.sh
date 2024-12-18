#! /bin/bash

# compile examples

# usage: ./compile_examples *.cc

# pass all examples you wish to compile on the command line. Some of
# the examples link to shared libraries, and building them will fail
# if the libraries aren't present. If you get linker errors, that is
# likely the reason. This script will also compile the examples both
# with g++ and clang++, and with all four possible SIMD backends,
# which also need to be available.
# Additionally, if grep finds 'dispatch_base' in the source file, it
# does one clang++ and one g++ compilation with MULTI_SIMD_ISA' defined,
# which produces a multi-SIMD-ISA build.
# all binaries are named so that the way of their compilation is
# recognizable. I use four prefixes for the four back-ends:
# gd_ for zimt's own 'goading' back-end
# vc_ for zimt's Vc back-end
# hwy_ for zimt's highway back-end
# stds_ for zimt's std::simd back-end
# multi-SIMD-ISA builds add msa_ to the prefix. The compiler which
# was used is appended as a suffix.
# The examples in this folder with a .cc extension should all compile
# with this script (provided their dependencies are present); files
# with -cpp extension require 'special treatment', look into the code
# for hint on how to compile them.

for f in $@
do
  # some examples use OpenImageIO or vigraimpex:

  if [[ $(grep OpenImageIO $f) != "" ]]
  then
     link_libs="-lOpenImageIO -lOpenImageIO_Util"
  else
     link_libs=""
  fi

  if [[ $(grep "vigra.impex" $f) != "" ]]
  then
     link_libs="$link_libs -lvigraimpex"
  fi

  if [[ $(grep 'no-std-simd' $f) == "" ]]
  then
     build_for_std_simd="y"
  else
     echo "$f is tagged not to build with std::simd"
     build_for_std_simd=""
  fi

  body=$(basename $f .cc)

  if [[ "$link_libs" != "" ]]
  then
    echo "will link $basename with $link_libs"
  fi

  for compiler in clang++ g++
  do

    # for most compilations
    # TODO: compiling with -std=c++11 fails with g++

    common_flags="-O3 -std=gnu++17 -msse2 -mssse3 -msse4.1 -msse4.2 -mpclmul -maes -mavx -mavx2 -mbmi -mbmi2 -mfma -mf16c"

    # for compilations with multi-SIMD-ISA internal dispatching

    msa_flags="-O3 -std=gnu++17"

    # for compilations with std::simd

    stds_flags="-O3 -std=gnu++17 -msse2 -mssse3 -msse4.1 -msse4.2 -mpclmul -maes -mavx -mavx2 -mbmi -mbmi2 -mfma -mf16c"

    # compile without explicit SIMD code

    echo $compiler $common_flags -ogd_${body}_$compiler $f $link_libs
    $compiler $common_flags -ogd_${body}_$compiler $f $link_libs

    # compile with Vc

    echo $compiler -DUSE_VC $common_flags -ovc_${body}_$compiler $f -lVc $link_libs
    $compiler -DUSE_VC $common_flags -ovc_${body}_$compiler $f -lVc $link_libs

    # compile with highway

    echo $compiler -DUSE_HWY $common_flags -ohwy_${body}_$compiler $f $link_libs
    $compiler -DUSE_HWY $common_flags -ohwy_${body}_$compiler $f $link_libs

    # compile with std::simd (needs std::simd implementation)

    if [[ "$build_for_std_simd" != "" ]]
    then
      echo $compiler -DUSE_STDSIMD $stds_flags -ostds_${body}_$compiler $f $link_libs
      $compiler -DUSE_STDSIMD $stds_flags -ostds_${body}_$compiler $f $link_libs
    fi

    # if grep finds 'dispatch_base' in the .cc file, we assume it's a multi-SIMD-ISA
    # program and #define MULTI_SIMD_ISA

    if [[ $(grep dispatch_base $f) != "" ]]
    then
        echo $compiler -DUSE_HWY $msa_flags -ohwy_msa_${body}_$compiler $f -DMULTI_SIMD_ISA $link_libs -lhwy -I.
        $compiler -DUSE_HWY $msa_flags -ohwy_msa_${body}_$compiler $f -DMULTI_SIMD_ISA $link_libs -lhwy -I.

        echo $compiler $msa_flags -ogd_msa_${body}_$compiler $f -DMULTI_SIMD_ISA $link_libs -lhwy -I.
        $compiler $msa_flags -ogd_msa_${body}_$compiler $f -DMULTI_SIMD_ISA $link_libs -lhwy -I.
    fi

  done

done
