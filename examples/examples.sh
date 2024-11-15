#! /bin/bash

# compile examples

# usage: ./compile_examples *.cc

# pass all examples you wish to compile on the command line. Some of
# the examples link to shared libraries, and building them will fail
# if the libraries aren't present. If you get linker errors, that is
# likely the reason. This script will also compile the examples both
# with g++ and clang++, and with all four possible SIMD backends,
# which also need to be available.
# The examples i this folder with a .cc extension should all compile
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

    common_flags="-O -std=gnu++17 -march=native -mavx2 -march=haswell -mpclmul -maes -Wno-deprecated-declarations"

    # compile without explicit SIMD code

    echo $compiler $common_flags -ovs_${body}_$compiler $f $link_libs
    $compiler $common_flags -ovs_${body}_$compiler $f $link_libs

    # compile with Vc

    echo $compiler -DUSE_VC $common_flags -ovc_${body}_$compiler $f -lVc $link_libs
    $compiler -DUSE_VC $common_flags -ovc_${body}_$compiler $f -lVc $link_libs

    # compile with highway

    echo $compiler -DUSE_HWY $common_flags -ohwy_${body}_$compiler $f $link_libs
    $compiler -DUSE_HWY $common_flags -ohwy_${body}_$compiler $f $link_libs

    # compile with std::simd (needs std::simd implementation)

    if [[ "$build_for_std_simd" != "" ]]
    then
      common_flags="-O3 -std=c++17 -march=native"

      echo $compiler -DUSE_STDSIMD $common_flags -ostds_${body}_$compiler $f $link_libs
      $compiler -DUSE_STDSIMD $common_flags -ostds_${body}_$compiler $f $link_libs
    fi

  done

done
