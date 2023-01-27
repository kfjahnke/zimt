#! /bin/bash

# compile all examples

for f in $@
do
  body=$(basename $f .cc)

  for compiler in clang++ g++
  do
  
    common_flags="-Ofast -std=c++11 -march=native"

    # compile without explicit SIMD code

    echo $compiler $common_flags -ovs_${body}_$compiler $f
    $compiler $common_flags -ovs_${body}_$compiler $f

    # compile with Vc

    echo $compiler -DUSE_VC $common_flags -ovc_${body}_$compiler $f -lVc
    $compiler -DUSE_VC $common_flags -ovc_${body}_$compiler $f -lVc

    # compile with highway

    echo $compiler -DUSE_HWY $common_flags -ohwy_${body}_$compiler $f
    $compiler -DUSE_HWY $common_flags -ohwy_${body}_$compiler $f

    # compile with std::simd (needs std:simd implementation)

    common_flags="-Ofast -std=c++17 -march=native"

    echo $compiler -DUSE_STDSIMD $common_flags -ostds_${body}_$compiler $f
    $compiler -DUSE_STDSIMD $common_flags -ostds_${body}_$compiler $f

  done

done
