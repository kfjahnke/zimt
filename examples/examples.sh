#! /bin/bash

# compile examples which don't need to link to libraries (with the exception
# of the Vc example, which links statically to libVc.so)

for f in $@
do
  body=$(basename $f .cc)

  for compiler in clang++ g++
  do

    common_flags="-O3 -std=c++17 -march=native -Wno-deprecated-declarations"

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

    common_flags="-O3 -std=c++17 -march=native"

    echo $compiler -DUSE_STDSIMD $common_flags -ostds_${body}_$compiler $f
    $compiler -DUSE_STDSIMD $common_flags -ostds_${body}_$compiler $f

  done

done
