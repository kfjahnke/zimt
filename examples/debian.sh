#! /bin/bash

# adapted examples.sh for debian, which is missing the std::simd
# implementation (still uses g++ 10) - and the install path chosen
# by M. Kretz' std::simd implementation seems to be outside the
# scope inspected by the compiler, hence the -I argument.
# Also, I had to add -pthread and -lpthread, which wasn't necessary
# on the other platforms where I ran tests.

# compile all examples

for f in $@
do
  body=$(basename $f .cc)

  for compiler in clang++ g++
  do

    common_flags="-O3 -std=c++11 -march=native -lpthread -pthread"

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

    common_flags="-Ofast -std=c++17 -march=native -lpthread -pthread \
                 -I /usr/include/c++/10"

    echo $compiler -DUSE_STDSIMD $common_flags -ostds_${body}_$compiler $f
    $compiler -DUSE_STDSIMD $common_flags -ostds_${body}_$compiler $f

  done

done
