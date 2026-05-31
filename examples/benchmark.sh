#! /bin/bash

# a good candidate for benchmarking the various variants using
# clang++ and g++ with different SIMD back-ends and fixed ISA
# or dynamic dispatch is bsp_eval.cc. this program sets up and
# evaluates a b-spline of given degree and boundary conditions
# repeatedly and prints out the options used and the time it
# takes.

for cmp in gnu clang
do
  for pfx in gd gd_msa stds stds_msa vc vc_msa hwy hwy_msa
  do
    echo build-${cmp}/${pfx}_bsp_eval 100 5 2
    build-${cmp}/${pfx}_bsp_eval 100 5 2
  done
done
