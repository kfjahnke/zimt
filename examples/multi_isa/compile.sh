# simple shell script to compile and run the demo for combining
# several ISAs in one binary using 'namespace dubbing', as explained
# in the README in 'using several ISAs/backends in one binary'

g++ -msse2 -c -DUSE_HWY -o foo_sse.o foo_sse.cc
g++ -mavx2 -march=haswell -mpclmul -maes -c -DUSE_HWY -o foo_avx2.o foo_avx2.cc
g++ main.cc *.o
./a.out
