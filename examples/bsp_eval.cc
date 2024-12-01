/************************************************************************/
/*                                                                      */
/*    zimt - abstraction layer for SIMD programming                     */
/*                                                                      */
/*            Copyright 2024 by Kay F. Jahnke                           */
/*                                                                      */
/*    The git repository for this software is at                        */
/*                                                                      */
/*    https://github.com/kfjahnke/zimt                                  */
/*                                                                      */
/*    Please direct questions, bug reports, and contributions to        */
/*                                                                      */
/*    kfjahnke+zimt@gmail.com                                           */
/*                                                                      */
/*    Permission is hereby granted, free of charge, to any person       */
/*    obtaining a copy of this software and associated documentation    */
/*    files (the "Software"), to deal in the Software without           */
/*    restriction, including without limitation the rights to use,      */
/*    copy, modify, merge, publish, distribute, sublicense, and/or      */
/*    sell copies of the Software, and to permit persons to whom the    */
/*    Software is furnished to do so, subject to the following          */
/*    conditions:                                                       */
/*                                                                      */
/*    The above copyright notice and this permission notice shall be    */
/*    included in all copies or substantial portions of the             */
/*    Software.                                                         */
/*                                                                      */
/*    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND    */
/*    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES   */
/*    OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND          */
/*    NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT       */
/*    HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,      */
/*    WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING      */
/*    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR     */
/*    OTHER DEALINGS IN THE SOFTWARE.                                   */
/*                                                                      */
/************************************************************************/

// This is a test program for zimt's recently acquired b-spline
// processing capabilites and also serves to measure performance of the
// b-spline evaluation code with splines of various degrees and boundary
// conditions and varying SIMD back-ends/ISAs.

// There are two different ways of compiling it. The first is to create 
// a single-ISA binary, fixing the SIMD ISA at compile time by passing
// appropriate flags to the compiler. This will work with all SIMD
// back-ends - here, I show the compiler invocations for an AVX2 version.
// Note that - especially when using highway - additional flags may be
// needed to get a program which actually uses the AVX2 instructions.
// When highway uses pragmas to set the stage for AVX2 code, it uses
// all these flags: -msse2 -mssse3 -msse4.1 -msse4.2 -mpclmul -maes
// -mavx -mavx2 -mbmi -mbmi2 -mfma -mf16c
// Note also the -I. directive to tell the compiler to find files to
// #include in the current folder as well.

//   clang++ -mavx2 bsp_eval.ccc -O3 -I. -DUSE_HWY -lhwy
//   clang++ -mavx2 bsp_eval.ccc -O3 -I. -DUSE_VC -lVc
//   clang++ -mavx2 bsp_eval.ccc -O3 -I. -DUSE_STDSIMD
//   clang++ -mavx2 bsp_eval.ccc -O3 -I.

// The second way is to use highway's automatic dispatch to embedded
// variants of the code running with different ISAs. This requires the
// definition of MULTI_SIMD_ISA and linkage to libhwy and can only
// be used for the highway and the 'goading' back-end. Here, no
// architecture flags are passed to the compiler:

//   clang++ bsp_eval.ccc -O3 -I. -DMULTI_SIMD_ISA -DUSE_HWY -lhwy
//   clang++ bsp_eval.ccc -O3 -I. -DMULTI_SIMD_ISA -lhwy

// binaries made with the second method will dispatch to what is deemd
// the best SIMD ISA available on the CPU on which the binary is run.
// Because this is done meticulously by highway's CPU detection code,
// the binary variant which is picked is usually optimal and may
// out-perform single-ISA variants with 'manually' supplied ISA flags,
// if the set of flags isn't optimal as well. The disadvantage of the
// multi-SIMD-ISA variants is (much) longer compile time and code size.

// Due to the 'commodification' the source code itself doesn't have
// to be modified in any way to produce one variant or another.
// This suggests that during the implementation of a new program a
// fixed-ISA build can be used to evolve the code with fast turn-around
// times, adding dispatch capability later on by passing the relevant
// compiler flags.

// This particular program - evaluating b-splines - shows quite some
// variation between builds with different back-ends. While, in general,
// using better SIMD ISAs tend to speed things up, the differences
// arising from using different back-ends are harder to explain. For
// example, std::simd comes out surprisingly well for larger spline
// degrees. My conclusion is that, depending on the given program,
// there is no ready answer to the question which back-end will produce
// the best result and that it pays to experiment. Please note that
// the code to interface with the libraries I employ for the back-ends
// is entirely mine, so the performance measurements do reflect my
// use of these libraries, rather than qualities of these libraries.
// I hope to tweak the interface code to get as much performance out
// of the back-ends as possible, but this is difficult territory.

// I'll mark code sections which will differ from one example to the
// next, prefixing with ////////... and postfixing with //-------...
// You'll notice that there are three four places where you have to
// change stuff to set up your own program, and all the additions
// are simple (except for your 'client code', which may be complex).

// if the code is compiled to use the Vc or std::simd back-ends, we
// can't (yet) use highway's foreach_target mechanism, so we #undef
// MULTI_SIMD_ISA, which is zimt's way of activating that mechanism.

#if defined MULTI_SIMD_ISA && ( defined USE_VC || defined USE_STDSIMD )
#warning "un-defining MULTI_SIMD_ISA due to use of Vc or std::simd"
#undef MULTI_SIMD_ISA
#endif

// we define a dispatch base class. All the 'payload' code is called
// through virtual member functions of this class. In this example,
// we only have a single payload function. We have to enclose this
// base class definition in an include guard, because it must not
// be compiled repeatedly, which happens when highway's foreach_target
// mechansim is used. This definition might go to a header file, but
// it's better placed here, because it holds the declarations of the
// pure virtual member function(s) used as conduit to the ISA-specific
// code, and here, in an example, we want to see both these declarations
// and, later on, the implementations, in the same file.

#ifndef DISPATCH_BASE
#define DISPATCH_BASE

struct dispatch_base
{
  // in dispatch_base and derived classes, we keep two flags.
  // 'backend' holds a value indicating which of zimt's back-end
  // libraries is used. 'hwy_isa' is only set when the highway
  // backend is used and holds highway's HWY_TARGET value for
  // the given nested namespace.

  int backend = -1 ;
  unsigned long hwy_isa = 0 ;

  // next we have pure virtual member function definitions for
  // payload code. In this example, we only have one payload
  // function which calls what would be 'main' in a simple
  // program without multiple SIMD ISAs or SIMD back-ends

  virtual int payload ( int argc , char * argv[] ) const = 0 ;
} ;

#endif

#ifdef MULTI_SIMD_ISA

// if we're using MULTI_SIMD_ISA, we have to define HWY_TARGET_INCLUDE
// to tell the foreach_target mechanism which file should be repeatedly
// re-included and re-copmpiled with SIMD-ISA-specific flags

#undef HWY_TARGET_INCLUDE

/////////////// Tell highway which file to submit to foreach_target

#define HWY_TARGET_INCLUDE "bsp_eval.cc"  // this very file

//--------------------------------------------------------------------

#include <hwy/foreach_target.h>  // must come before highway.h

#include <hwy/highway.h>

#endif // #ifdef MULTI_SIMD_ISA

//////////////// Put the #includes needed for your program here:

#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <chrono>
#include "../zimt/eval.h"

//--------------------------------------------------------------------

// to make highway's use of #pragma directives to the compiler
// effective, we surround the SIMD-ISA-specific code with
// HWY_BEFORE_NAMESPACE() and HWY_AFTER_NAMESPACE().

HWY_BEFORE_NAMESPACE() ;

// this macro puts us into a nested namespace inside namespace 'project'.
// For single-SIMD-ISA builds, this is conventionally project::zsimd,
// and for multi-SIMD-ISA builds it is project::HWY_NAMESPACE. The macro
// is defined in common.h. After the macro invocation, we can access
// all zimt names with a simple zimt:: prefix - both 'true' zimt names
// and SIMD-ISA-specific versions living in the nested namespace.

BEGIN_ZIMT_SIMD_NAMESPACE(project)

// you can use float, but then can't use very high spline degrees.

typedef float dtype ;
typedef zimt::xel_t < dtype , 2 > crd_t ;
typedef zimt::xel_t < dtype , 3 > px_t ;

// Here, we define the SIMD-ISA-specific derived 'dispatch' class:

struct dispatch
: public dispatch_base
{
  // We fit the derived dispatch class with a c'tor which fills in
  // information about the nested SIMD ISA we're currently in.

  dispatch()
  {
    backend = int ( zimt::simdized_type<int,4>::backend ) ;
    #if defined USE_HWY || defined MULTI_SIMD_ISA
      hwy_isa = HWY_TARGET ;
    #endif
  }

  // 'payload', the SIMD-ISA-specific overload of dispatch_base's
  // pure virtual member function, now has the code which was in
  // main() when this example was first coded without dispatch.
  // One might be more tight-fisted with which part of the former
  // 'main' should go here and which part should remain in the
  // new 'main', but the little extra code which wouldn't benefit
  // from vectorization doesn't make much of a difference here.
  // Larger projects would have both several payload-type functions
  // and a body of code which is independent of vectorization.

///////////////// write a payload function with a 'main' signature
  
  int payload ( int argc , char * argv[] ) const
  {
    // we can get information about the specific dispatch object:

    std::cout << "payload code is using back-end: "
              << zimt::backend_name [ backend ] << std::endl ;

    #if defined USE_HWY || defined MULTI_SIMD_ISA

    std::cout << "highway target: "
              << hwy::TargetName ( hwy_isa ) << std::endl ;

    #endif

    long TIMES = 1 ;
    if ( argc > 1 )
      TIMES = std::atoi ( argv[1] ) ;
    else
    {
      std::cout << "enter number of repetitions: " ;
      std::cin >> TIMES ;
    }
      
    // get the spline degree and boundary conditions from the console
    // if necessary

    int spline_degree ;
    if ( argc > 2 )
      spline_degree = std::atoi ( argv[2] ) ;
    else
    {
      std::cout << "enter spline degree: " ;
      std::cin >> spline_degree ;
    }
    
    int bci = -1 ;
    zimt::bc_code bc ;
    if ( argc > 3 )
      bci = std::atoi ( argv[3] ) ;
    else
    {
      while ( bci < 1 || bci > 4 )
      {
        std::cout << "choose boundary condition" << std::endl ;
        std::cout << "1) MIRROR" << std::endl ;
        std::cout << "2) PERIODIC" << std::endl ;
        std::cout << "3) REFLECT" << std::endl ;
        std::cout << "4) NATURAL" << std::endl ;
        std::cin >> bci ;
      }
    }
    
    switch ( bci )
    {
      case 1 :
        bc = zimt::MIRROR ;
        break ;
      case 2 :
        bc = zimt::PERIODIC ;
        break ;
      case 3 :
        bc = zimt::REFLECT ;
        break ;
      case 4 :
        bc = zimt::NATURAL ;
        break ;
    }

    // we want a 2D b-spline of 1024X1024 2-channel values

    typedef zimt::bspline < px_t , 2 > spline_type ;
    zimt::xel_t < std::size_t , 2 > shape { 1024 , 1024 } ;

    spline_type bsp ( shape , spline_degree , bc ) ;

    // and an array of random values with equal extents

    zimt::array_t < 2 , px_t > a ( shape ) ;
    px_t * p = a.data() ;
    std::mt19937 gen(42); // Standard mersenne_twister_engine
    std::uniform_real_distribution<> dis(0.0, 1.0);
    for ( std::size_t i = 0 ; i < shape.prod() ; i++ )
    {
      p[i] = { dis(gen) , dis(gen) , dis(gen) } ;
    }

    // prefilter overload which 'pulls in' knot point data from an array

    bsp.prefilter ( a ) ;

    // std::cout << "created bspline object:" << std::endl
    //           << bsp << std::endl ;

    // create an evaluator

    auto ev = zimt::make_safe_evaluator ( bsp ) ;

    // set up an array to receive results

    zimt::array_t < 2 , px_t > trg ( shape ) ;

    // we want to time the operation

    std::chrono::system_clock::time_point start
      = std::chrono::system_clock::now() ;

    for ( std::size_t i = 0 ; i < TIMES ; i++ )
    {
      // a slightly more involved test: we have a 2D array with pairs of
      // double values ('a', see above), which we've 'prefiltered into'
      // a 2D b-spline. Now we run a 'transform' with *no source array*
      // which results in the 2D discrete coordinates of the target array
      // being used as source values. So the spline is evaluated at all
      // discrete coordinates and the result should be - within the
      // spline's fidelity - precisely the knot point values over which
      // the spline was erected. This is code equivalent to the use of
      // 'restore' which uses separable convolution with the b-spline
      // reconstruction kernel - the set of basis function values at
      // discrete coordinates.

      zimt::transform ( ev , trg ) ;
    }

    std::chrono::system_clock::time_point end
      = std::chrono::system_clock::now() ;

    std::cout << TIMES << " runs took:               "
              << std::chrono::duration_cast<std::chrono::milliseconds>
                  ( end - start ) . count()
              << " ms" << std::endl ;

    // we take a look at the result data - they should be very close to
    // the knot point data, since we've evaluated precisely at discrete
    // coordinates.

    const auto * p1 = a.data() ;
    const auto * p2 = trg.data() ;

    px_t mx = 0 , mn = 0 ;
    
    for ( std::size_t i = 0 ; i < shape.prod() ; i++ )
    {
      auto d = abs ( p1[i] - p2[i] ) ;
      mx = mx.at_least ( d ) ;
    }

    std::cout << "delta max: " << mx.hmax() << std::endl ;

    return 0 ;
  }
} ;

//--------------------------------------------------------------------

// we also code a local function _get_dispatch which returns a pointer
// to 'dispatch_base', which points to an object of the derived class
// 'dispatch'. This is used with highway's HWY_DYNAMIC_DISPATCH and
// returns the dispatch pointer for the SIMD ISA which highway deems
// most appropriate for the CPU on which the code is currently running.

const dispatch_base * const _get_dispatch()
{
  static dispatch d ;
  return &d ;
}

END_ZIMT_SIMD_NAMESPACE

HWY_AFTER_NAMESPACE() ;

// Now for code which isn't SIMD-ISA-specific. ZIMT_ONCE is defined
// as either HWY_ONCE (if MULTI_SIMD_ISA is #defined) or simply true
// otherwise - then, there is only one compilation anyway.

#if ZIMT_ONCE

namespace project {

#ifdef MULTI_SIMD_ISA

// we're using highway's foreach_target mechanism. To get access to the
// SIMD-ISA-specific variant of _get_dispatch (in project::HWY_NAMESPACE)
// we use the HWY_EXPORT macro:

HWY_EXPORT(_get_dispatch);

// now we can code get_dispatch: it simply uses HWY_DYNAMIC_DISPATCH
// to pick the SIMD-ISA-specific get_dispatch variant, which in turn
// yields the desired dispatch_base pointer.

const dispatch_base * const get_dispatch()
{
  return HWY_DYNAMIC_DISPATCH(_get_dispatch)() ;
}

#else // #ifdef MULTI_SIMD_ISA

// if we're not using highway's foreach_target mechanism, there is
// only a single _get_dispatch variant in namespace project::zsimd.
// So we call that one, to receive the desired dispatch_base pointer.

const dispatch_base * const get_dispatch()
{
  return zsimd::_get_dispatch() ;
}

#endif // #ifdef MULTI_SIMD_ISA

}  // namespace project

int main ( int argc , char * argv[] )
{
  // Here we use zimt's dispatch mechanism: first, we get a pointer
  // to the dispatcher, then we invoke a member function of the
  // dispatcher. What's the point? We can call a SIMD-ISA-specific
  // bit of code without having to concern ourselves with figuring
  // out which SIMD ISA to use on the current CPU: this happens via
  // highway's dispatch mechanism, or is fixed at compile time, but
  // in any case we receive a dispatch_base pointer routing to the
  // concrete variant. project::get_dispatch might even be coded
  // to provide pointers to dispatch objects in separate TUs, e.g.
  // when these TUs use different back-ends or compiler flags. Here,
  // we can remain unaware of how the concrete dispatch object is
  // set up and the pointer obtained.

  auto dp = project::get_dispatch() ;

  // we can get information about the specific dispatch object:

  std::cout << "calling payload with "
#ifdef MULTI_SIMD_ISA
            << "dynamic dispatch" << std::endl ;
#else
            << "static dispatch" << std::endl ;
#endif

  // now we call the payload via the dispatch_base pointer.

  int success = dp->payload ( argc , argv ) ;

  std::cout << std::endl ;

// to invoke other variants with explicit calls, uncomment this section
// (x86 only - other CPU types would need other N_... macros)

// #ifdef MULTI_SIMD_ISA
// 
//   std::cout << "******** calling payloads with successively better ISAs:"
//             << std::endl ;
//   
//   std::cout << "******** calling SSE2 payload" << std::endl ;
//   project::N_SSE2::dispatch d3 ;
//   d3.payload ( argc , argv ) ;
//   
//   std::cout << "******** calling SSSE3 payload" << std::endl ;
//   project::N_SSSE3::dispatch d4 ;
//   d4.payload ( argc , argv ) ;
//   
//   std::cout << "******** calling SSE4 payload" << std::endl ;
//   project::N_SSE4::dispatch d5 ;
//   d5.payload ( argc , argv ) ;
//   
//   std::cout << "******** calling AVX2 payload" << std::endl ;
//   project::N_AVX2::dispatch d6 ;
//   d6.payload ( argc , argv ) ;
//   
//   std::cout << "******** calling AVX3 payload" << std::endl ;
//   project::N_AVX3::dispatch d7 ;
//   d7.payload ( argc , argv ) ;
// 
// #endif
}

#endif  // ZIMT_ONCE

