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

// This is a test program for zimt's separable convolution code.

// There are two different ways of compiling it. The first is to create 
// a single-ISA binary, fixing the SIMD ISA at compile time by passing
// appropriate flags to the compiler. This will work with all SIMD
// back-ends - here, I show the compiler invocations for an AVX2 version.
// Note the -I. directive to tell the compiler to find files to #include
// in the current folder as well.

//   clang++ -mavx2 convolve.ccc -O2 -I. -DUSE_HWY -lhwy
//   clang++ -mavx2 convolve.ccc -O2 -I. -DUSE_VC -lVc
//   clang++ -mavx2 convolve.ccc -O2 -I. -DUSE_STDSIMD
//   clang++ -mavx2 convolve.ccc -O2 -I.

// The second way is to use highway's automatic dispatch to embedded
// variants of the code running with different ISAs. This requires the
// definition of MULTI_SIMD_ISA and linkage to libhwy and can only
// be used for the highway and the 'goading' back-end. Here, no
// architecture flags are passed to the compiler:

//   clang++ convolve.ccc -O2 -I. -DMULTI_SIMD_ISA -DUSE_HWY -lhwy
//   clang++ convolve.ccc -O2 -I. -DMULTI_SIMD_ISA -lhwy

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

// Here's a typical compilation and test run on my machine:

// ~/zimt/examples$ clang++ -O3 -std=gnu++17 -march=native -mavx2
//   -march=haswell -mpclmul -maes -Wno-deprecated-declarations
//   -ohwy_convolve_clang++ convolve.cc -lvigraimpex -DMULTI_SIMD_ISA
//   -I. -lhwy -DUSE_HWY

// ~/zimt/examples$ ./hwy_convolve_clang++ *.JPG .1 .1 .1 .1 .1 .1 .1 .1 .1 .1
// payload code is using back-end: highway
// highway target: AVX2
// payload function took 100 ms
// storing the target image as 'convolved.tif'


// if the code is compiled to use the Vc or std::simd back-ends, we
// can't (yet) use highway's foreach_target mechanism, so we #undef
// MULTI_SIMD_ISA, which is zimt's way of activating that mechanism.

#if defined MULTI_SIMD_ISA && ( defined USE_VC || defined USE_STDSIMD )
#warning "un-defining MULTI_SIMD_ISA due to use of Vc or std::simd"
#undef MULTI_SIMD_ISA
#endif

// I'll mark code sections which will differ from one example to the
// next, prefixing with ////////... and postfixing with //-------...
// You'll notice that there are only few places where you have to
// change stuff to set up your own program, and all the additions
// are simple (except for your 'client code', which may be complex).

/////////////////////// #include 'regular' headers here:

#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <chrono>

#include <vigra/stdimage.hxx>
#include <vigra/imageinfo.hxx>
#include <vigra/impex.hxx>

//--------------------------------------------------------------------

// the payload function receives zimt data types. We #include the
// zimt headers needed to declare the arguments - these headers
// are independent of the SIMD ISA.

#include "../zimt/simd/simd_tag.h"
#include "../zimt/array.h"
#include "../zimt/xel.h"

// in this program, we want to interface with vigra data types

#include "../zimt/zimt_vigra.h"

// we silently assume we have a colour image

typedef zimt::xel_t < float , 3 > pixel_type; 

// we define a dispatch base class. All the 'payload' code is called
// through virtual member functions of this class. In this example,
// we only have a single payload function. We have to enclose this
// base class definition in an include guard, because it must not
// be compiled repeatedly, which happens when highway's foreach_target
// mechansim is used.

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

  virtual int payload
    ( zimt::view_t < 2 , pixel_type > & img ,
      const std::vector < zimt::xlf_type > & kernel ) const = 0 ;
} ;

#endif

#ifdef MULTI_SIMD_ISA

// if we're using MULTI_SIMD_ISA, we have to define HWY_TARGET_INCLUDE
// to tell the foreach_target mechanism which file should be repeatedly
// re-included and re-copmpiled with SIMD-ISA-specific flags

#undef HWY_TARGET_INCLUDE

/////////////// Tell highway which file to submit to foreach_target

#define HWY_TARGET_INCLUDE "convolve.cc"  // this very file

//--------------------------------------------------------------------

// now the 'magic header':

#include <hwy/foreach_target.h>  // must come before highway.h

#include <hwy/highway.h>

#endif // #ifdef MULTI_SIMD_ISA

/////////////////////// #include additional zimt headers here.

// convolve.h pulls in the separable convolution code and is
// adapted to benefit from the multi-SIMD-ISA mechanism. We
// can only #include it here, after foreach_target.h and
// highway.h, because it relies on highway code.

#include "../zimt/convolve.h"

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

///////////////// now comes the SIMD_ISA-specific code:

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
  // In this example, we use a payload which only covers the part
  // of the program which actually benefits from vectorization, and
  // put the remainder of the program into 'main'.

  int payload ( zimt::view_t < 2 , pixel_type > & img ,
                const std::vector < zimt::xlf_type > & kernel ) const
  {
    // we can get information about the specific dispatch object:

    std::cout << "payload code is using back-end: "
              << zimt::backend_name [ backend ] << std::endl ;

    #if defined USE_HWY || defined MULTI_SIMD_ISA

    std::cout << "highway target: "
              << hwy::TargetName ( hwy_isa ) << std::endl ;

    #endif

    // finally, we call zimt::convolve

    zimt::convolve
    ( img ,
      img ,
      { zimt::MIRROR , zimt::MIRROR } ,
      kernel ,
      kernel.size() / 2 ) ;
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

//////////////////// we code a conventional 'main' which invokes the
// SIMD-ISA-specific payload code via a dispatch pointer.

int main ( int argc , char * argv[] )
{
  if ( argc < 3 )
  {
    std::cerr << "pass a colour image file as argument," << std::endl ;
    std::cerr << "followed by the filter's coefficients" << std::endl ;
    exit( -1 ) ;
  }

  // get the image file name
  
  vigra::ImageImportInfo imageInfo ( argv[1] ) ;

  char * end ;

  std::vector < zimt::xlf_type > kernel ;

  for ( int i = 2 ; i < argc ; i++ )
    kernel.push_back ( zimt::xlf_type ( strtold ( argv [ i ] , &end ) ) ) ;

  // set up a zimt::array_t with this shape

  zimt::array_t < 2 , pixel_type >
    a ( zimt::to_zimt ( imageInfo.shape() ) ) ;

  // import the image

  vigra::importImage ( imageInfo , zimt::to_vigra ( a ) ) ;

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

  // apply the filter, measure the time the payload code takes

  std::chrono::system_clock::time_point t_start
    = std::chrono::system_clock::now();

  dp->payload ( a , kernel ) ;

  std::chrono::system_clock::time_point t_end
    = std::chrono::system_clock::now();

  std::cout << "payload function took "
            << std::chrono::duration_cast<std::chrono::milliseconds>
                 ( t_end - t_start ) . count()
       << " ms" << std::endl ;

  // store the result with vigra impex. Note that vigra will compress
  // the dynamic range if your filter amplifies peaks in the incoming
  // signal - use a filter with unit gain and all-positive coefficients
  // to get the same dynamic range in the output.

  vigra::ImageExportInfo eximageInfo ( "convolved.tif" );
  
  std::cout << "storing the target image as 'convolved.tif'" << std::endl ;
  
  vigra::exportImage ( zimt::to_vigra ( a ) ,
                       eximageInfo
                       .setPixelType("UINT8") ) ;
}

#endif  // ZIMT_ONCE

