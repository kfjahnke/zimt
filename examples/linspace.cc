/************************************************************************/
/*                                                                      */
/*    zimt - abstraction layer for SIMD programming                     */
/*                                                                      */
/*            Copyright 2023 by Kay F. Jahnke                           */
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

// example code producing arrays holding linear gradients, just like
// what you get from NumPy's 'linspace'. this example is elaborated
// to function either with highway's foreach_target mechanism, using
// internal dispatch to the best SIMD ISA on the CPU it's run on, or,
// alternatively, using a single fixed-SIMD-ISA where the ISA is
// determined at compile time by passing specific compiler flags.
// The 'payload' code expects to be called from a 'driver' program
// like driver.cc - this has to be linked in.

// if the code is compiled to use the Vc or std::simd back-ends, we
// can't (yet) use highway's foreach_target mechanism, so we #undef
// MULTI_SIMD_ISA, which is zimt's way of activating that mechanism.

#if defined MULTI_SIMD_ISA && ( defined USE_VC || defined USE_STDSIMD )
#warning "un-defining MULTI_SIMD_ISA due to use of Vc or std::simd"
#undef MULTI_SIMD_ISA
#endif

#ifdef MULTI_SIMD_ISA

// if we're using MULTI_SIMD_ISA, we have to define HWY_TARGET_INCLUDE
// to tell the foreach_target mechanism which file should be repeatedly
// re-included and re-copmpiled with SIMD-ISA-specific flags

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "linspace.cc"  // this file

#include <hwy/foreach_target.h>  // must come before highway.h
#include <hwy/highway.h>

#endif

// now we #include the zimt headers we need:

#include "../zimt/wielding.h"

// this macro puts us into a nested namespace inside namespace 'project'.
// For single-SIMD-ISA builds, this is conventionally project::zsimd,
// and for multi-SIMD-ISA builds it is project::HWY_NAMESPACE. The macro
// is defined in common.h

BEGIN_ZIMT_SIMD_NAMESPACE(project)

// we use a namespace alias 'zimt' for the corresponding nested
// namespace in namespace zimt. since the nested namespace in
// namespace zimt has a using declaration for 'plain' namespace
// zimt, we can use a zimt:: qualifier for all symbols from
// 'plain' zimt as well as all symbolds from zimt::ZIMT_SIMD_ISA.

namespace zimt = zimt::ZIMT_SIMD_ISA ;

static ZIMT_ATTR int _payload()
{
#ifdef USE_HWY
  std::cout << "paylod: target = " << hwy::TargetName ( HWY_TARGET )
            << std::endl ;
#endif

  zimt::bill_t bill ;
  static const std::size_t VSZ = 16 ;

  // let's start with a simple 1D linspace.

  {
    typedef  zimt::xel_t < float , 1 > delta_t ;
    delta_t start { .5 } ;
    delta_t step { .1 } ;
    zimt::linspace_t < float , 1 , 1 , VSZ > l ( start , step , bill ) ;
    typedef  zimt::echo < float , 1 , VSZ > act_t ;
    zimt::array_t < 1 , delta_t > a ( 7 ) ;
    zimt::storer < float , 1 , 1 , VSZ > p ( a , bill ) ;

    zimt::process ( a.shape , l , act_t() ,  p , bill ) ;

    std::cout << "********** 1D:" << std::endl << std::endl ;

    for ( std::size_t i = 0 ; i < 7 ; i++ )
      std::cout << " " << a [ i ] ;
    std::cout << std::endl << std::endl ;
  }

  // now we'll go 2D

  {
    typedef  zimt::xel_t < float , 2 > delta_t ;
    delta_t start { .5 , 0.7 } ;
    delta_t step { .1 , .2 } ;
    zimt::linspace_t < float , 2 , 2 , VSZ > l ( start , step , bill ) ;
    typedef  zimt::pass_through < float , 2 , VSZ > act_t ;
    zimt::array_t < 2 , delta_t > a ( { 7 , 5 } ) ;
    zimt::storer < float , 2 , 2 , VSZ > p ( a , bill ) ;

    zimt::process ( a.shape , l , act_t() ,  p , bill ) ;

    std::cout << "********** 2D:" << std::endl << std::endl ;

    for ( std::size_t y = 0 ; y < 5 ; y++ )
    {
      for ( std::size_t x = 0 ; x < 7 ; x++ )
        std::cout << " " << a [ { x , y } ] ;
      std::cout << std::endl ;
    }
    std::cout << std::endl ;
  }

  // just to show off, 3D

  {
   typedef  zimt::xel_t < float , 3 > delta_t ;
    delta_t start { .5 , 0.7 , -.9 } ;
    delta_t step { .1 , .2 , -.4 } ;
    zimt::linspace_t < float , 3 , 3 , VSZ > l ( start , step , bill ) ;
    typedef  zimt::pass_through < float , 3 , VSZ > act_t ;
    zimt::array_t < 3 , delta_t > a ( { 2 , 3 , 4 } ) ;
    zimt::storer < float , 3 , 3 , VSZ > p ( a , bill ) ;

    std::cout << "********** 3D:" << std::endl << std::endl ;

    zimt::process ( a.shape , l , act_t() ,  p , bill ) ;
    for ( std::size_t z = 0 ; z < 4; z++ )
    {
      for ( std::size_t y = 0 ; y < 3 ; y++ )
      {
        for ( std::size_t x = 0 ; x < 2 ; x++ )
          std::cout << " " << a [ { x , y , z } ] ;
        std::cout << std::endl ;
      }
      std::cout << std::endl ;
    }

    zimt::process ( a.shape , l , act_t() ,  p , bill ) ;
    for ( std::size_t z = 0 ; z < 4; z++ )
    {
      for ( std::size_t y = 0 ; y < 3 ; y++ )
      {
        for ( std::size_t x = 0 ; x < 2 ; x++ )
          std::cout << " " << a [ { x , y , z } ] ;
        std::cout << std::endl ;
      }
      std::cout << std::endl ;
    }
  }
  return 0 ;
}

END_ZIMT_SIMD_NAMESPACE

#if HWY_ONCE

namespace project {

#ifdef MULTI_SIMD_ISA
  
HWY_EXPORT(_payload);

// Callpayload is defined in namespace project, but it might be in
// another namespace as well.

int payload ( int argc , char * argv[] )
{
  return HWY_DYNAMIC_DISPATCH(_payload)() ;
}

#else

int payload ( int argc , char * argv[] )
{
  return zsimd::_payload() ;
}

#endif

}  // namespace project

#endif  // HWY_ONCE

