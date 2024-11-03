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

// convolve an image with a kernel. This file has the payload code
// which performs the actual convolution, it goes together with a
// 'driver' program drv_convolve.cc which has main and calls the
// payload code.
// Here is a program which works best with goading, so -DUSE_HWY is
// actually detrimental! But we can use -DMULTI_SIMD_ISA to produce
// SIMD-ISA-specific variants of the payload code and dispatch to
// them. This demonstrates that use of highway's CPU detection and
// dispatch mechanism can actually help 'ordinary' code, because
// autovectorization will still occur and produce machine code for
// the given architecture.

#include <iostream>
#include <vector>

// if the code is compiled to use the Vc or std::simd back-ends, we
// can't (yet) use highway's foreach_target mechanism, so we #undef
// MULTI_SIMD_ISA, which is zimt's way of activating that mechanism.

#if defined MULTI_SIMD_ISA && ( defined USE_VC || defined USE_STDSIMD )
#warning "un-defining MULTI_SIMD_ISA due to use of Vc or std::simd"
#undef MULTI_SIMD_ISA
#endif

// the payload function receives zimt data types. We #include the
// zimt headers needed to declare the arguments - these headers
// are independent of the SIMD ISA.

#include "../zimt/simd/simd_tag.h"
#include "../zimt/array.h"
#include "../zimt/xel.h"

// we silently assume we have a colour image

typedef zimt::xel_t < float , 3 > pixel_type; 

// target_type is a 2D array of pixels  

typedef zimt::array_t < 2 , pixel_type > target_type ;

// we define a dispatch base class. All tye 'payload' code is called
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

  zimt::backend_e backend = zimt::NBACKENDS ;
  unsigned long hwy_isa = 0 ;

  // next we have pure virtual member function definitions for
  // payload code. In this example, we only have one payload
  // function:

  virtual int payload ( zimt::view_t < 2 , pixel_type > & img ,
                        const std::vector < zimt::xlf_type > & kernel )
                      const = 0 ;
} ;

#endif

#ifdef MULTI_SIMD_ISA

// if we're using MULTI_SIMD_ISA, we have to define HWY_TARGET_INCLUDE
// to tell the foreach_target mechanism which file should be repeatedly
// re-included and re-copmpiled with SIMD-ISA-specific flags

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "convolve.cc"  // this file

#include <hwy/foreach_target.h>  // must come before highway.h
#include <hwy/highway.h>

#endif

// now we #include the zimt headers we need:

#include "../zimt/convolve.h"

// to make highway's use of #pragme directives to the compiler
// effective, we surround the SIMD-ISA-specific code with
// HWY_BEFORE_NAMESPACE() and HWY_AFTER_NAMESPACE().

HWY_BEFORE_NAMESPACE() ;

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

// to dispatch to the local _payload function, we call it from a
// member function 'payload' in class dispatch. Another possibility
// would have been to inline the code here, but I think it's
// clearer this way.

int _payload ( zimt::view_t < 2 , pixel_type > & img ,
               const std::vector < zimt::xlf_type > & kernel )
{
  zimt::convolve
  ( img ,
    img ,
    { zimt::MIRROR , zimt::MIRROR } ,
    kernel ,
    kernel.size() / 2 ) ;
  return 0 ;
}

struct dispatch
: public dispatch_base
{
  // We fit the derived dispatch class with a c'tor which fills in
  // information about the nested SIMD ISA we're currently in.

  dispatch()
  {
    backend = zimt::simdized_type<int,4>::backend ;
    #if defined USE_HWY || defined MULTI_SIMD_ISA
      hwy_isa = HWY_TARGET ;
    #endif
  }

  int payload ( zimt::view_t < 2 , pixel_type > & img ,
                const std::vector < zimt::xlf_type > & kernel ) const
  {
    return _payload ( img , kernel ) ;
  }
} ;

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

#else

// if we're not using highway's foreach_target mechanism, there is
// only a single _get_dispatch variant in namespace project::zsimd.
// So we call that one, to receive the desired dispatch_base pointer.

const dispatch_base * const get_dispatch()
{
  return zsimd::_get_dispatch() ;
}

#endif

// we now have get_dispatch, which will yield a dispatch_base pointer.
// Now we can code project::payload. We obtain the disptach_base pointer.
// the object it points to has a virtual member 'payload', which we
// invoke:

int payload ( zimt::view_t < 2 , pixel_type > & img ,
              const std::vector < zimt::xlf_type > & kernel )
{
  auto dp = get_dispatch() ;
  std::cout << "obtained dispatch pointer " << dp << std::endl ;
  std::cout << "dispatching to back-end   "
            << zimt::backend_name [ dp->backend ] << std::endl ;
#if defined USE_HWY || defined MULTI_SIMD_ISA
  std::cout << "dispatch hwy_isa is       "
            << hwy::TargetName ( dp->hwy_isa ) << std::endl ;
#endif

  int success ;

  // apply the filter several times to get a better time
  // measurement - just once is quite fast.

  for ( int times = 0 ; times < 1  ; times++ )
    success = dp->payload ( img , kernel ) ;

  return success ;
}

}  // namespace project

#endif  // HWY_ONCE

