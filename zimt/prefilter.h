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

/*! \file prefilter.h

    \brief Code to create the coefficient array for a b-spline.

    This is a port from vspline/prefilter.h

    Note: the bulk of the code was factored out to filter.h, while this text still
    outlines the complete filtering process.
    
    B-spline coefficients can be generated in two ways (that I know of): the first
    is by solving a set of equations which encode the constraints of the spline.
    A good example of how this is done can be found in libeinspline. I term it
    the 'linear algebra approach'. In this implementation, I have chosen what I
    call the 'DSP approach'. In a nutshell, the DSP approach looks at the b-spline's
    reconstruction as a convolution of the coefficients with a specific kernel. This
    kernel acts as a low-pass filter. To counteract the effect of this filter and
    obtain the input signal from the convolution of the coefficients, a high-pass
    filter with the inverse transfer function to the low-pass is used. This high-pass
    has infinite support, but can still be calculated precisely within the bounds of
    the arithmetic precision the CPU offers, due to the properties it has.
    
    I recommend [CIT2000] for a formal explanation. At the core of my prefiltering
    routines there is code from Philippe Thevenaz' accompanying code to this paper,
    with slight modifications translating it to C++ and making it generic.
    The greater part of this file deals with 'generifying' the process and to
    employing multithreading and the CPU's vector units to gain speed.
    
    This code makes heavy use of vigra, which provides handling of multidimensional
    arrays and efficient handling of aggregate types - to only mention two of it's
    many qualities. Explicit vectorization is done with Vc, which allowed me to code
    the horizontal vectorization I use in a generic fashion. If Vc is not available,
    the code falls back to presenting the data so that autovectorization becomes
    very likely - a technique I call 'goading'.
    
    In another version of this code I used vigra's BSplineBase class to obtain prefilter
    poles. This required passing the spline degree/order as a template parameter. Doing it
    like this allows to make the Poles static members of the solver, but at the cost of
    type proliferation. Here I chose not to follow this path and pass the spline order as a
    parameter to the spline's constructor, thus reducing the number of solver specializations
    and allowing automated testing with loops over the degree. This variant may be slightly
    slower. The prefilter poles I use are precalculated externally with gsl/blas and polished
    in high precision to provide the most precise data possible. this avoids using
    vigra's polynomial root code which failed for high degrees when I used it.

    [CIT2000] Interpolation Revisited by Philippe Thévenaz, Member,IEEE, Thierry Blu, Member, IEEE, and Michael Unser, Fellow, IEEE in IEEE TRANSACTIONS ON MEDICAL IMAGING, VOL. 19, NO. 7, JULY 2000,
*/

// #ifndef ZIMT_PREFILTER_H
// #define ZIMT_PREFILTER_H

#include <limits>

#include "common.h"
#include "poles.h"
#include "recursive.h"

#if defined(ZIMT_PREFILTER_H) == defined(HWY_TARGET_TOGGLE)
  #ifdef ZIMT_PREFILTER_H
    #undef ZIMT_PREFILTER_H
  #else
    #define ZIMT_PREFILTER_H
  #endif

HWY_BEFORE_NAMESPACE() ;
BEGIN_ZIMT_SIMD_NAMESPACE(zimt)

using namespace std ;
// using namespace zimt ;

/// 'prefilter' handles b-spline prefiltering for the whole range of
/// acceptable input and output. It combines two bodies of code to
/// achieve this goal:
/// - 'wielding' code in filter.h, which is not specific to b-splines.
/// - the recursive filtering code in recursive.h used with b-spline poles
///
/// Note that vsize , the vectorization width, can be passed explicitly.
/// If Vc is in use and math_ele_type can be used with hardware
/// vectorization, the arithmetic will be done with Vc::SimdArrays
/// of the given size. Otherwise 'goading' will be used: the data are
/// presented in TinyVectors of vsize math_ele_type, hoping that the
/// compiler may autovectorize the operation.

// KFJ 2018-12-20 added default for math_ele_type, static_cast to
// int for bcv's dimension, default for 'tolerance' - so now the
// prototype matches that of the functions in general_filter.h

template < std::size_t dimension ,
           typename in_value_type ,
           typename out_value_type ,
           typename math_ele_type =
                    ET < PROMOTE ( in_value_type , out_value_type ) > ,
           size_t vsize =
                  vector_traits < math_ele_type > :: size
         >
void prefilter ( const
                 view_t
                   < dimension ,
                     in_value_type > & input ,
                 view_t
                   < dimension ,
                     out_value_type > & output ,
                 xel_t < bc_code , dimension > bcv ,
                 int degree ,
                 xlf_type tolerance
                  = std::numeric_limits < math_ele_type > :: epsilon(),
                 xlf_type boost = xlf_type ( 1 ) ,
                 int njobs = default_njobs )
{
  if ( degree <= 1 )
  {
    // if degree is <= 1, there is no filter to apply, but we may need
    // to apply 'boost' and/or copy input to output. We use 'amplify'
    // for the purpose, which multithreads the operation (if it is at
    // all necessary). I found this is (slightly) faster than doing the
    // job in a single thread - the process is mainly memory-bound, so
    // the gain is moderate.

    amplify < dimension , in_value_type , out_value_type , math_ele_type >
      ( input , output , math_ele_type ( boost ) ) ;

    return ;
  }
  
  std::vector < iir_filter_specs > vspecs ;
  
  // package the arguments to the filter; one set of arguments
  // per axis of the data

  auto poles = zimt_constants::precomputed_poles [ degree ] ;
  
  for ( int axis = 0 ; axis < dimension ; axis++ )
  {
    vspecs.push_back
      ( iir_filter_specs
        ( bcv [ axis ] , degree / 2 , poles , tolerance , 1 ) ) ;
  }
  
  // 'boost' is only applied to dimension 0, since it is meant to
  // affect the whole data set just once, not once per axis.

  vspecs [ 0 ] . boost = boost ;

  // KFJ 2018-05-08 with the automatic use of vectorization the
  // distinction whether math_ele_type is 'vectorizable' or not
  // is no longer needed: simdized_type will be a Vc::SimdArray
  // if possible, a simd_type otherwise.
  
  typedef recursive_filter < simdized_type ,
                             math_ele_type ,
                             vsize
                           > filter_type ;

  // now call the 'wielding' code in filter.h

    filter
    < in_value_type , out_value_type , dimension , filter_type > 
    ( input , output , vspecs ) ;
}

END_ZIMT_SIMD_NAMESPACE
HWY_AFTER_NAMESPACE() ;

#endif // sentinel
