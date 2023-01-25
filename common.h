/************************************************************************/
/*                                                                      */
/*    zimt - abstraction layer for SIMD programming                     */
/*                                                                      */
/*            Copyright 2023 by Kay F. Jahnke                           */
/*                                                                      */
/*    The git repository for this software is at                        */
/*                                                                      */
/*    https://bitbucket.org/kfj/zimt                                    */
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

/*! \file common.h

    \brief definitions common to all files in this project, utility code
    
    This file contains
    
    - some common enums and strings
    
    - definition of a few utility types used throughout zimt
    
    - exceptions used throughout zimt
    
    It includes zimt/vector.h which defines zimt's use of
    vectorization (meaning SIMD operation) and associated types and code. 
*/

#ifndef ZIMT_COMMON
#define ZIMT_COMMON

#include <type_traits>
#include <limits>
#include <functional>
#include <assert.h>
#include <cmath>
#include "xel.h"

namespace zimt
{
  class UnsuitableTypeForExpandElements { } ;

  template < typename T >
  struct get_ele_t
  {
    static const size_t size = 1 ;
    typedef UnsuitableTypeForExpandElements type ;
  } ;

  template < typename T , std::size_t SZ >
  struct get_ele_t < zimt::xel_t < T , SZ > >
  {
    static const size_t size = SZ ;
    typedef T type ;
  } ;

  template < typename T , std::size_t SZ ,
             template < typename , std::size_t > class X >
  struct get_ele_t < X < T , SZ > >
  {
    static const size_t size = SZ ;
    typedef T type ;
  } ;

  template < typename T >
  struct element_expandable
  {
    static const bool value = false ;
  } ;

  template < typename T , std::size_t SZ >
  struct element_expandable < zimt::xel_t < T , SZ > >
  {
    static const bool value = true ;
  } ;

  // optimistic: if it's an X of SZ T, let's assume it's
  // element_expandable.

  template < typename T , std::size_t SZ ,
             template < typename , std::size_t > class X >
  struct element_expandable < X < T , SZ > >
  {
    static const bool value = true ;
  } ;

// why 'xlf'? it's for eXtra Large Float'. I use this for the 'most precise
// floating point type which can be used with standard maths' - currently
// this is long double, but in the future this might be quads or even better.
// I use this type where I have very precise constants (like in poles.h)
// which I try and preserve in as much precision as feasible before they
// end up being cast down to some lesser type.

typedef long double xlf_type ;

/// using definition for the 'elementary type' of a type via zimt's
/// get_ele_t mechanism. Since this is a frequently used idiom
/// in zimt but quite a mouthful, here's an abbreviation:

template < typename T >
using ET =
typename zimt::get_ele_t < T > :: type ;

/// produce a std::integral_constant from the size obtained from
/// zimt's get_ele_t mechanism

template < typename T >
using EN =
typename
std::integral_constant
  < int ,
    zimt::get_ele_t < T > :: size
  > :: type ;

/// is_element_expandable tests if a type T is known to zimt's
/// get_ele_t mechanism. If this is so, the type is
/// considered 'element-expandable'.
  
template < class T >
using is_element_expandable = typename
  std::integral_constant
  < bool ,
    ! std::is_same
      < typename zimt::get_ele_t < T > :: type ,
        zimt::UnsuitableTypeForExpandElements
      > :: value
  > :: type ;

/*
// for mathematics, we use a few using declarations. First we have
// promote_type, which yields a type an arithmetic operation between
// a T1 and a T2 should yield. We use zimt's PromoteTraits to obtain
// this type. Note how, by starting out with the formation of the
// Promote type, we silently enforce that T1 and T2 have equal channel
// count or are both fundamental types.

template < class T1 , class T2 >
using promote_type
      = typename zimt::PromoteTraits < T1 , T2 > :: Promote ;

// using zimt's RealPromote on the promote_type gives us a real type
// suitable for real arithmetics involving T1 and T2. We'll use this
// type in template argument lists where T1 and T2 are given already,
// as default value for 'math_type', which is often user-selectable in
// zimt. With this 'sensible' default, which is what one would
// 'usually' pick 'anyway', oftentimes the whole template argument
// list of a specific zimt function can be fixed by ATD, making it
// easy for the users: they merely have to pass, say, an input and output
// array (fixing T1 and T2), and the arithmetics will be done in a suitable
// type derived from these types. This allows users to stray from the
// 'normal' path (by, for example, stating they want the code to perform
// arithemtics in double precision even if T1 and T2 are merely float),
// but makes 'normal' use of the code concise and legible, requiring
// no explicit template arguments.

template < class T1 , class T2 >
using common_math_type
      = typename zimt::NumericTraits
                 < promote_type < T1 , T2 > > :: RealPromote ;

// while most of the T1, T2 we'll be dealing with will be small aggregates
// of fundametal types - like xel_ts of float - we also want a
// convenient way to get the fundamental type involved - or the 'ele_type'
// in zimt parlance. Again we use zimt to obtain this value. The
// elementary type can be obtained by using zimt's get_ele_t
// mechanism, which, as a traits class, can easily be extended to any
// type holding an aggregate of equally typed fundamentals. Note that we might
// start out with the elementary types of T1 and T2 and take their Promote,
// yielding the same resultant type. If promote_type<T1,T2> is fundamental,
// the get_ele_t mechanism passes it through unchanged.

template < class T1 , class T2 >
using promote_ele_type
      = typename zimt::get_ele_t
                 < promote_type < T1 , T2 > > :: type ;

// when doing arithmetic in zimt, typically we use real fundametal types
// which 'suit' the data types we want to put into the calculations. Again we
// use zimt to determine an apprpriate fundamental type by obtaining the
// RealPromote of the promote_ele_type. 

template < class T1 , class T2 >
using common_math_ele_type
      = typename zimt::NumericTraits
                 < promote_ele_type < T1 , T2 > > :: RealPromote ;
*/

/// produce the 'canonical type' for a given type T. If T is
/// single-channel, this will be the elementary type itself,
/// otherwise it's a xel_t of the elementary type.
/// optionally takes the number of elements the resulting
/// type should have, to allow construction from a fundamental
/// and a number of channels.
  
// zimt 'respects the singular'. this is to mean that throughout
// zimt I avoid using fixed-size containers holding a single value
// and use the single value itself. While this requires some type
// artistry in places, it makes using the code more natural. Only
// when it comes to providing generic code to handle data which may
// or may not be aggregates, zimt moves to using 'synthetic' types,
// which are always xel_ts, possibly with only one element.
// user code can pass data as canonical or synthetic types. All
// higher-level operations will produce the same type as output
// as they receive as input. The move to 'synthetic' types is only
// done internally to ease the writing of generic code.
// both the 'canonical' and the 'synthetic' type have lost all
// 'special' meaning of their components (like their meaning as
// the real and imaginary part of a complex number). zimt will
// process all types in it's scope as such 'neutral' aggregates of
// several identically-typed elementary values.

template < typename T , int N = EN < T > :: value >
using canonical_type =
typename
  std::conditional
  < N == 1 ,
    ET < T > ,
    zimt::xel_t < ET < T > , N >
  > :: type ;

template < typename T , int N = EN < T > :: value >
using synthetic_type =
zimt::xel_t < ET < T > , N > ;

template < class T , size_t sz = 1 >
struct invalid_scalar
{
  static_assert ( sz == 1 , "scalar values can only take size 1" ) ;
} ;

/// definition of a scalar with the same template argument list as
/// a simdized type, to use 'scalar' in the same syntactic slot

template < class T , size_t sz = 1 >
using scalar =
  typename std::conditional
    < sz == 1 ,
      T ,
      invalid_scalar < T , sz >
    > :: type ;

/// zimt creates zimt::MultiArrays of vectorized types. As long as
/// the vectorized types are Vc::SimdArray or zimt::simd_type, using
/// std::allocator is fine, but when using other types, using a specific
/// allocator may be necessary. Currently this is never the case, but I
/// have the lookup of allocator type from this traits class in place if
/// it should become necessary.

template < typename T >
struct allocator_traits
{
  typedef std::allocator < T > type ;
} ;

// TODO My use of exceptions is a bit sketchy...

/// for interfaces which need specific implementations we use:

struct not_implemented
: std::invalid_argument
{
  not_implemented ( const char * msg )
  : std::invalid_argument ( msg ) { }  ;
} ;

/// dimension-mismatch is thrown if two arrays have different dimensions
/// which should have the same dimensions.

struct dimension_mismatch
: std::invalid_argument
{
  dimension_mismatch ( const char * msg )
  : std::invalid_argument ( msg ) { }  ;
} ;

/// shape mismatch is the exception which is thrown if the shapes of
/// an input array and an output array do not match.

struct shape_mismatch
: std::invalid_argument
{
  shape_mismatch  ( const char * msg )
  : std::invalid_argument ( msg ) { }  ;
} ;

/// exception which is thrown if an opertion is requested which zimt
/// does not support

struct not_supported
: std::invalid_argument
{
  not_supported  ( const char * msg )
  : std::invalid_argument ( msg ) { }  ;
} ;

/// out_of_bounds is thrown by mapping mode REJECT for out-of-bounds coordinates
/// this exception is left without a message, it only has a very specific application,
/// and there it may be thrown often, so we don't want anything slowing it down.

struct out_of_bounds
{
} ;

/// exception which is thrown when assigning an rvalue which is larger than
/// what the lvalue can hold

struct numeric_overflow
: std::invalid_argument
{
  numeric_overflow  ( const char * msg )
  : std::invalid_argument ( msg ) { }  ;
} ;

} ; // end of namespace zimt

// with these common definitions done, we now include 'vector.h', which
// defines vectorization methods used throughout zimt.

#include "vector.h"

#ifdef USE_FMA

// 2019-02-13 tentative use of fma in zimt
// we need definitions of fma for fundamentals, zimt::xel_ts and
// zimt::simd_type. The least spcific template delegates to std::fma,
// then we have two more specific templates for the container types.

template < typename arg_t >
arg_t fma ( const arg_t & arg1 , const arg_t & arg2 , const arg_t & arg3 )
{
  return std::fma ( arg1 , arg2 , arg3 ) ;
}

template < typename T , int SZ >
zimt::xel_t < T , SZ > fma ( const zimt::xel_t < T , SZ > & arg1 ,
                                   const zimt::xel_t < T , SZ > & arg2 ,
                                   const zimt::xel_t < T , SZ > & arg3 )
{
  zimt::xel_t < T , SZ > result ;
  for ( size_t i = 0 ; i < SZ ; i++ )
    result[i] = fma ( arg1[i] , arg2[i] , arg3[i] ) ;
  return result ;
}

template < typename T , size_t SZ >
zimt::simd_type < T , SZ > fma ( const zimt::simd_type < T , SZ > & arg1 ,
                                  const zimt::simd_type < T , SZ > & arg2 ,
                                  const zimt::simd_type < T , SZ > & arg3 )
{
  zimt::simd_type < T , SZ > result ;
  for ( size_t i = 0 ; i < SZ ; i++ )
    result[i] = fma ( arg1[i] , arg2[i] , arg3[i] ) ;
  return result ;
}

#endif // USE_FMA

#ifndef WIELDING_SEGMENT_SIZE
#define WIELDING_SEGMENT_SIZE 0
#endif

#endif // ZIMT_COMMON
