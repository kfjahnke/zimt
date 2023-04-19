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
#include <utility>
#include <stdexcept>

namespace zimt
{
  // forward declaration of xel_t

  template < typename T , std::size_t > struct xel_t ;

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

// we use a simple scheme for type promotion: the promoted type
// of two values should be the same as the type we would receive
// when adding the two values. That's standard C semantics, but
// it won't widen the result type to avoid overflow or increase
// precision - such conversions have to be made by user code if
// necessary.

#define PROMOTE(A,B)  \
decltype ( std::declval < A > () + std::declval < B > () )

/// zimt creates views/arrays of vectorized types. As long as
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

// grok_t is the base class used to 'grok' functors. It uses
// 'C style' type erasure for the 'grokkee', but handles the
// grokkee with type-safe code.

struct grok_t
{
protected:

  // we need two types of function to provide infrastructure
  // of grok_t: the first one produces a copy of the 'grokkee'
  // and the second one deletes it. Note that the grokkee itself
  // is copied to allocated memory in the c'tor and subsequently
  // only referred to via a void* - this is a very 'C' way of
  // handling the type erasure, but since all access to the void
  // pointer happens within class grok_t and the classes which
  // inherit from it, there is no reason not to go for this
  // simple and straightforward approach. It's easy to understand
  // and seems to optimize well.

  std::function < void* ( void* ) > rep ;
  std::function < void ( void* ) > trm ;

  // and here we have the pointer to the copy of the grokkee
  // cast to void*.

  void * p_context ;

  // the 'normal' c'tor creates two lambdas, one to 'replicate'
  // the 'hidden' grokkee referenced by the void pointer, and one
  // to destruct it in a type-safe manner.

  template < typename grokkee_t >
  grok_t ( const grokkee_t & grokkee )
  {
     p_context = new grokkee_t ( grokkee ) ;

    // the std::functions are initialized with wrappers taking
    // p_context and a set of arguments which are passed on to
    // the grokkee's member functions. Note that we can't pass
    // p_context into the lambdas/std::functions: these will be
    // copied to copies of the grok_t object, the copies will
    // have their own context pointer, and this pointer must be
    // used by the std::functions when they are invoked.

    rep = [] ( void * p_ctx ) -> void*
          {
            auto p_gk = static_cast<grokkee_t*> ( p_ctx ) ;
            return new grokkee_t ( *p_gk ) ;
          } ;

    trm = [] ( void * p_ctx )
          {
            auto p_gk = static_cast<grokkee_t*> ( p_ctx ) ;
            delete p_gk ;
          } ;
  }

  // Copy assignment, and also the copy c'tor relying on it,
  // create a copy of the context object in a type-safe manner.
  // Because the std::functions receive the grokkee as their first
  // argument via p_context (which is unique to each grok_type
  // object) they can be copied from the rhs object as they are
  // - they hold no reference to any information specific to rhs.
  // The new context for the new grok_type object is provided by
  // calling 'rep'. One might assume that copying a grok_type
  // object should actually copy the context as well, but what
  // we want to copy is the functionality, whereas the context
  // object itself is only useful when it comes to functors
  // with state, like in reductions. If there are more of them
  // than strictly necessary, this is no problem, but there
  // mustn't be too few: if several threads were to access the
  // same context object concurrently, this would spell disaster.
  // The functors we're dealing with here are typically copied
  // so that each worker thread has it's own copy - see
  // zimt::process in wielding.h

  grok_t & operator= ( const grok_t & rhs )
  {
    // copy the std::functions

    rep = rhs.rep ;
    trm = rhs.trm ;

    // now use 'rep' to create a copy of the 'hidden' grokkee

    p_context = rep ( rhs.p_context ) ;
    return *this ;
  }

  // copy construction delegates to copy assignment

  grok_t ( const grok_t & rhs )
  {
    *this = rhs ;
  }

  // default c'tor - this is to create, like, vectors of grok_t
  // to assign to them later. Called after default construction,
  // this will crash.

  grok_t() { } ;

  // finally, the d'tor destroys the context object in a type-safe
  // manner by passing it to 'trm', which knows how to cast it to
  // it's 'true' type and then calls delete on that.

  ~grok_t()
  {
    if ( p_context )
      trm ( p_context ) ;
  }
} ;

} ; // end of namespace zimt

#endif // ZIMT_COMMON
