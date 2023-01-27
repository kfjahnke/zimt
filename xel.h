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


/*! \file simd_type.h

    \brief arithmetic container type
*/

#ifndef VSPLINE_XEL_H
#define VSPLINE_XEL_H

#include <cmath>
#include <limits>
#include <assert.h>
#include <iostream>
#include <initializer_list>

namespace zimt
{
/// class template xel_t provides a fixed-size container type for
/// small sets of fundamentals or SIMD data types which are stored
/// in a C vector.
/// The type offers arithmetic capabilities which are implemented by
/// using loops over the elements in the vector. We use the same
/// 'innards' as for class simd_type, so our type inherits SIMDish
/// capabilities like load/store and gather/scatter, and masking.
/// While this type's semantic slot is more for 'xel' data - stuff
/// like pixels - whose channels are semantically different (like,
/// R, G, B) the SIMDish capabilities do no harm, and because we have
/// a completely separate class definition with a specific name, this
/// type has no relation to class simd_type at all, even though the
/// inner workings are the same. This is useful when we want to
/// use both xel_t and simd_type together to form 'simdized' xel
/// types: we create them as xel of a simd-capable type (which may
/// be simd_type or some other such type).

#define XEL xel_t

template < typename _value_type ,
           std::size_t _vsize >
class XEL
{
#include "xel_inner.h"

// assignment from equally-sized container.
// Note that the rhs can use any elementary type which can be legally
// assigned to value_type. This allows transport of information from
// differently typed objects, but there are no further constraints on
// the types involved, which may degrade precision. It's the user's
// responsibility to make sure such assignments have the desired effect
// and overload them if necessary.

template < typename U , template < typename , std::size_t > class V >
XEL & operator= ( const V < U , vsize > & rhs )
{
  for ( size_type i = 0 ; i < vsize ; i++ )
    (*this) [ i ] = rhs [ i ] ;
  return *this ;
}

template < typename U , template < typename , std::size_t > class V >
XEL ( const V < U , vsize > & ini )
{
  *this = ini ;
}

// 'projection' to some other fixed-size aggregate via a templated
// conversion operator: converts a xel_t to an object of class C
// which holds vsize value_type. We want to use xel_t as all-purpose
// fixed-size aggregate and still allow user code which assigns
// xel_t values to, say, std::arrays when they are returned by
// functions.

template < template < typename , std::size_t > class C >
operator C < value_type , vsize > ()
{
  C < value_type , vsize > result ;
  for ( size_type i = 0 ; i < vsize ; i++ )
    result [ i ] = _store [ i ] ;
  return result ;
}

value_type prod() const
{
  value_type s ( _store[0] ) ;
  for ( std::size_t e = 1 ; e < vsize ; e++ )
    s *= _store[e] ;
  return s ;
}

bool operator== ( const xel_t < value_type , vsize > rhs ) const
{
  for ( std::size_t i = 0 ; i < vsize ; i++ )
  {
    if ( _store[i] != rhs[i] )
      return false ;
  }
  return true ;
}

bool operator!= ( const xel_t < value_type , vsize > rhs ) const
{
  return ! ( (*this) == rhs ) ;
}

} ;

#undef XEL

} ;

#endif // #define VSPLINE_XEL_H

