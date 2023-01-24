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

    \brief SIMD type using small loops

    zimt can use Vc for explicit vectorization, and at the time of
    this writing, this is usually the best option. But Vc is not available
    everywhere, or it's use may be unwanted. To help with such situations,
    zimt defines it's own 'SIMD' type, which is implemented as a simple
    C vector and small loops operating on it. If these constructs are
    compiled with compilers capable of autovectorization (and with the
    relevent flags activating use of SIMD instruction sets like AVX)
    the resulting code will oftentimes be 'proper' SIMD code, because
    the small loops are presented so that the compiler can easily recognize
    them as potential clients of loop vectorization. I call this technique
    'goading': By presenting the data flow in deliberately vector-friendly
    format, the compiler is more likely to 'get it'.

    class template simd_type is designed to provide an interface similar
    to Vc::SimdArray, to be able to use it as a drop-in replacement.
    It aims to provide those SIMD capabilities which are actually used by
    zimt and is *not* a complete replacement for Vc::SimdArray.

    Wherever possible, the code is as simple as possible, avoiding frills
    and trickery which might keep the compiler from recognizing potentially
    auto-vectorizable constructs. The resulting code is - in my limited
    experience - often not too far from explicit SIMD code. Some constructs
    do actually produce binary which is en par with code using Vc, namely
    such code which does not use gather, scatter or masked operations.
    So b-spline prefiltering, restoration of original data, and general
    filtering is very fast, while code involving b-spline evaluation
    shows a speed penalty, since vectorized b-spline evaluation (as coded
    in zimt) relies massively on gather operations of a kind which seem
    not to be auto-vectorized into binary gather commands - this is my guess,
    I have not investigated the binary closely.

    The code presented here adds some memory access functions which are
    not present in Vc::SimdArray, namely strided load/store operations
    and load/store using functors.

    Note that I use clang++ most of the time, and the code has evolved to
    produce fast binary with clang++. Your mileage will vary with other
    compilers.

    Class zimt::simd_type is actually quite similar to vigra::TinyVector
    which also stores in a plain C array and provides arithmetic. But that
    type is quite complex, using CRTP with a base class, explicitly coding
    loop unrolling, catering for deficient compilers and using vigra's
    sophisticated type promotion mechanism. zimt::simd_type on the other
    hand is stripped down to the bare essentials, to make the code as simple
    as possible, in the hope that 'goading' will indeed work. It replaces
    zimt's previous SIMD type, zimt::simd_tv, which was derived
    from vigra::TinyVector.

    One word of warning: the lack of type promotion requires you to pick
    a value_type of sufficient precision and capacity for the intended
    operation. In other words: you won't get an int when multiplying two
    shorts.

    Note also that this type is intended for *horizontal* vectorization,
    and you'll get the best results when picking a vector size which is
    a small-ish power of two - preferably at least the number of values
    of the given value_type which a register of the intended vector ISA
    will contain.

    zimt uses TinyVectors of SIMD data types, but their operations are
    coded with loops over the TinyVector's elements throughout zimt's
    code base. In zimt's opt directory, you can find 'xel_of_vector.h',
    which can provide overloads for all operator functions involving
    TinyVectors of zimt::simd_type - or, more generally, small
    aggregates of vector data. Please see this header's comments for
    more detailed information.

    Note also that throughout zimt, there is almost no explicit use of
    zimt::simd_type. zimt picks appropriate SIMD data types with
    mechanisms 'one level up', coded in vector.h. vector.h checks if use
    of Vc is possible and whether Vc can vectorize a given type, and
    produces a 'simdized type', which you mustn't confuse with a simd_type.
*/

#ifndef VSPLINE_SIMD_TYPE_H
#define VSPLINE_SIMD_TYPE_H

#include <iostream>
#include <initializer_list>

namespace zimt
{

/// class simd_type serves as fallback type to provide SIMD semantics
/// without explicit SIMD code. It can be used throughout when use of
/// the SIMD 'backends' is unwanted, or to 'fill the gap' where some
/// SIMD backends do not provide implementations for specific
/// combinations of value_type and vsize. The latter type of use is
/// handled by 'vector.h'. Using simd_type may well result in actual
/// SIMD instructions being issued by the compiler due to
/// autovectorization - the data are presented in small loops which
/// are deliberately autovectorization-friendly - a technique I call
/// 'goading'.
/// The type is generated by filling an 'empty class shell' with the
/// code in xel_XXX.h - this unusual construct is chosen to make
/// it easy to construct carbon copy class templates with the same
/// functionality but different name - without having to deal with
/// inheritance, CRTP or additional template arguments, but yielding
/// two totally independent types which aren't treated by the compiler
/// as being the same (std::is_same comes out false). This helps us
/// in putting the types into their specific sematic slot: xel_t is
/// used for xel-like aggregates of semantically different channels,
/// whereas simd_type is a SIMDish vector of semantically equal
/// lanes whose only commonality is that they populate the same
/// vector (as long as we stick with 'horizontal vectorization').
/// Here, we define a class with xel functionality and add masking
/// which is essential in SIMD code, while it makes little sense in
/// a non-SIMD arithmetic type like xel_t.

#define XEL simd_type

template < typename _value_type ,
           std::size_t _vsize >
class XEL
{
#include "xel_inner.h"

// assigment and c'tor from another simd_type with equal vsize

template < typename T >
simd_type & operator= ( const simd_type < T , vsize > & rhs )
{
  for ( size_type i = 0 ; i < vsize ; i++ )
    (*this) [ i ] = rhs [ i ] ;
  return *this ;
}

template < typename T >
simd_type ( const simd_type < T , vsize > & rhs )
{
  *this = rhs ;
}

#include "xel_mask.h"
} ;

#undef XEL

// reductions for masks. It's often necessary to determine whether
// a mask is completely full or empty, or has at least some non-false
// members. The code might be extended to test arbitrary vectors rather
// than only masks. As it stands, to apply the functions to an
// arbitrary vector, use a construct like 'any_of ( v == 0 )' instead of
// 'any_of ( v )'.

template < std::size_t vsize >
bool any_of ( simd_type < bool , vsize > arg )
{
  bool result = false ;
  for ( std::size_t i = 0 ; i < vsize ; i++ )
    result |= arg [ i ] ;
  return result ;
}

template < std::size_t vsize >
bool all_of ( simd_type < bool , vsize > arg )
{
  bool result = true ;
  for ( std::size_t i = 0 ; i < vsize ; i++ )
    result &= arg [ i ] ;
  return result ;
}

template < std::size_t vsize >
bool none_of ( simd_type < bool , vsize > arg )
{
  bool result = true ;
  for ( std::size_t i = 0 ; i < vsize ; i++ )
    result &= ( ! arg [ i ] ) ;
  return result ;
}

} ;

template < typename T , std::size_t N >
struct std::allocator_traits < zimt::simd_type < T , N > >
{
  typedef std::allocator < zimt::simd_type < T , N > > type ;
} ;

#endif // #define VSPLINE_SIMD_TYPE_H
