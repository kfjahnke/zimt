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

    class template simd_t is designed to provide an interface similar
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

    Class zimt::simd_t is actually quite similar to vigra::TinyVector
    which also stores in a plain C array and provides arithmetic. But that
    type is quite complex, using CRTP with a base class, explicitly coding
    loop unrolling, catering for deficient compilers and using vigra's
    sophisticated type promotion mechanism. zimt::simd_t on the other
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
    TinyVectors of zimt::simd_t - or, more generally, small
    aggregates of vector data. Please see this header's comments for
    more detailed information.

    Note also that throughout zimt, there is almost no explicit use of
    zimt::simd_t. zimt picks appropriate SIMD data types with
    mechanisms 'one level up', coded in vector.h. vector.h checks if use
    of Vc is possible and whether Vc can vectorize a given type, and
    produces a 'simdized type', which you mustn't confuse with a simd_t.
*/

#ifndef ZIMT_SIMD_TYPE_H
#define ZIMT_SIMD_TYPE_H

#include <iostream>
#include <initializer_list>
#include "common.h"

// we'll include some headers with repetetive definitions where
// we use 'XEL' for the types which are elaborated

#define XEL simd_t

namespace zimt
{
/// class simd_type serves as fallback type to provide SIMD semantics
/// without explicit SIMD code. It can be used throughout when use of
/// the SIMD 'backends' is unwanted, or to 'fill the gap' where some
/// SIMD backends do not provide implementations for specific
/// combinations of value_type and vsize. The latter type of use is
/// handled by 'vector.h'. Using simd_t may well result in actual
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
/// whereas simd_t is a SIMDish vector of semantically equal
/// lanes whose only commonality is that they populate the same
/// vector (as long as we stick with 'horizontal vectorization').
/// Here, we define a class with xel functionality and add masking
/// which is essential in SIMD code, while it makes little sense in
/// a non-SIMD arithmetic type like xel_t.

// we start out with a class called simd_t for brevity, at the end of
// this header we'll introduce simd_type with a 'using' statement

template < typename T , std::size_t N >
struct simd_t
: public simd_tag < T , N , GOADING >
{
typedef zimt::simd_tag < T , N , GOADING > tag_t ;
using typename tag_t::value_type ;
using tag_t::vsize ;
using tag_t::backend ;

#include "xel_mask.h"
#include "xel_inner.h"

// binary operators (used to be in xel_inner.h)

// we use a simple scheme for type promotion: the promoted type
// of two values fed to a binary operator should be the same as
// the type we would receive when adding the two values.

#define PROMOTE(A,B)  \
XEL < decltype (   std::declval < A::value_type > () \
                 + std::declval < B > () ) , N >

// for simd_type, we accept only other simd_type and fundamentals
// as second operand. We code the three variants as templates, to
// impose the desired type restrictions and to avoid the pitfall
// of having arguments implicitly converted, which is prone to
// happen when coding non-template functions for the purpose.

#define OP_FUNC(OPFUNC,OP,CONSTRAINT) \
  template < typename RHST , \
             typename = typename std::enable_if \
                       < std::is_fundamental < RHST > :: value \
                       > :: type \
           > \
  PROMOTE(XEL,RHST) \
  OPFUNC ( XEL < RHST , N > rhs ) const \
  { \
    CONSTRAINT \
    PROMOTE(XEL,RHST) help ; \
    for ( size_type i = 0 ; i < N ; i++ ) \
      help [ i ] = (*this) [ i ] OP rhs [ i ] ; \
    return help ; \
  } \
  template < typename RHST , \
             typename = typename std::enable_if \
                       < std::is_fundamental < RHST > :: value \
                       > :: type \
           > \
  PROMOTE(XEL,RHST) \
  OPFUNC ( RHST rhs ) const \
  { \
    CONSTRAINT \
    PROMOTE(XEL,RHST) help ; \
    for ( size_type i = 0 ; i < N ; i++ ) \
      help [ i ] = (*this) [ i ] OP rhs ; \
    return help ; \
  } \
  template < typename LHST , \
             typename = typename std::enable_if \
                       < std::is_fundamental < LHST > :: value \
                       > :: type \
           > \
  friend PROMOTE(XEL,LHST) OPFUNC ( LHST lhs , XEL rhs ) \
  { \
    CONSTRAINT \
    PROMOTE(XEL,LHST) help ; \
    for ( size_type i = 0 ; i < N ; i++ ) \
      help [ i ] = lhs OP rhs [ i ] ; \
    return help ; \
  }

OP_FUNC(operator+,+,)
OP_FUNC(operator-,-,)
OP_FUNC(operator*,*,)
OP_FUNC(operator/,/,)

OP_FUNC(operator%,%,INTEGRAL_ONLY)
OP_FUNC(operator&,&,INTEGRAL_ONLY)
OP_FUNC(operator|,|,INTEGRAL_ONLY)
OP_FUNC(operator^,^,INTEGRAL_ONLY)
OP_FUNC(operator<<,<<,INTEGRAL_ONLY)
OP_FUNC(operator>>,>>,INTEGRAL_ONLY)

OP_FUNC(operator&&,&&,BOOL_ONLY)
OP_FUNC(operator||,||,BOOL_ONLY)

#undef OP_FUNC
#undef PROMOTE

// types used for masks and index vectors. In terms of 'true' SIMD
// arithmetics, these definitions may not be optimal - especially the
// definition of a mask as a simd_t of bool is questionable - one
// might consider using a bit field or a sufficiently large integral
// type. But using a simd_t of bool makes processing simple, in
// a way it's the 'generic' mask type, whereas SIMD masks used by
// the hardware are the truly 'exotic' types. The problem here is
// the way C++ encodes booleans - they are usually encoded as some
// smallish integral type, rather than a single bit.

// we define both the 'old school' and the 'camel case' variants

typedef simd_t < int , N > index_type ;

typedef simd_t < int , N > IndexType ;

// mimick Vc's IndexesFromZero. This function produces an index
// vector filled with indexes starting with zero.

static const index_type IndexesFromZero()
{
  typedef typename index_type::value_type IT ;
  static const IT ceiling = std::numeric_limits < IT > :: max() ;
  assert ( ( N - 1 ) <= std::size_t ( ceiling ) ) ;

  static const index_type ix ( index_type::iota() ) ;
  return ix ;
}

// variant which starts from a different starting point and optionally
// uses steps other than one.

static const index_type IndexesFrom ( std::size_t start ,
                                      std::size_t step = 1 )
{
  typedef typename index_type::value_type IT ;
  static const IT ceiling = std::numeric_limits < IT > :: max() ;
  assert ( start + ( N - 1 ) * step <= std::size_t ( ceiling ) ) ;

  return ( IndexesFromZero() * int(step) ) + int(start) ;
}

// memory access functions, which load and store vector data.
// We start out with functions transporting data from memory into
// the simd_t. Some of these operations have corresponding
// c'tors which use the member function to initialize (*this).

// load uses a simple loop, which is about as easy to recognize as
// an autovectorizable construct as it gets:

void load ( const value_type * const p_src )
{
  for ( size_type i = 0 ; i < N ; i++ )
    (*this) [ i ] = p_src [ i ] ;
}

// generic gather performs the gather operation using a loop.
// Rather than loading consecutive values, The offset from 'p_src'
// is looked up in 'indexes' for each vector element. We allow
// any old indexable type as index_type, not just 'index_type'
// defined above.

template < typename index_type >
void gather ( const value_type * const p_src ,
              const index_type & indexes )
{
  for ( size_type i = 0 ; i < N ; i++ )
    (*this) [ i ] = p_src [ indexes [ i ] ] ;
}

// c'tor from pointer and indexes, uses gather

template < typename index_type >
simd_t ( const value_type * const p_src ,
            const index_type & indexes )
{
  gather ( p_src , indexes ) ;
}

// store saves the content of the container to memory

void store ( value_type * const p_trg ) const
{
  for ( size_type i = 0 ; i < N ; i++ )
    p_trg [ i ] = (*this) [ i ] ;
}

// scatter is the reverse operation to gather, see the comments there.

template < typename index_type >
void scatter ( value_type * const p_trg ,
                const index_type & indexes ) const
{
  for ( size_type i = 0 ; i < N ; i++ )
    p_trg [ indexes [ i ] ] = (*this) [ i ] ;
}

// 'regular' gather and scatter, accessing strided memory so that the
// first address visited is p_src/p_trg, and successive addresses are
// 'step' apart - in units of T. Might also be done with goading, the
// loop should autovectorize.

void rgather ( const value_type * const p_src ,
                const int & step )
{
  index_type indexes ( IndexesFrom ( 0 , step ) ) ;
  gather ( p_src , indexes ) ;
}

void rscatter ( value_type * p_trg ,
                const int & step ) const
{
  index_type indexes ( IndexesFrom ( 0 , step ) ) ;
  scatter ( p_trg , indexes ) ;
}

// use 'indexes' to perform a gather from the data held in '(*this)'
// and return the result of the gather operation.

template < typename index_type >
simd_t shuffle ( index_type indexes )
{
  simd_t result ;
  for ( size_type i = 0 ; i < N ; i++ )
    result [ i ] = (*this) [ indexes [ i ] ] ;
  return result ;
}

// operator[] with an index_type argument performs the same
// operation

simd_t operator[] ( index_type indexes )
{
  return shuffle ( indexes ) ;
}

// assigment and c'tor from another simd_t with equal vsize

template < typename U >
simd_t & operator= ( const simd_t < U , vsize > & rhs )
{
  for ( size_type i = 0 ; i < vsize ; i++ )
    (*this) [ i ] = rhs [ i ] ;
  return *this ;
}

template < typename U >
simd_t ( const simd_t < U , vsize > & rhs )
{
  *this = rhs ;
}

// broadcasting functions processing single value_type

typedef std::function < value_type() > gen_f ;
typedef std::function < value_type ( const value_type & ) > mod_f ;
typedef std::function < value_type ( const value_type & , const value_type & ) > bin_f ;

simd_t & broadcast ( gen_f f )
{
  for ( std::size_t i = 0 ; i < size() ; i++ )
  {
    (*this)[i] = f() ;
  }
  return *this ;
}

simd_t & broadcast ( mod_f f )
{
  for ( std::size_t i = 0 ; i < size() ; i++ )
  {
    (*this)[i] = f ( (*this)[i] ) ;
  }
  return *this ;
}

simd_t & broadcast ( bin_f f , const simd_t & rhs )
{
  for ( std::size_t i = 0 ; i < size() ; i++ )
  {
    (*this)[i] = f ( (*this)[i] , rhs[i] ) ;
  }
  return *this ;
}

} ;

// reductions for masks. It's often necessary to determine whether
// a mask is completely full or empty, or has at least some non-false
// members. The code was extended to test arbitrary vectors rather
// than only masks.

template < typename P , std::size_t vsize >
bool any_of ( simd_t < P , vsize > arg )
{
  bool result = false ;
  for ( std::size_t i = 0 ; i < vsize ; i++ )
    result = result || arg [ i ] ;
  return result ;
}

template < typename P , std::size_t vsize >
bool all_of ( simd_t < P , vsize > arg )
{
  bool result = true ;
  for ( std::size_t i = 0 ; i < vsize ; i++ )
    result = result && arg [ i ] ;
  return result ;
}

template < typename P , std::size_t vsize >
bool none_of ( simd_t < P , vsize > arg )
{
  bool result = true ;
  for ( std::size_t i = 0 ; i < vsize ; i++ )
    result = result && ( ! arg [ i ] ) ;
  return result ;
}

template < typename T , std::size_t N >
using simd_type = simd_t < T , N > ;

} ;

template < typename T , std::size_t N >
struct std::allocator_traits < zimt::simd_t < T , N > >
{
  typedef std::allocator < zimt::simd_t < T , N > > type ;
} ;

#undef XEL

#endif // #define ZIMT_SIMD_TYPE_H