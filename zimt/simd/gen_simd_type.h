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

    zimt can use highway for explicit vectorization, and at the time of
    this writing, this is usually the best option. But highway is not
    available everywhere, or it's use may be unwanted. To help with such
    situations, zimt defines it's own 'SIMD' type, which is implemented
    as a simple    C vector and small loops operating on it. If these
    constructs are compiled with compilers capable of autovectorization
    (and with the relevant flags activating use of SIMD instruction sets
    like AVX2) the resulting code will oftentimes be 'proper' SIMD code,
    because the small loops are presented so that the compiler can easily
    recognize them as potential clients of loop vectorization. I call this
    technique 'goading': By presenting the data flow in deliberately
    vector-friendly format, the compiler is more likely to 'get it'.
    Of course, to emit SIMD-ISA-specific binary from auto-vectorized
    code, the compiler will need to be instructed to do so - by default,
    it would only allow for a very moderate level of SIMD - like SSE2
    in i86 CPUs - so as not to produce instructions which older CPUs
    can't execute. This produces what I call the 'SIMD dilemma': if
    programmers want SIMD binary, their code may not run on older CPUs,
    and if they make binary to run everywhere, it won't exploit newer
    CPUs. To deal with this dilemma, the 'done thing' is to provide
    binary variants for each CPU which is likely to run the code and
    dispatch to the right version at runtime after detecting which CPU
    is currently being used. This can be done 'manually' (currently I
    do so in lux), but this is quite verbose and requires additional
    'external' code (in cmake, for linking) to work out properly.
    highway provides an automatic mechanism, which is very sophisticated
    and not very intrusive - it only requires that code using it carries
    a special sentinel and lends itself to being re-included several
    times by highway's foreach_target mechanism. My initial implementation
    of zimt did not use this mechanism, but I am changing zimt to allow
    it's use, and in fact I will work towards making it the default,
    because it works so well. The disadvantage is bloat and longer
    turnaround time, so zimt code is made to run without highway's
    foreach_target mechanism - the switch only requires a single
    #define which can be supplied during compilation. This can be used
    to work the code until it's bug-free and functional with quick
    turnaround, adding the multi-SIMD-ISA dispatch later on when
    approaching deployment.

    highway's foreach_target mechanism will work with any 'client'
    code - the code doesn't have to use highway's functions or data
    types at all. In this header, we define a data type which offers
    the 'standard' SIMD interface of 'simdized' zimt objects, but
    does rely on a C array and small loops - and autovectorization.
    These types can benefit greatly from highway's foreach_target
    mechanism, because with the appropriate internal dispatch, the
    compiler's autovectorization can actually emit specific SIMD
    binary instructions for every supported SIMD ISA. So when using
    gen_simd_type, #defining MULTI_SIMD_ISA makes sense and results
    in a binary which automagically adapts to the host CPU. But using
    hwy_simd_type will usually perform better, because it uses
    explicit vectorization, in contrast to relying on autovectorization.
    Most code will therefore use hwy_simd_type, but using gen_simd_type
    instead can provide a reference to compare the highway-derived
    code to, to see if autovectorization may be just as good - or,
    unexpectedly, even better for a given task. gen_simd_type also
    has slightly larger scope (the number of lanes doesn't have to
    be a power of two) and serves as fallback for lane types which
    highway or the other SIMD back-ends won't handle.

    class template gen_simd_type is designed to provide an interface similar
    to Vc::SimdArray, to be able to use it as a drop-in replacement.
    It aims to provide those SIMD capabilities which are actually used by
    zimt and is *not* a complete replacement for Vc::SimdArray, but
    this is where it originated (I had to preserve a large-ish body of
    Vc-based code and I also like Vc's object-oriented approach).

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
    not to be auto-vectorized into binary gather commands - this is my
    guess, I have not investigated the binary closely.

    The code presented here adds some memory access functions which are
    not present in Vc::SimdArray, namely strided load/store operations
    and load/store using functors.

    Note that I use clang++ most of the time, and the code has evolved to
    produce fast binary with clang++. Your mileage will vary with other
    compilers.

    Class zimt::gen_simd_type is actually quite similar to vigra::TinyVector
    which also stores in a plain C array and provides arithmetic. But that
    type is quite complex, using CRTP with a base class, explicitly coding
    loop unrolling, catering for deficient compilers and using vigra's
    sophisticated type promotion mechanism. zimt::gen_simd_type on the other
    hand is stripped down to the bare essentials, to make the code as simple
    as possible, in the hope that 'goading' will indeed work.

    One word of warning: the lack of type promotion requires you to pick
    a value_type of sufficient precision and capacity for the intended
    operation. In other words: you won't get an int when multiplying two
    shorts.

    Note also that this type is intended for *horizontal* vectorization,
    and you'll get the best results when picking a vector size which is
    a small-ish power of two - preferably at least the number of values
    of the given value_type which a register of the intended vector ISA
    will contain.

    zimt uses zimt::vec_t of SIMD data types, but their operations are
    coded with loops over the vec_t's elements throughout zimt's
    code base. In zimt's opt directory, you can find 'xel_of_vector.h',
    which can provide overloads for all operator functions involving
    xel_t of zimt::gen_simd_type - or, more generally, small
    aggregates of vector data. Please see this header's comments for
    more detailed information. This is work in progress, using
    xel_t of zimt SIMD data types should evolve to work 'out of the
    box' without the need for additional headers, so you may not
    need this header at all.

    Note also that throughout zimt, there is almost no explicit use of
    zimt::gen_simd_type. zimt picks appropriate SIMD data types with
    mechanisms 'one level up', coded in vector.h.
*/

// in a hwy context, what's the point of providing gen_simd_type?
// Even if gen_simd_type does not actually use any of the highway SIMD code
// 'proper', highway's foreach_target mechanism will 'roll out' the code to
// a set of 'incarnations' which use specific SIMD ISAs - all which highway
// supports for the given CPU. Then highway can use dynamic dispatch to
// route to the incarnation which *fits the CPU the code is run on*.
// So on every machine, we'll run the machine code which would have been
// produced by compiling with the 'normal' gen_simd_type.h and the compiler
// flags for the current CPU's SIMD ISA. Since gen_simd_type is quite
// 'transparent', as long as the code is 'boring' enough (so that it
// autovectorizes properly), we'll see SIMD benefits, rather than having to
// contend ourselves with some lowest common denominator by omitting SIMD
// ISA-specific compiler flags or risking illegal instructions by allowing
// a specific SIMD ISA for the binary which the current CPU may not compute.
// In fact highway's foreach_target mechanism can be used to leverage
// autovectorization properly, internalizing all the hard CPU-checking and
// dispatching, even if no actual highway SIMD code is used, so it's a
// reasonably painless first step towards vectorizing a program which does
// not use zimt or highway SIMD: having autovectorized code for the CPU
// running the program is definitely a step up from autovectorized code
// only for one specific SIMD ISA fixed at compile time. But note that
// using highway's foreach_target mechanism also requires #defining
// MULTI_SIMD_ISA and linking with libhwy.´

#if defined(GEN_SIMD_TYPE_H) == defined(HWY_TARGET_TOGGLE)
  #ifdef GEN_SIMD_TYPE_H
    #undef GEN_SIMD_TYPE_H
  #else
    #define GEN_SIMD_TYPE_H
  #endif

#include <iostream>
#include <initializer_list>
#include "../common.h"
#include "simd_tag.h"

HWY_BEFORE_NAMESPACE() ;
BEGIN_ZIMT_SIMD_NAMESPACE(zimt)

// namespace hn = hwy::HWY_NAMESPACE ;

// using namespace simd ;

/// class gen_simd_type serves as fallback type to provide SIMD semantics
/// without explicit SIMD code. It can be used throughout when use of
/// the SIMD 'backends' is unwanted, or to 'fill the gap' where some
/// SIMD backends do not provide implementations for specific
/// combinations of value_type and vsize. The latter type of use is
/// handled by 'vector.h'. Using gen_simd_type may well result in actual
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
/// whereas gen_simd_type is a SIMDish vector of semantically equal
/// lanes whose only commonality is that they populate the same
/// vector (as long as we stick with 'horizontal vectorization').
/// Here, we define a class with xel functionality and add masking
/// which is essential in SIMD code, while it makes little sense in
/// a non-SIMD arithmetic type like xel_t.

// we start out with a class called gen_simd_type for brevity, at the end of
// this header we'll introduce simd_type with a 'using' statement

template < typename T , std::size_t N >
struct gen_simd_type
: public simd_tag < T , N , GOADING >
{
typedef simd_tag < T , N , GOADING > tag_t ;
using typename tag_t::value_type ;
using tag_t::vsize ;
using tag_t::backend ;

#define XEL gen_simd_type

#include "vector_mask.h"
#include "vector_common.h"

template < typename U ,
           template < typename , std::size_t > class X >
gen_simd_type ( const X < U , N > & rhs )
{
  for ( size_type i = 0 ; i < N ; i++ )
    (*this) [ i ] = T ( rhs [ i ] ) ;
}

// binary operators (used to be in xel_inner.h)

#define INTEGRAL_ONLY \
  static_assert ( is_integral < value_type > :: value , \
                  "this operation is only allowed for integral types" ) ;

#define BOOL_ONLY \
  static_assert ( std::is_same < value_type , bool > :: value , \
                  "this operation is only allowed for booleans" ) ;

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
  XEL < PROMOTE ( T , RHST ) , N > \
  OPFUNC ( XEL < RHST , N > rhs ) const \
  { \
    CONSTRAINT \
    XEL < PROMOTE ( T , RHST ) , N > help ; \
    for ( size_type i = 0 ; i < N ; i++ ) \
      help [ i ] = (*this) [ i ] OP rhs [ i ] ; \
    return help ; \
  } \
  template < typename RHST , \
             typename = typename std::enable_if \
                       < std::is_fundamental < RHST > :: value \
                       > :: type \
           > \
  XEL < PROMOTE ( T , RHST ) , N > \
  OPFUNC ( RHST rhs ) const \
  { \
    CONSTRAINT \
    XEL < PROMOTE ( T , RHST ) , N > help ; \
    for ( size_type i = 0 ; i < N ; i++ ) \
      help [ i ] = (*this) [ i ] OP rhs ; \
    return help ; \
  } \
  template < typename LHST , \
             typename = typename std::enable_if \
                       < std::is_fundamental < LHST > :: value \
                       > :: type \
           > \
  friend XEL < PROMOTE ( LHST , T ) , N > \
  OPFUNC ( LHST lhs , XEL rhs ) \
  { \
    CONSTRAINT \
    XEL < PROMOTE ( LHST , T ) , N > help ; \
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

// OP_FUNC(operator&&,&&,BOOL_ONLY)
// OP_FUNC(operator||,||,BOOL_ONLY)
OP_FUNC(operator&&,&&,)
OP_FUNC(operator||,||,)

#undef OP_FUNC
#undef INTEGRAL_ONLY
#undef BOOL_ONLY

// types used for masks and index vectors. In terms of 'true' SIMD
// arithmetics, these definitions may not be optimal - especially the
// definition of a mask as a gen_simd_type of bool is questionable - one
// might consider using a bit field or a sufficiently large integral
// type. But using a gen_simd_type of bool makes processing simple, in
// a way it's the 'generic' mask type, whereas SIMD masks used by
// the hardware are the truly 'exotic' types. The problem here is
// the way C++ encodes booleans - they are usually encoded as some
// smallish integral type, rather than a single bit.

// we define both the 'old school' and the 'camel case' variants

typedef gen_simd_type < int , N > index_type ;

typedef gen_simd_type < int , N > IndexType ;

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
// the gen_simd_type. Some of these operations have corresponding
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
gen_simd_type ( const value_type * const p_src ,
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

void rgather ( const value_type * const & p_src ,
                const std::size_t & step )
{
  if ( step == 1 )
  {
    load ( p_src ) ;
  }
  else
  {
    auto indexes = IndexesFrom ( 0 , step ) ;
    gather ( p_src , indexes ) ;
  }
}

void rscatter ( value_type * const & p_trg ,
                const std::size_t & step ) const
{
  if ( step == 1 )
  {
    store ( p_trg ) ;
  }
  else
  {
    auto indexes = IndexesFrom ( 0 , step ) ;
    scatter ( p_trg , indexes ) ;
  }
}

// use 'indexes' to perform a gather from the data held in '(*this)'
// and return the result of the gather operation.

template < typename index_type >
gen_simd_type shuffle ( index_type indexes )
{
  gen_simd_type result ;
  for ( size_type i = 0 ; i < N ; i++ )
    result [ i ] = (*this) [ indexes [ i ] ] ;
  return result ;
}

// operator[] with an index_type argument performs the same
// operation

gen_simd_type operator[] ( index_type indexes )
{
  return shuffle ( indexes ) ;
}

// assigment and c'tor from another gen_simd_type with equal vsize

template < typename U >
gen_simd_type & operator= ( const gen_simd_type < U , vsize > & rhs )
{
  for ( size_type i = 0 ; i < vsize ; i++ )
    (*this) [ i ] = rhs [ i ] ;
  return *this ;
}

template < typename U >
gen_simd_type ( const gen_simd_type < U , vsize > & rhs )
{
  *this = rhs ;
}

// broadcasting functions processing single value_type

typedef std::function < value_type() > gen_f ;
typedef std::function < value_type ( const value_type & ) > mod_f ;
typedef std::function < value_type ( const value_type & , const value_type & ) > bin_f ;

gen_simd_type & broadcast ( gen_f f )
{
  for ( std::size_t i = 0 ; i < size() ; i++ )
  {
    (*this)[i] = f() ;
  }
  return *this ;
}

gen_simd_type & broadcast ( mod_f f )
{
  for ( std::size_t i = 0 ; i < size() ; i++ )
  {
    (*this)[i] = f ( (*this)[i] ) ;
  }
  return *this ;
}

gen_simd_type & broadcast ( bin_f f , const gen_simd_type & rhs )
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
bool any_of ( gen_simd_type < P , vsize > arg )
{
  bool result = false ;
  for ( std::size_t i = 0 ; i < vsize ; i++ )
    result = result || arg [ i ] ;
  return result ;
}

template < typename P , std::size_t vsize >
bool all_of ( gen_simd_type < P , vsize > arg )
{
  bool result = true ;
  for ( std::size_t i = 0 ; i < vsize ; i++ )
    result = result && arg [ i ] ;
  return result ;
}

template < typename P , std::size_t vsize >
bool none_of ( gen_simd_type < P , vsize > arg )
{
  bool result = true ;
  for ( std::size_t i = 0 ; i < vsize ; i++ )
    result = result && ( ! arg [ i ] ) ;
  return result ;
}

#undef XEL

END_ZIMT_SIMD_NAMESPACE
HWY_AFTER_NAMESPACE() ;

namespace zimt
{
  template < typename T , size_t N >
  struct is_integral < zimt::gen_simd_type < T , N > >
  : public std::is_integral < T >
  { } ;
} ;

#endif // #define GEN_SIMD_TYPE_H
