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


/*! \file xel.h

    \brief arithmetic container type
*/

// zimt::xel_t is a 'classic' header - it doesn't have any code which
// varies with the SIMD ISA. So we use a normal sentinel construct.
// If code managed by foreach_target.h uses xel_t, that's no problem:
// the first inclusion establishes xel_t and it's capabilities,
// subsequent incarnations use the established definitions whose
// repeated definition is prevented by the sentinel.

#ifndef ZIMT_XEL_T_H
#define ZIMT_XEL_T_H

#include <cmath>
#include <limits>
#include <assert.h>
#include <iostream>
#include <initializer_list>
#include "common.h"
#include "vector.h"

#define XEL xel_t

// TODO: this has extensions to bunch and fluff used only for
// access to tiled storage

namespace zimt
{
template < typename T , std::size_t N >
struct xel_t ;

// all xel data are deemed integral if their value_type is.

// template < typename T >
// struct is_integral
// : public std::is_integral < T >
// { } ;

template < typename T , size_t N >
struct is_integral < xel_t < T , N > >
: public std::is_integral < T >
{ } ;

// 'form' is a traits class used to abstract from the concrete
// 'atomic' type held by a xel_t construct, to help with the
// formulation of type restrictions in binary operators. The
// 'level-0' definition accepts arbitrary types T, with the
// intention that xel_t should treat fundamentals and simdized
// values as semantically equivalent.

template < typename T , typename F = void >
struct form
{
  typedef F type ;
  typedef F atom_t ;
  static const std::size_t level = 0 ;
  static const std::size_t size = 1 ;
} ;

template < typename T , std::size_t N , typename F >
struct form < xel_t < T , N > , F >
{
  typedef typename
    std::conditional
      < form < T > :: level == 0 ,
        xel_t < F , N > ,
        xel_t < form < T , F > , N >
      > :: type type ;

  typedef typename
    std::conditional
      < form < T > :: level == 0 ,
        T ,
        typename form < T > :: atom_t
      > :: type atom_t ;

  static const std::size_t level = form < T > :: level + 1 ;
  static const std::size_t size = N ;
} ;

// free function templates for deinterleave and interleave, which can
// be overridden for specific vec_t - e.g. zimt::vc_simd_type. The
// fallback, general case uses a regular gather/scatter, which is
// will work for any vec_t but may be less efficient than specialized
// code for de/interleaving

template < typename vec_t , std::size_t nch >
void deinterleave ( const xel_t < ET < vec_t > , nch > * _src ,
                    xel_t < vec_t , nch > & v )
{
  static const typename vec_t::index_type indexes
    = vec_t::IndexesFromZero() * int ( nch ) ;
  auto * src = _src->data() ;
  for ( std::size_t i = 0 ; i < nch ; i++ , src++ )
    v[i].gather ( src , indexes ) ;
}

template < typename vec_t , std::size_t nch >
void interleave ( const xel_t < vec_t , nch > & v ,
                  xel_t < ET < vec_t > , nch > * _trg )
{
  static const typename vec_t::index_type indexes
    = vec_t::IndexesFromZero() * int ( nch ) ;
  auto * trg = _trg->data() ;
  for ( std::size_t i = 0 ; i < nch ; i++ , trg++ )
    v[i].scatter ( trg , indexes ) ;
}

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

template < typename T , std::size_t N >
struct xel_t
{
  typedef T value_type ;
  static const std::size_t nch = N ;

#include "simd/vector_common.h"

// binary operators

// with the restrictions below, we avoid the pitfalls of directly
// accepting xel_t or value_type arguments (with the unwanted
// implicit type conversions).
// But we're limited to RHS which are 'one level down'. simd_t
// and other simdized types are at the same level as fundamentals
// (because we've said so in class 'form'), so this restriction
// is less of an issue than one might think. I even feel that
// the restriction is sensible, to avoid hard-to-comprehend code
// using complicated broadcasting constructs. In a way a xel_t of
// several simdized values is as far as we want to go, requiring
// explicit user code for 'deeper' constructs. With the binary
// operator code as it stands we can interoperate such values with
// both xel_t of the same number of fundamentals and single simdized
// values, both with 'C semantics' type promotion.

#define INTEGRAL_ONLY \
  static_assert ( zimt::is_integral < value_type > :: value , \
                  "this operation is only allowed for integral types" ) ;

#define BOOL_ONLY \
  static_assert ( std::is_same < value_type , bool > :: value , \
                  "this operation is only allowed for booleans" ) ;

#define OP_FUNC(OPFUNC,OP,CONSTRAINT) \
  template < typename RHST , \
             typename = typename std::enable_if \
                       < std::is_same \
                           < typename form < T > :: type , \
                             typename form < RHST > :: type \
                           > :: value \
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
                       < std::is_same \
                           < typename form < T > :: type , \
                             typename form < RHST > :: type \
                           > :: value \
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
                       < std::is_same \
                           < typename form < T > :: type , \
                             typename form < LHST > :: type \
                           > :: value \
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

OP_FUNC(operator&&,&&,BOOL_ONLY)
OP_FUNC(operator||,||,BOOL_ONLY)

#undef OP_FUNC
#undef INTEGRAL_ONLY
#undef BOOL_ONLY

// assignment from equally-sized container.
// Note that the rhs can use any elementary type which can be legally
// assigned to value_type. This allows transport of information from
// differently typed objects, but there are no further constraints on
// the types involved, which may degrade precision. It's the user's
// responsibility to make sure such assignments have the desired effect
// and overload them if necessary.

template < typename U , template < typename , std::size_t > class V >
xel_t & operator= ( const V < U , nch > & rhs )
{
  for ( size_type i = 0 ; i < nch ; i++ )
    (*this) [ i ] = rhs [ i ] ;
  return *this ;
}

template < typename U , template < typename , std::size_t > class V >
xel_t ( const V < U , nch > & ini )
{
  *this = ini ;
}

// 'projection' to some other fixed-size aggregate via a templated
// conversion operator: converts a xel_t to an object of class C
// which holds nch value_type. We want to use xel_t as all-purpose
// fixed-size aggregate and still allow user code which assigns
// xel_t values to, say, std::arrays when they are returned by
// functions.

template < template < typename , std::size_t > class C >
operator C < value_type , nch > ()
{
  C < value_type , nch > result ;
  for ( size_type i = 0 ; i < nch ; i++ )
    result [ i ] = _store [ i ] ;
  return result ;
}

bool operator== ( const xel_t < value_type , nch > rhs ) const
{
  for ( std::size_t i = 0 ; i < nch ; i++ )
  {
    if ( _store[i] != rhs[i] )
      return false ;
  }
  return true ;
}

bool operator!= ( const xel_t < value_type , nch > rhs ) const
{
  return ! ( (*this) == rhs ) ;
}

// produce a xel with one more element at position 'at', defaulting to
// the last position.

xel_t < value_type , nch + 1 > widen ( const value_type & by ,
                                         const std::size_t & at = nch  )
{
  xel_t < value_type , nch + 1 > result ;
  if ( at >= nch )
  {
    for ( std::size_t d = 0 ; d < nch ; d++ )
    {
      result [ d ] = _store [ d ] ;
    }
    result [ nch ] = by ;
  }
  else
  {
    std::size_t t = 0 ;
    for ( std::size_t d = 0 ; d < nch ; d++ )
    {
      if ( d == at )
        result [ t++ ] = by ;
      result [ t++ ] = _store [ d ] ;
    }
  }
  return result ;
}

// to use the idiom A(M) = B for simple xel_t, we have to repeat
// the code we use for vectorized types using bool instead of
// a mask. With this bit of syntactic sugar, we can use more
// vector code for scalar values as well.

#define INTEGRAL_ONLY \
  static_assert ( std::is_integral < value_type > :: value , \
                  "this operation is only allowed for integral types" ) ;

#define BOOL_ONLY \
  static_assert ( std::is_same < value_type , bool > :: value , \
                  "this operation is only allowed for booleans" ) ;

struct masked_xel_t
{
  bool whether ; // if the 'mask' is true
  xel_t & whither ;   // whither will be assigned to

  masked_xel_t ( bool _whether ,
                 xel_t & _whither )
  : whether ( _whether ) ,
    whither ( _whither )
    { }

  #define OPEQ_FUNC(OPFUNC,OPEQ,CONSTRAINT) \
    xel_t & OPFUNC ( value_type rhs ) \
    { \
      CONSTRAINT \
      if ( whether ) \
           whither OPEQ rhs ; \
      return whither ; \
    } \
    xel_t & OPFUNC ( xel_t rhs ) \
    { \
      CONSTRAINT \
        if ( whether ) \
             whither OPEQ rhs ; \
      return whither ; \
    }

  OPEQ_FUNC(operator=,=,)
  OPEQ_FUNC(operator+=,+=,)
  OPEQ_FUNC(operator-=,-=,)
  OPEQ_FUNC(operator*=,*=,)
  OPEQ_FUNC(operator/=,/=,)
  OPEQ_FUNC(operator%=,%=,INTEGRAL_ONLY)
  OPEQ_FUNC(operator&=,&=,INTEGRAL_ONLY)
  OPEQ_FUNC(operator|=,|=,INTEGRAL_ONLY)
  OPEQ_FUNC(operator^=,^=,INTEGRAL_ONLY)
  OPEQ_FUNC(operator<<=,<<=,INTEGRAL_ONLY)
  OPEQ_FUNC(operator>>=,>>=,INTEGRAL_ONLY)

#undef OPEQ_FUNC

#undef INTEGRAL_ONLY
#undef BOOL_ONLY
} ;

masked_xel_t & operator() ( const bool & condition )
{
  return masked_xel_t ( condition , *this ) ;
}

bool any_of ( bool condition )
{
  return condition ;
}

bool all_of ( bool condition )
{
  return condition ;
}

bool none_of ( bool condition )
{
  return ! condition ;
}

// next we have code which will only be present for xel_t of some
// SIMD data type, like hwy_simd_type or gen_simd_type. For such
// types, value_type will inherit from simd_flag, so we
// can use enable_if to produce the code if appropriate.

template < typename = std::enable_if
  < std::is_base_of < simd_flag , value_type > :: value > >
void fluff_contiguous ( xel_t < ET < value_type > , nch > * trg ,
                        std::true_type ) const // multi-channel
{
  interleave ( *this , trg ) ;
}

template < typename = std::enable_if
  < std::is_base_of < simd_flag , value_type > :: value > >
void fluff_contiguous ( xel_t < ET < value_type > , nch > * _trg ,
                        std::false_type ) const // single channel
{
  auto * trg = _trg->data() ;
  (*this)[0].store ( trg ) ;
}

// unstrided fluff. The target memory is contiguous, meaning one
// xel is following the other without gaps. For this scenario, we
// don't need a stride, but we dispatch further on whether we have
// single-channel or multi-channel data.

template < typename = std::enable_if
  < std::is_base_of < simd_flag , value_type > :: value > >
void fluff ( xel_t < ET < value_type > , nch > * trg ) const
{
  typedef std::integral_constant < bool , ( nch > 1 ) > tag_t ;
  fluff_contiguous ( trg , tag_t() ) ;
}

// strided 'fluff'. The target memory is not contiguous (hence the
// stride) - but we do a runtime dispatch to unstrided fluff (above)
// if the stride is one.

template < typename = std::enable_if
  < std::is_base_of < simd_flag , value_type > :: value > >
void fluff ( xel_t < ET < value_type > , nch > * _trg ,
             std::size_t stride ) const
{
  const typename value_type::index_type indexes
    = value_type::IndexesFromZero() * int ( stride * nch ) ;
  auto * trg = _trg->data() ;
  if ( stride == 1 )
  {
    fluff ( _trg ) ;
  }
  else
  {
    for ( std::size_t i = 0 ; i < nch ; i++ , trg++ )
      (*this)[i].scatter ( trg , indexes ) ;
  }
}

template < typename = std::enable_if
  < std::is_base_of < simd_flag , value_type > :: value > >
void fluff ( xel_t < ET < value_type > , nch > * trg ,
             std::size_t stride ,
             std::size_t cap ) const
{
  for ( std::size_t e = 0 ; e < cap ; e++ )
  {
    for ( std::size_t ch = 0 ; ch < nch ; ch++ )
    {
      trg[e*stride][ch] = (*this)[ch][e] ;
    }
  }
}

template < typename = std::enable_if
  < std::is_base_of < simd_flag , value_type > :: value > >
void bunch_contiguous ( const xel_t < ET < value_type > , nch > * src ,
                        std::true_type ) // multi-channel
{
  deinterleave ( src , *this ) ;
}

template < typename = std::enable_if
  < std::is_base_of < simd_flag , value_type > :: value > >
void bunch_contiguous ( const xel_t < ET < value_type > , nch > * _src ,
                        std::false_type ) // single channel
{
  auto const * src = _src->data() ;
  (*this)[0].load ( src ) ;
}

template < typename = std::enable_if
  < std::is_base_of < simd_flag , value_type > :: value > >
void bunch ( const xel_t < ET < value_type > , nch > * src )
{
  typedef std::integral_constant < bool , ( nch > 1 ) > tag_t ;
  bunch_contiguous ( src , tag_t() ) ;
}

template < typename = std::enable_if
  < std::is_base_of < simd_flag , value_type > :: value > >
void bunch ( const xel_t < ET < value_type > , nch > * _src ,
             std::size_t stride )
{
  const typename value_type::index_type indexes
    = value_type::IndexesFromZero() * int ( stride * nch ) ;
  auto const * src = _src->data() ;
  if ( stride == 1 )
  {
    bunch ( _src ) ;
  }
  else
  {
    for ( std::size_t i = 0 ; i < nch ; i++ , src++ )
      (*this)[i].gather ( src , indexes ) ;
  }
}

template < typename = std::enable_if
  < std::is_base_of < simd_flag , value_type > :: value > >
void stuff ( std::size_t cap )
{
  auto mask = ! ( value_type::IndexesFromZero() < int(cap) ) ;
  for ( std::size_t ch = 0 ; ch < nch ; ch++ )
  {
    (*this)[ch] ( mask ) = (*this)[ch][cap-1] ;
  }
}

// capped bunch. TODO: rewrite using masks

template < typename = std::enable_if
  < std::is_base_of < simd_flag , value_type > :: value > >
void bunch ( const xel_t < ET < value_type > , nch > * src ,
             std::size_t stride ,
             std::size_t cap ,
             bool _stuff = false )
{
  for ( std::size_t e = 0 ; e < cap ; e++ )
  {
    for ( std::size_t ch = 0 ; ch < nch ; ch++ )
    {
      (*this)[ch][e] = src [ e * stride ] [ ch ] ;
    }
  }
  if ( _stuff )
  {
    stuff ( cap ) ;
  }
}

// The next to member functions are for stashing and unstashing
// simdized data. This is used e.g. by class vstorer and vloader
// (see get.h and put.h)

// load operation from contiguous memory of the elementary
// type of a simdized datum.

template < typename = std::enable_if
  < std::is_base_of < simd_flag , value_type > :: value > >
void load ( const ET < value_type > * src )
{
  for ( std::size_t ch = 0 ; ch < nch ; ch++ )
  {
    (*this)[ch].load ( src ) ;
    src += value_type::size() ;
  }
}

// store operation of a simdized datum to contiguous memory
// of the elementary type

template < typename = std::enable_if
  < std::is_base_of < simd_flag , value_type > :: value > >
void store ( ET < value_type > * trg ) const
{
  for ( std::size_t ch = 0 ; ch < nch ; ch++ )
  {
    (*this)[ch].store ( trg ) ;
    trg += value_type::size() ;
  }
}

// gather and scatter of simdized xel from/to memory

template < typename index_type ,
           typename = std::enable_if
  < std::is_base_of < simd_flag , value_type > :: value > >
void gather ( const ET < value_type > * p_src ,
              const index_type & indexes )
{
  for ( std::size_t ch = 0 ; ch < nch ; ch++ , p_src++ )
  {
    (*this)[ch].gather ( p_src , indexes ) ;
  }
}

template < typename index_type ,
           typename = std::enable_if
  < std::is_base_of < simd_flag , value_type > :: value > >
void scatter ( ET < value_type > * p_trg ,
               const index_type & indexes )
{
  for ( std::size_t ch = 0 ; ch < nch ; ch++ , p_trg++ )
  {
    (*this)[ch].scatter ( p_trg , indexes ) ;
  }
}

} ; // end of struct xel_t

template < typename T , std::size_t D >
T squared_norm ( const zimt::xel_t < T , D > & v )
{
  T sqn = v[0] * v[0] ;
  for ( std::size_t i = 1 ; i < D ; i++ )
    sqn += v[i] * v[i] ;
  return sqn ;
}

template < typename T , std::size_t D >
T norm ( const zimt::xel_t < T , D > & v )
{
  return sqrt ( squared_norm ( v ) ) ;
}

} ; // end of namespace zimt

#undef XEL

#endif // #define ZIMT_XEL_T_H

