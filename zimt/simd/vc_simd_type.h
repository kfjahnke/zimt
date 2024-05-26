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

/*! \file vc_simd_type.h

    \brief SIMD type derived from Vc::SimdArray
    
    Initially, zimt used Vc::SimdArray directly, but over time
    I have written interfaces to several SIMD implementations based
    on zimt's own gen_simd_type, and I now prefer to introduce SIMD
    capability to my code through a common interface derived from
    gen_simd_type, which allows for simple switching from one
    SIMD implementation to another.

*/

#ifndef VC_SIMD_TYPE_H
#define VC_SIMD_TYPE_H

#include <iostream>
#include <Vc/Vc>

#include "gen_simd_type.h"

namespace simd
{

template < typename _value_type ,
           std::size_t _vsize >
struct vc_simd_type ;

// conversion to and from gen_simd_type of equal T

template < typename T , std::size_t vsize >
void convert ( const gen_simd_type < T , vsize > & src ,
                     vc_simd_type < T , vsize > & trg )
{
  trg.load ( src.data() ) ;
}

template < typename T , std::size_t vsize >
void convert ( const vc_simd_type < T , vsize > & src ,
                     gen_simd_type < T , vsize > & trg )
{
  src.store ( trg.data() ) ;
}

// conversion to and from gen_simd_type of different T
// This uses goading, because we can't be sure that src_t can be
// handled by Vc.

template < typename src_t , typename trg_t , std::size_t vsize >
void convert ( const gen_simd_type < src_t , vsize > & src ,
                     vc_simd_type < trg_t , vsize > & trg )
{
  for ( std::size_t i = 0 ; i < vsize ; i++ )
    trg[i] = src[i] ;
}

template < typename src_t , typename trg_t , std::size_t vsize >
void convert ( const vc_simd_type < src_t , vsize > & src ,
                     gen_simd_type < trg_t , vsize > & trg )
{
  for ( std::size_t i = 0 ; i < vsize ; i++ )
    trg[i] = src[i] ;
}

/// class template vc_simd_type provides a fixed-size SIMD type.
/// This implementation of zimt::vc_simd_type uses Vc::SimdArray
/// The 'acrobatics' may seem futile - why inherit privately from
/// Vc::SimdArray, then code a class template which does essentially
/// the same? There are several reasons: first, the wrapper class
/// results in a common interface shared with the other SIMD
/// implementations, second, there are some added members which
/// can't be 'put into' Vc::SimdArray from the outside. And, third,
/// the template signature is uniform, avoiding Vc::SimdArray's
/// two additional template arguments.
// A note on the disuse of tag_t::vsize: this would be nice, but
// g++ does not extend the usage of vsize to the friend function
// template definitions used to implement binary operators with
// a fundamental as LHS and proclaims vsize as undefined. One might
// argue that the using declaration is strictly for code which is
// 'inside' struct vc_simd_type, but I prefer clang++'s wider
// notion of where vsize is applicable.

template < typename _value_type ,
           std::size_t _vsize >
struct vc_simd_type
: private Vc::SimdArray < _value_type , _vsize > ,
  public simd_tag < _value_type , _vsize , VC >
{
  typedef simd_tag < _value_type , _vsize , VC > tag_t ;
  using typename tag_t::value_type ;
  // using tag_t::vsize ; // works with clang++, but not with g++, hence:
  static const std::size_t vsize = _vsize ;
  using tag_t::backend ;

  typedef Vc::SimdArray < _value_type , _vsize > base_t ;

  // we need to alow access to new and delete of the base class

  using base_t::operator new ;
  using base_t::operator new[] ;
  using base_t::operator delete ;
  using base_t::operator delete[] ;

  // access the underlying base type

  base_t & to_base()
  {
    return ( reinterpret_cast < base_t & > ( * this ) ) ;
  }

  const base_t & to_base() const
  {
    return ( reinterpret_cast < const base_t & > ( * this ) ) ;
  }

  // make it easy to convert to Vc::SimdArray

  operator base_t()
  {
    return to_base() ;
  }

  operator base_t() const
  {
    return to_base() ;
  }

  typedef std::size_t size_type ;

  // provide the size as a constexpr

  static constexpr size_type size()
  {
    return vsize ;
  }

  // both styles of the types used for indices and masks

  using typename base_t::index_type ;
  using typename base_t::mask_type ;

  using typename base_t::IndexType ;
  using typename base_t::MaskType ;

  // type for an individual index value, while index_type
  // holds vsize of index_ele_type

  typedef typename index_type::value_type index_ele_type ;

  // operator[] is mapped to SimdArray element access

  using base_t::operator[] ;

  // assignment from a value_type, base_t.

  vc_simd_type & operator= ( const value_type & rhs )
  {
    to_base() = rhs ;
    return *this ;
  }
  vc_simd_type & operator= ( const value_type && rhs )
  {
    to_base() = rhs ;
    return *this ;
  }

  vc_simd_type & operator= ( const base_t & ini )
  {
    to_base() = ini ;
    return *this ;
  }
  vc_simd_type & operator= ( const base_t && ini )
  {
    to_base() = ini ;
    return *this ;
  }

  vc_simd_type & operator= ( const vc_simd_type & ini ) = default ;
  vc_simd_type & operator= ( vc_simd_type && ini ) = default ;

  // c'tor from value_type, base_t. We use the assignment operator for
  // initialization.

  vc_simd_type ( const value_type & ini )
  : base_t ( ini )
  { }
  vc_simd_type ( const value_type && ini )
  : base_t ( ini )
  { }

  vc_simd_type ( const base_t & ini )
  : base_t ( ini )
  { }
  
  vc_simd_type ( const base_t && ini )
  : base_t ( ini )
  { }

  // these two c'tors are left in default mode

  vc_simd_type() = default ;
  vc_simd_type ( const vc_simd_type & ) = default ;
  vc_simd_type ( vc_simd_type && ) = default ;

  // c'tor and copy c'tor from another vc_simd_type of different
  // value_type, delegating to the base class

  template < typename U >
  vc_simd_type ( const vc_simd_type < U , vsize > & ini )
  : base_t ( ini.to_base() )
  { }

  template < typename U >
  vc_simd_type ( const vc_simd_type < U , vsize > && ini )
  : base_t ( ini.to_base() )
  { }

  template < typename U >
  vc_simd_type & operator= ( const vc_simd_type < U , vsize > & ini )
  {
    to_base() = ini.to_base() ;
    return *this ;
  }

  template < typename U >
  vc_simd_type & operator= ( const vc_simd_type < U , vsize > && ini )
  {
    to_base() = ini.to_base() ;
    return *this ;
  }

  // because all vc_simd_type objects are of a distinct size
  // explicitly coded in template arg vsize, we can initialize
  // from an initializer_list. This is probably not very fast,
  // but it's nice to have for experimentation.

  vc_simd_type ( const std::initializer_list < value_type > & rhs )
  {
    assert ( rhs.size() == vsize ) ; // TODO: prefer constexpr
    std::size_t i = 0 ;
    for ( const auto & src : rhs )
      (*this) [ i++ ] = src ;
  }

  // assignment from a gen_simd_type on the rhs

  template < typename U >
  vc_simd_type & operator= ( const gen_simd_type < U , vsize > & rhs )
  {
    convert ( rhs , *this ) ;
    return *this ;
  }

  template < typename U >
  vc_simd_type ( const gen_simd_type < U , vsize > & rhs )
  {
    *this = rhs ;
  }

  // conversion to a gen_simd_type

  template < typename U >
  operator gen_simd_type < U , vsize > ()
  {
    gen_simd_type < U , vsize > result ;
    convert ( *this , result ) ;
    return result ;
  }

  // use Vc's IndexesFromZero, Zero and One - but convert the result
  // to the corresponding zimt type (g++ is picky about that)

  static const index_type IndexesFromZero()
  {
    return index_type ( base_t::IndexesFromZero() ) ;
  }
  
  // ditto for Zero and One

  static const vc_simd_type Zero()
  {
    return base_t::Zero() ;
  }
  
  static const vc_simd_type One()
  {
    return base_t::One() ;
  }
  
  // iota() produces values rising from zero

  static const vc_simd_type iota()
  {
    return base_t::IndexesFromZero() ;
  }

  // variant which starts from a different starting point and optionally
  // uses steps other than one.

  static const index_type IndexesFrom ( const index_ele_type & start ,
                                        const index_ele_type & step )
  {
    return ( ( IndexesFromZero() * step ) + start ) ;
  }

  // echo the vector to a std::ostream, read it from an istream

  friend std::ostream & operator<< ( std::ostream & osr ,
                                     vc_simd_type it )
  {
    osr << it.to_base() ;
    return osr ;
  }

  friend std::istream & operator>> ( std::istream & isr ,
                                     vc_simd_type it )
  {
    for ( size_type i = 0 ; i < vsize ; i++ )
      isr >> it [ i ] ;
    return isr ;
  }

  // memory access functions, which load and store vector data.
  // We start out with functions transporting data from memory into
  // the vc_simd_type. Some of these operations have corresponding
  // c'tors which use the member function to initialize to_base().

  using base_t::load ;
  using base_t::store ;

  // gather/scatter, first with index_type, then with a vc_simd_type
  // object providing indexes, delegating to Vc for the purpose.

  void gather ( const value_type * const p_src ,
                const index_type & indexes )
  {
    to_base().gather ( p_src , indexes ) ;
  }

  void scatter ( value_type * const p_trg ,
                 const index_type & indexes ) const
  {
    to_base().scatter ( p_trg , indexes ) ;
  }

  template < typename _index_type >
  void gather ( const value_type * const p_src ,
                const _index_type & _indexes )
  {
    for ( std::size_t i = 0 ; i < vsize ; i++ )
      (*this) [ i ] = p_src [ _indexes [ i ] ] ;
  }

  template < typename _index_type >
  void scatter ( value_type * const p_trg ,
                 const _index_type & _indexes ) const
  {
    for ( std::size_t i = 0 ; i < vsize ; i++ )
      p_trg [ _indexes [ i ] ] = (*this) [ i ] ;
  }

  // c'tor from pointer and indexes, uses gather

  template < typename index_type >
  vc_simd_type ( const value_type * const p_src ,
                 const index_type & indexes )
  {
    gather ( p_src , indexes ) ;
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

  // broadcasting functions processing single value_type

  typedef std::function < value_type() > gen_f ;
  typedef std::function < value_type ( const value_type & ) > mod_f ;
  typedef std::function < value_type ( const value_type & , const value_type & ) > bin_f ;

  vc_simd_type & broadcast ( gen_f f )
  {
    for ( std::size_t i = 0 ; i < size() ; i++ )
    {
      (*this)[i] = f() ;
    }
    return *this ;
  }

  vc_simd_type & broadcast ( mod_f f )
  {
    for ( std::size_t i = 0 ; i < size() ; i++ )
    {
      (*this)[i] = f ( (*this)[i] ) ;
    }
    return *this ;
  }

  vc_simd_type & broadcast ( bin_f f , const vc_simd_type & rhs )
  {
    for ( std::size_t i = 0 ; i < size() ; i++ )
    {
      (*this)[i] = f ( (*this)[i] , rhs[i] ) ;
    }
    return *this ;
  }

  typedef Vc::Vector < value_type > vec_t ;
  static const std::size_t hsize = vec_t::size() ;
  static const std::size_t nvec = vsize / hsize ;

  typedef std::function < vec_t() > gen_vf ;
  typedef std::function < vec_t ( const vec_t & ) > mod_vf ;
  typedef std::function < vec_t ( const vec_t & , const vec_t & ) > bin_vf ;

  // we take a lazy approach to broadcasting vector functions and go via
  // a vsize-sized buffer of value_type, expecting that the buffer will
  // be optimized away

  // broadcast a vector generator function

  template < typename = std::enable_if < ( vsize % hsize == 0 ) > >
  vc_simd_type & vbroadcast ( gen_vf f )
  {
    alignas ( sizeof ( vec_t ) ) value_type buffer [ vsize ] ;
    for ( std::size_t i = 0 , ofs = 0 ; i < nvec ; i++ , ofs += hsize )
      f().store ( buffer + ofs ) ;
    load ( buffer ) ;
    return *this ;
  }

  // broadcast a vector modulator

  template < typename = std::enable_if < ( vsize % hsize == 0 ) > >
  vc_simd_type & vbroadcast ( mod_vf f )
  {
    alignas ( sizeof ( vec_t ) ) value_type buffer [ vsize ] ;
    store ( buffer ) ;
    for ( std::size_t i = 0 , ofs = 0 ; i < nvec ; i++ , ofs += hsize )
      f ( vec_t ( buffer + ofs ) ).store ( buffer + ofs ) ;
    load ( buffer ) ;
    return *this ;
  }

  // broadcast a vector binary function

  template < typename = std::enable_if < ( vsize % hsize == 0 ) > >
  vc_simd_type & vbroadcast ( bin_vf f , const vc_simd_type & rhs )
  {
    alignas ( sizeof ( vec_t ) ) value_type buffer [ vsize ] ;
    alignas ( sizeof ( vec_t ) ) value_type rhs_buffer [ vsize ] ;
    store ( buffer ) ;
    rhs.store ( rhs_buffer ) ;
    for ( std::size_t i = 0 , ofs = 0 ; i < nvec ; i++ , ofs += hsize )
      f ( vec_t ( buffer + ofs ) , vec_t ( rhs_buffer + ofs ) )
        .store ( buffer + ofs ) ;
    load ( buffer ) ;
    return *this ;
  }

  // apply functions from namespace std to each element in a vector,
  // or to each corresponding set of elements in a set of vectors
  // - going up to three for fma.
  // Here we delegate to the Vc functions.

  #define BROADCAST_STD_FUNC(FUNC) \
    friend vc_simd_type FUNC ( const vc_simd_type & arg ) \
    { \
      return FUNC ( arg.to_base() ) ; \
    }

  BROADCAST_STD_FUNC(abs)
  BROADCAST_STD_FUNC(trunc)
  BROADCAST_STD_FUNC(round)
  BROADCAST_STD_FUNC(floor)
  BROADCAST_STD_FUNC(ceil)
  BROADCAST_STD_FUNC(log)
  BROADCAST_STD_FUNC(exp)
  BROADCAST_STD_FUNC(sqrt)

  BROADCAST_STD_FUNC(sin)
  BROADCAST_STD_FUNC(cos)
  // BROADCAST_STD_FUNC(tan)
  BROADCAST_STD_FUNC(asin)
  BROADCAST_STD_FUNC(acos)
  BROADCAST_STD_FUNC(atan)

  #undef BROADCAST_STD_FUNC

  // Vc doesn't offer tan(), but it has sincos.

  friend vc_simd_type tan ( const vc_simd_type & arg )
  {
    base_t sin , cos ;
    Vc::sincos ( arg.to_base() , &sin , &cos ) ;
    auto result = sin / cos ;
    result ( cos == 0 ) = M_PI_2 ;
    return result ;
  }

  friend void sincos ( const vc_simd_type & x ,
                       vc_simd_type & s ,
                       vc_simd_type & c )
  {
    Vc::sincos ( x.to_base() , &(s.to_base()) , &(c.to_base()) ) ;
  }

  #define BROADCAST_STD_FUNC2(FUNC) \
    friend vc_simd_type FUNC ( const vc_simd_type & arg1 , \
                               const vc_simd_type & arg2 ) \
    { \
      return FUNC ( arg1.to_base() , arg2.to_base() ) ; \
    }

  BROADCAST_STD_FUNC2(atan2)

  #undef BROADCAST_STD_FUNC2

  // Vc has no pow() function

  friend vc_simd_type pow ( const vc_simd_type & base ,
                            const vc_simd_type & exponent )
  {
    return exp ( exponent.to_base() * log ( base.to_base() ) ) ;
  }

  // macros used for the parameter 'CONSTRAINT' in the definitions
  // further down. Some operations are only allowed for integral types
  // or boolans. This might be enforced by enable_if, here we use a
  // static_assert with a clear error message.
  // TODO: might relax constraints by using 'std::is_convertible'

  #define INTEGRAL_ONLY \
    static_assert ( std::is_integral < value_type > :: value , \
                    "this operation is only allowed for integral types" ) ;

  #define BOOL_ONLY \
    static_assert ( std::is_same < value_type , bool > :: value , \
                    "this operation is only allowed for booleans" ) ;

  // augmented assignment operators. Some operators are only applicable
  // to specific data types, which is enforced by 'CONSTRAINT'.
  // One might consider widening the scope by making these operator
  // functions templates and accepting arbitrary indexable types.
  // Only value_type and vc_simd_type are taken as rhs arguments.

  #define OPEQ_FUNC(OPFUNC,OPEQ,CONSTRAINT) \
    vc_simd_type & OPFUNC ( const value_type & rhs ) \
    { \
      CONSTRAINT \
      to_base() OPEQ rhs ; \
      return *this ; \
    } \
    vc_simd_type & OPFUNC ( const vc_simd_type & rhs ) \
    { \
      CONSTRAINT \
      to_base() OPEQ rhs.to_base() ; \
      return *this ; \
    }

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

  // binary operators and left and right scalar operations with
  // value_type, unary operators -, ! and ~

#define OP_FUNC(OPFUNC,OP,CONSTRAINT) \
  template < typename RHST , \
             typename = typename std::enable_if \
                       < std::is_fundamental < RHST > :: value \
                       > :: type \
           > \
  vc_simd_type < PROMOTE ( value_type , RHST ) , size() > \
  OPFUNC ( vc_simd_type < RHST , vsize > rhs ) const \
  { \
    CONSTRAINT \
    vc_simd_type < PROMOTE ( value_type , RHST ) , vsize > help ( *this ) ; \
    return help.to_base() OP rhs.to_base() ; \
  } \
  template < typename RHST , \
             typename = typename std::enable_if \
                       < std::is_fundamental < RHST > :: value \
                       > :: type \
           > \
  vc_simd_type < PROMOTE ( value_type , RHST ) , vsize > \
  OPFUNC ( RHST rhs ) const \
  { \
    CONSTRAINT \
    vc_simd_type < PROMOTE ( value_type , RHST ) , vsize > help ( *this ) ; \
    return help.to_base() OP rhs ; \
  } \
  template < typename LHST , \
             typename = typename std::enable_if \
                       < std::is_fundamental < LHST > :: value \
                       > :: type \
           > \
  friend vc_simd_type < PROMOTE ( LHST , value_type ) , vsize > \
  OPFUNC ( LHST lhs , vc_simd_type rhs ) \
  { \
    CONSTRAINT \
    vc_simd_type < PROMOTE ( LHST , value_type ) , vsize > help ( lhs ) ; \
    return help.to_base() OP rhs.to_base() ; \
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

  #define OP_FUNC(OPFUNC,OP,CONSTRAINT) \
    vc_simd_type OPFUNC() const \
    { \
      return OP to_base() ; \
    }

  OP_FUNC(operator-,-,)
  OP_FUNC(operator!,!,BOOL_ONLY)
  OP_FUNC(operator~,~,INTEGRAL_ONLY)

  #undef OP_FUNC

  // provide methods to produce a mask on comparing a vector
  // with another vector or a value_type.

  #define COMPARE_FUNC(OP,OPFUNC) \
  friend mask_type OPFUNC ( const vc_simd_type & lhs , \
                            const vc_simd_type & rhs ) \
  { \
    return lhs.to_base() OP rhs.to_base() ; \
  } \
  friend mask_type OPFUNC ( const vc_simd_type & lhs , \
                            const value_type & rhs ) \
  { \
    return lhs.to_base() OP rhs ; \
  } \
  friend mask_type OPFUNC ( const value_type & lhs , \
                            const vc_simd_type & rhs ) \
  { \
    return lhs OP rhs.to_base() ; \
  }

  COMPARE_FUNC(<,operator<) ;
  COMPARE_FUNC(<=,operator<=) ;
  COMPARE_FUNC(>,operator>) ;
  COMPARE_FUNC(>=,operator>=) ;
  COMPARE_FUNC(==,operator==) ;
  COMPARE_FUNC(!=,operator!=) ;

  #undef COMPARE_FUNC

  // next we define a masked vector as an object holding two members:
  // one mask, determining which of the vector's elements will be
  // 'open' to an effect, and one reference to a vector, which will
  // be affected by the operation.
  // The resulting object will only be viable as long as the referred-to
  // vector stays 'alive' - it's meant as a construct to be processed
  // in the same scope, as the lhs of an assignment, typically using
  // notation introduced by Vc: a vector's operator() is overloaded to
  // to produce a masked_type when called with a mask_type object, and
  // the resulting masked_type object is then assigned to.
  // Note that this does not have any effect on those values in 'whither'
  // for which the mask is false. They remain unchanged.

  struct masked_type
  {
    const mask_type whether ; // if the mask is true at whether[i]
    vc_simd_type & whither ;  // whither[i] will be assigned to

    masked_type ( const mask_type & _whether ,
                  vc_simd_type & _whither )
    : whether ( _whether ) ,
      whither ( _whither )
      { }

    // for the masked vector, we define the complete set of assignments:

    #define OPEQ_FUNC(OPFUNC,OPEQ,CONSTRAINT) \
      vc_simd_type & OPFUNC ( const value_type & rhs ) \
      { \
        CONSTRAINT \
        whither.to_base() ( whether ) OPEQ rhs ; \
        return whither ; \
      } \
      vc_simd_type & OPFUNC ( const vc_simd_type & rhs ) \
      { \
        CONSTRAINT \
        whither.to_base() ( whether ) OPEQ rhs.to_base() ; \
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

  // mimicking Vc, we define operator() with a mask_type argument
  // to produce a masked_type object, which can be used later on to
  // masked-assign to the referred-to vector. With this definition
  // we can use the same syntax Vc uses, e.g. v1 ( v1 > v2 ) = v3
  // This helps write code which compiles with Vc and without,
  // because this idiom is 'very Vc'.

  masked_type operator() ( const mask_type & mask )
  {
    return masked_type ( mask , *this ) ;
  }

  // member functions at_least and at_most. These functions provide the
  // same functionality as max, or min, respectively. Given vc_simd_type X
  // and some threshold Y, X.at_least ( Y ) == max ( X , Y )
  // Having the functionality as a member function makes it easy to
  // implement, e.g., min as: min ( X , Y ) { return X.at_most ( Y ) ; }

  #define CLAMP(FNAME,REL) \
    vc_simd_type FNAME ( const vc_simd_type & threshold ) const \
    { \
      return REL ( to_base() , threshold.to_base() ) ; \
    } \
    vc_simd_type FNAME ( const value_type & threshold ) const \
    { \
      return REL ( to_base() , threshold ) ; \
    } \

  CLAMP(at_least,Vc::max)
  CLAMP(at_most,Vc::min)

  #undef CLAMP

  // sum of vector elements. Note that there is no type promotion; the
  // summation is done to value_type. Caller must make sure that overflow
  // is not a problem.

  value_type sum() const
  {
    return to_base().sum() ;
  }
} ;

template < typename T , std::size_t N >
struct allocator_traits < vc_simd_type < T , N > >
{
  typedef Vc::Allocator < vc_simd_type < T , N > >
    type ;
} ;

} ; // namespace simd

namespace std
{
  template < typename T , std::size_t N >
  struct allocator_traits < simd::vc_simd_type < T , N > >
  {
    typedef Vc::Allocator < simd::vc_simd_type < T , N > >
      allocator_type ;
  } ;
} ;


namespace zimt
{

  template < typename T , size_t N >
  struct is_integral < simd::vc_simd_type < T , N > >
  : public std::is_integral < T >
  { } ;

} ;
#endif // #define VC_SIMD_TYPE_H
