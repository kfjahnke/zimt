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
    on zimt's own simd_type, and I now prefer to introduce SIMD
    capability to my code through a common interface derived from
    zimt::simd_type, which allows for simple switching from one
    SIMD implementation to another.

*/

#ifndef VSPLINE_VC_SIMD_TYPE_H
#define VSPLINE_VC_SIMD_TYPE_H

#include <iostream>
#include <Vc/Vc>

#include "simd_type.h"

namespace zimt
{

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

template < typename _value_type ,
           std::size_t _vsize >
struct vc_simd_type
: private Vc::SimdArray < _value_type , _vsize > ,
  public zimt::simd_tag < _value_type , _vsize , zimt::VC >
{
  typedef Vc::SimdArray < _value_type , _vsize > base_t ;

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
  typedef _value_type value_type ;
  static const size_type vsize = _vsize ;
  static const int ivsize = _vsize ;      // finessing for g++

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

  // special c'tor for zimt::simd_type of unsigned char as rhs

  vc_simd_type & operator=
    ( const simd_type < unsigned char , vsize > & rhs )
  {
    to_base().load ( ( unsigned char * ) ( & rhs ) ) ;
//     for ( size_type i = 0 ; i < vsize ; i++ )
//       (*this) [ i ] = rhs [ i ] ;
    return *this ;
  }

  vc_simd_type & operator=
    ( const simd_type < unsigned char , vsize > && rhs )
  {
    to_base().load ( ( unsigned char * ) ( & rhs ) ) ;
//     for ( size_type i = 0 ; i < vsize ; i++ )
//       (*this) [ i ] = rhs [ i ] ;
    return *this ;
  }

  vc_simd_type ( const simd_type < unsigned char , vsize > & ini )
  {
    *this = ini ;
  }

  vc_simd_type ( const simd_type < unsigned char , vsize > && ini )
  {
    *this = ini ;
  }

  // assignment from equally-sized container. Most containers use std::size_t
  // for the template argument defining the number of elements they hold,
  // but some (notably vigra::TinyVector) use int, which is probably a relic
  // from times when non-type template arguments were of a restricted type
  // set only. By providing a specialization for SIZE_TYPE int, we make
  // equally-sized vigra::TinyVectors permitted initializers.
  // the c'tor from an equally-sized container also uses the corresponding
  // operator= overload, so we use one macro for both.
  // we also need two different variants of vsize for g++; clang++ accepts
  // size_type vsize for both places where VSZ is used, but g++ requires
  // an integer.
  // Note that the rhs can use any elementary type which can be legally
  // assigned to value_type. This allows transport of information from
  // differently typed objects, but there are no further constraints on
  // the types involved, which may degrade precision. It's the user's
  // responsibility to make sure such assignments have the desired effect
  // and overload them if necessary.

  #define BUILD_FROM_CONTAINER(SIZE_TYPE,VSZ) \
    template < typename U , template < typename , SIZE_TYPE > class V > \
    vc_simd_type & operator= ( const V < U , VSZ > & rhs ) \
    { \
      static_assert ( vsize == VSZ , "incompatible vector size" ) ; \
      for ( size_type i = 0 ; i < vsize ; i++ ) \
        (*this) [ i ] = rhs [ i ] ; \
      return *this ; \
    } \
    template < typename U , template < typename , SIZE_TYPE > class V > \
    vc_simd_type ( const V < U , VSZ > & ini ) \
    { \
      *this = ini ; \
    }

  BUILD_FROM_CONTAINER(std::size_t,vsize)

  #undef BUILD_FROM_CONTAINER

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

  // use Vc's IndexesFromZero, Zero and One.

  using base_t::IndexesFromZero ;
  using base_t::Zero ;
  using base_t::One ;
  
  // iota() is a synonym for IndexesFromZero

  static const vc_simd_type iota()
  {
    return IndexesFromZero() ;
  }

  // variant which starts from a different starting point and optionally
  // uses steps other than one.

  static const index_type IndexesFrom ( const index_ele_type & start ,
                                        const index_ele_type & step )
  {
    return ( ( IndexesFromZero() * step ) + start ) ;
  }

  // overload staring from zero, but with steps != 1

  static const index_type IndexesFrom ( const index_ele_type & step )
  {
    return ( ( IndexesFromZero() * step ) ) ;
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
    to_base().gather ( p_src , _indexes.to_base() ) ;
  }

  template < typename _index_type >
  void scatter ( value_type * const p_trg ,
                 const _index_type & _indexes ) const
  {
    to_base().scatter ( p_trg , _indexes.to_base() ) ;
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

  void rgather ( const value_type * const p_src ,
                 const index_ele_type & step )
  {
    gather ( p_src , IndexesFrom ( step ) ) ;
  }

  void rscatter ( value_type * p_trg ,
                  const index_ele_type & step ) const
  {
    scatter ( p_trg , IndexesFrom ( step ) ) ;
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
  BROADCAST_STD_FUNC(tan)
  BROADCAST_STD_FUNC(asin)
  BROADCAST_STD_FUNC(acos)
  BROADCAST_STD_FUNC(atan)

  #undef BROADCAST_STD_FUNC

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
    vc_simd_type OPFUNC ( const vc_simd_type & rhs ) const \
    { \
      CONSTRAINT \
      return to_base() OP rhs.to_base() ; \
    } \
    vc_simd_type OPFUNC ( const value_type & rhs ) const \
    { \
      CONSTRAINT \
      return to_base() OP rhs ; \
    } \
    friend vc_simd_type OPFUNC ( const value_type & lhs , \
                                 const vc_simd_type & rhs ) \
    { \
      CONSTRAINT \
      return lhs OP rhs.to_base() ; \
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

// template < typename T , std::size_t N >
// struct allocator_traits < vc_simd_type < T , N > >
// {
//   typedef Vc::Allocator < vc_simd_type < T , N > >
//     type ;
// } ;

} ;

#endif // #define VSPLINE_VC_SIMD_TYPE_H
