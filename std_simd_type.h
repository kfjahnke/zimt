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

    \brief SIMD type derived from std::simd

    To use this header, an implementation of std::simd has to be
    installed, and the -std=c++17 option is needed as well.
    It has been tried with clang++ and g++; you'll need a
    recent version.

*/

#ifndef VSPLINE_STD_SIMD_TYPE_H
#define VSPLINE_STD_SIMD_TYPE_H

#ifndef VSPLINE_VECTOR_NBYTES

#define VSPLINE_VECTOR_NBYTES 64

#endif

#include <iostream>
#include <experimental/simd>

namespace zimt
{

/// class template simd_type provides a fixed-size SIMD type.
/// This implementation of zimt::simd_type uses std::simd as
/// base class. This class is used as a stand-in for Vc::SimdArray
/// - it does not cover the whole interface, but a reasonably
/// large subset - my choice of SimdArray functionality is what
/// I need in lux/zimt.
/// Most of the 'loop variant' of simd_type has been ported to
/// use std::simd instead, making use of std::simd's
/// - constructors
/// - copy_from and copy_to
/// - masks and where expressions
/// - operator functions
/// - overloads for several mathematical functions
/// - min/max

template < typename _value_type ,
           std::size_t _vsize >
struct simd_type
: private std::experimental::simd
          < _value_type ,
            std::experimental::simd_abi::fixed_size < _vsize >
          >
{
  typedef std::experimental::simd_abi::fixed_size < _vsize > abi_t ;

  typedef std::size_t size_type ;
  typedef _value_type value_type ;
  static const size_type vsize = _vsize ;

  typedef std::experimental::simd < value_type , abi_t > base_t ;
  typedef std::experimental::simd < int , abi_t > index_type ;
  using typename base_t::mask_type ;

  // provide the size as a constexpr

  static constexpr size_type size()
  {
    return vsize ;
  }

  typedef mask_type MaskType ;
  typedef index_type IndexType ;

  // operator[] is mapped to std::simd element access

  using base_t::operator[] ;

  // assignment from a value_type. The assignment is coded as a loop,
  // but it should be obvious to the compiler's loop vectorizer that
  // the loop is a 'SIMD operation in disguise', so here we have the
  // first appearance of 'goading'.

  simd_type & operator= ( const value_type & rhs )
  {
    to_base() = base_t ( rhs ) ;
    return *this ;
  }

  // c'tor from value_type. We use the assignment operator for
  // initialization.

  simd_type ( const value_type & ini )
  : base_t ( ini )
  { }

  simd_type ( const base_t & ini )
  : base_t ( ini )
  { }

  // these two c'tors are left in default mode

  simd_type() = default ;
  simd_type ( const simd_type & ) = default ;

  base_t & to_base()
  {
    return ( * static_cast < base_t * > ( this ) ) ;
  }

  const base_t & to_base() const
  {
    return ( * static_cast < const base_t * const > ( this ) ) ;
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
    simd_type & operator= ( const V < U , VSZ > & rhs ) \
    { \
      for ( size_type i = 0 ; i < vsize ; i++ ) \
        to_base() [ i ] = value_type ( rhs [ i ] ) ; \
      return *this ; \
    } \
    template < typename U , template < typename , SIZE_TYPE > class V > \
    simd_type ( const V < U , VSZ > & ini ) \
    { \
      *this = ini ; \
    }

  BUILD_FROM_CONTAINER(std::size_t,vsize)

  #undef BUILD_FROM_CONTAINER

  // because all simd_type objects are of a distinct size
  // explicitly coded in template arg vsize, we can initialize
  // from an initializer_list. This is probably not very fast,
  // but it's nice to have for experimentation.

  simd_type ( const std::initializer_list < value_type > & rhs )
  {
    assert ( rhs.size() == vsize ) ; // TODO: prefer constexpr
    std::size_t i = 0 ;
    for ( const auto & src : rhs )
      (*this) [ i++ ] = src ;
  }

  static const simd_type iota()
  {
    simd_type result ;
    for ( size_type i = 0 ; i < vsize ; i++ )
      result [ i ] = value_type ( i ) ;
    return result ;
  }

  // mimick Vc's IndexesFromZero. This function produces an index
  // vector filled with indexes starting with zero.

  static const index_type IndexesFromZero()
  {
    typedef typename index_type::value_type IT ;
    static const IT ceiling = std::numeric_limits < IT > :: max() ;
    assert ( ( vsize - 1 ) <= std::size_t ( ceiling ) ) ;

    index_type ix ;
    for ( size_type i = 0 ; i < vsize ; i++ )
      ix [ i ] = int ( i ) ;
    return ix ;
  }

  // variant which starts from a different starting point and optionally
  // uses steps other than one.

  static const index_type IndexesFrom ( std::size_t start ,
                                        std::size_t step = 1 )
  {
    typedef typename index_type::value_type IT ;
    static const IT ceiling = std::numeric_limits < IT > :: max() ;
    assert ( start + ( vsize - 1 ) * step <= std::size_t ( ceiling ) ) ;

    return ( IndexesFromZero() * int(step) ) + int(start) ;
  }

  // functions Zero and One produce simd_type objects filled with
  // 0, or 1, respectively

  static const simd_type Zero()
  {
    return simd_type ( value_type ( 0 ) ) ;
  }

  static const simd_type One()
  {
    return simd_type ( value_type ( 1 ) ) ;
  }

  // echo the vector to a std::ostream, read it from an istream

  friend std::ostream & operator<< ( std::ostream & osr ,
                                     simd_type it )
  {
    osr << "(( " ;
    for ( size_type i = 0 ; i < vsize - 1 ; i++ )
      osr << it [ i ] << ", " ;
    osr << it [ vsize - 1 ] << " ))" ;
    return osr ;
  }

  friend std::istream & operator>> ( std::istream & isr ,
                                     simd_type it )
  {
    for ( size_type i = 0 ; i < vsize ; i++ )
      isr >> it [ i ] ;
    return isr ;
  }

  // memory access functions, which load and store vector data.
  // We start out with functions transporting data from memory into
  // the simd_type. Some of these operations have corresponding
  // c'tors which use the member function to initialize to_base().

  // load delegates to std::simd::copy_from. TODO: consider
  // overalignment

  void load ( const value_type * const p_src )
  {
    to_base().copy_from ( p_src ,
                          std::experimental::element_aligned_tag() ) ;
  }

  // std::simd does not offer gather/scatter, but it offers a
  // c'tor taking a functor to set the elements. In theory, this
  // is a good idea, because the optimizer might realize that
  // the sum of invocations of the functor can be represented
  // by a gather/scatter operation, but how well this works is
  // a different matter and mileage varies.

#define GS_LAMBDA

#ifdef GS_LAMBDA

  template < typename index_type >
  void gather ( const value_type * const p_src ,
                const index_type & indexes )
  {
    // assign base_t object created by gen-type c'tor
    to_base() = base_t ( [&] ( const size_t & i )
                    { return p_src [ indexes[i] ] ; } ) ;
  }

#else

  template < typename index_type >
  void gather ( const value_type * const p_src ,
                const index_type & indexes )
  {
    for ( std::size_t i = 0 ; i < vsize ; i++ )
      (*this)[i] = p_src [ indexes [ i ] ] ;
  }

#endif

  // c'tor from pointer and indexes, uses gather

  template < typename index_type >
  simd_type ( const value_type * const p_src ,
              const index_type & indexes )
  {
    gather ( p_src , indexes ) ;
  }

  // store saves the content of the vector to memory

  void store ( value_type * const p_trg ) const
  {
    to_base().copy_to ( p_trg ,
                        std::experimental::element_aligned_tag() ) ;
  }

  // scatter is the reverse operation to gather

#ifdef GS_LAMBDA

  template < typename index_type >
  void scatter ( value_type * const p_trg ,
                 const index_type & indexes ) const
  {
    // gen-type c'tor is only used for side effects; let the compiler
    // figure out that the result is unused.

    base_t dummy ( [&] ( const size_t & i )
                   { return p_trg [ indexes[i] ] = to_base()[i] ; } ) ;
  }

#else

  template < typename index_type >
  void scatter ( value_type * const p_trg ,
                 const index_type & indexes ) const
  {
    for ( std::size_t i = 0 ; i < vsize ; i++ )
      p_trg [ indexes [ i ] ] = (*this)[i] ;
  }

#endif

  // 'regular' gather and scatter, accessing strided memory so that the
  // first address visited is p_src/p_trg, and successive addresses are
  // 'step' apart - in units of T. Might also be done with goading, the
  // loop should autovectorize.

  void rgather ( const value_type * const p_src ,
                 const std::size_t & step )
  {
    auto indexes = IndexesFrom ( 0 , step ) ;
    gather ( p_src , indexes ) ;
  }

  void rscatter ( value_type * p_trg ,
                  const std::size_t & step ) const
  {
    auto indexes = IndexesFrom ( 0 , step ) ;
    scatter ( p_trg , indexes ) ;
  }

  // apply functions from namespace std to each element in a vector,
  // or to each corresponding set of elements in a set of vectors
  // - going up to three for fma.
  // many standard functions autovectorize well. Note that the
  // autovectorization of standard functions often needs additional
  // compiler flags, like, e.g., -fno-math-errno for clang++, to
  // produce hardware SIMD instructions.

  #define BROADCAST_STD_FUNC(FUNC) \
    friend simd_type FUNC ( simd_type arg ) \
    { \
      return FUNC ( arg.to_base() ) ; \
    }

    // TODO: getting zero back for negative args, hence no BROADCAST_STD_FUNC
    // this happens with clang++ only, I opened an issue with VcDevel/std-simd:
    // https://github.com/VcDevel/std-simd/issues/31

//   BROADCAST_STD_FUNC(abs)

  friend simd_type abs ( simd_type arg )
  {
    arg ( arg < 0 ) = - arg ;
    return arg ;
  }

  BROADCAST_STD_FUNC(trunc)

  BROADCAST_STD_FUNC(round)
  BROADCAST_STD_FUNC(floor)
  BROADCAST_STD_FUNC(ceil)
  BROADCAST_STD_FUNC(log)
  BROADCAST_STD_FUNC(exp)
  BROADCAST_STD_FUNC(sqrt)

  // the support for autovectorization of trigonometric functions is
  // sketchy - e.g. the clang++ reference does not mention them as
  // functions which autovectorize. Vc offers hand-coded trigonometric
  // functions which might be worth while porting to std::simd, but so
  // far I haven't seen this happen. In my application, this results in
  // bad performance when these functions are used.

  BROADCAST_STD_FUNC(tan)
  BROADCAST_STD_FUNC(asin)
  BROADCAST_STD_FUNC(acos)
  BROADCAST_STD_FUNC(atan)

  // TODO: odd: with clang++, sin and cos don't perform as expected;
  // using a loop does the trick:

#ifdef __clang__

  friend simd_type cos ( simd_type arg )
  {
    simd_type result ;
    for ( std::size_t i = 0 ; i < size() ; i++ )
      result[i] = std::cos ( arg[i] ) ;
    return result ;
  }

  friend simd_type sin ( simd_type arg )
  {
    simd_type result ;
    for ( std::size_t i = 0 ; i < size() ; i++ )
      result[i] = std::sin ( arg[i] ) ;
    return result ;
  }

#else

  BROADCAST_STD_FUNC(sin)
  BROADCAST_STD_FUNC(cos)

#endif

  #undef BROADCAST_STD_FUNC

  #define BROADCAST_STD_FUNC2(FUNC) \
    friend simd_type FUNC ( simd_type arg1 , \
                            simd_type arg2 ) \
    { \
      return FUNC ( arg1.to_base() , arg2.to_base() ) ; \
    }

  // a short note on atan2: Vc provides a hand-written vectorized version
  // of atan2 which is especially fast and superior to autovectorized code.

  BROADCAST_STD_FUNC2(atan2)
  BROADCAST_STD_FUNC2(pow)

  #undef BROADCAST_STD_FUNC2

  #define BROADCAST_STD_FUNC3(FUNC) \
    friend simd_type FUNC ( simd_type arg1 , \
                            simd_type arg2 , \
                            simd_type arg3 ) \
    { \
      return FUNC ( arg1.to_base() , arg2.to_base() , arg3.to_base() ) ; \
    }

  BROADCAST_STD_FUNC3(fma)

  #undef BROADCAST_STD_FUNC3

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
  // Only value_type and simd_type itto_base() are taken as rhs arguments.

  #define OPEQ_FUNC(OPFUNC,OPEQ,CONSTRAINT) \
    simd_type & OPFUNC ( value_type rhs ) \
    { \
      CONSTRAINT \
      to_base() OPEQ rhs ; \
      return *this ; \
    } \
    simd_type & OPFUNC ( simd_type rhs ) \
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
    simd_type OPFUNC ( simd_type rhs ) const \
    { \
      CONSTRAINT \
      return this->to_base() OP rhs.to_base() ; \
    } \
    simd_type OPFUNC ( value_type rhs ) const \
    { \
      CONSTRAINT \
      return this->to_base() OP rhs ; \
    } \
    friend simd_type OPFUNC ( value_type lhs , \
                              simd_type rhs ) \
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
    simd_type OPFUNC() const \
    { \
      return OP this->to_base() ; \
    }

  OP_FUNC(operator-,-,)
  OP_FUNC(operator!,!,BOOL_ONLY)
  OP_FUNC(operator~,~,INTEGRAL_ONLY)

  #undef OP_FUNC

  // provide methods to produce a mask on comparing a vector
  // with another vector or a value_type.

  #define COMPARE_FUNC(OP,OPFUNC) \
  friend mask_type OPFUNC ( simd_type lhs , \
                            simd_type rhs ) \
  { \
    return lhs.to_base() OP rhs.to_base() ; \
  } \
  friend mask_type OPFUNC ( simd_type lhs , \
                            value_type rhs ) \
  { \
    return lhs.to_base() OP rhs ; \
  } \
  friend mask_type OPFUNC ( value_type lhs , \
                            simd_type rhs ) \
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

  // next we define a masked vector as an object holding two references:
  // one reference to a mask type, determining which of the vector's
  // elements will be 'open' to an effect, and one reference to a vector,
  // which will be affected by the operation.
  // The resulting object will only be viable as long as the referred-to
  // mask and vector are alive - it's meant as a construct to be processed
  // in the same scope, as the lhs of an assignment, typically using
  // notation introduced by Vc: a vector's operator() is overloaded to
  // to produce a masked_type when called with a mask_type object, and
  // the resulting masked_type object is then assigned to.
  // Note that this does not have any effect on those values in 'whither'
  // for which the mask is false. They remain unchanged.

  typedef std::experimental::where_expression < mask_type , base_t > we_t ;

  struct masked_type
  {
    mask_type whether ;   // if the mask is true at whether[i]
    simd_type & whither ; // whither[i] will be assigned to

    masked_type ( mask_type _whether ,
                  simd_type & _whither )
    : whether ( _whether ) ,
      whither ( _whither )
      { }

    // for the masked vector, we define the complete set of assignments:

    #define OPEQ_FUNC(OPFUNC,OPEQ,CONSTRAINT) \
      simd_type & OPFUNC ( value_type rhs ) \
      { \
        CONSTRAINT \
        we_t ( whether , whither.to_base() ) OPEQ rhs ; \
        return whither ; \
      } \
      simd_type & OPFUNC ( simd_type rhs ) \
      { \
        CONSTRAINT \
        we_t ( whether , whither.to_base() ) OPEQ rhs.to_base() ; \
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

  masked_type operator() ( mask_type mask )
  {
    return masked_type ( mask , *this ) ;
  }

  // member functions at_least and at_most. These functions provide the
  // same functionality as max, or min, respectively. Given simd_type X
  // and some threshold Y, X.at_least ( Y ) == max ( X , Y )
  // Having the functionality as a member function makes it easy to
  // implement, e.g., min as: min ( X , Y ) { return X.at_most ( Y ) ; }

  #define CLAMP(FNAME,REL) \
    simd_type FNAME ( simd_type threshold ) const \
    { \
      return REL ( to_base() , threshold.to_base() ) ; \
    } \
    simd_type FNAME ( value_type threshold ) const \
    { \
      return REL ( to_base() , threshold ) ; \
    } \

  CLAMP(at_least,max)
  CLAMP(at_most,min)

  #undef CLAMP

  // sum of vector elements. Note that there is no type promotion; the
  // summation is done to value_type. Caller must make sure that overflow
  // is not a problem.

  value_type sum() const
  {
    value_type s ( 0 ) ;
    for ( std::size_t e = 0 ; e < vsize ; e++ )
      s += (*this) [ e ] ;
    return s ;
  }
} ;

} ;

#endif // #define VSPLINE_SIMD_TYPE_H
