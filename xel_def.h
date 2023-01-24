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

#ifndef XEL_DEF_INCLUDED

template < typename _value_type ,
           std::size_t _vsize >
class XEL
{
  // storage of data is in a simple C array. This array is private,
  // and the only access to it is via member functions. Using a plain
  // C array will not overalign the data, even though this may be
  // desirable, especially on older hardware. We rely on the
  // compiler to handle this as efficiently as possible.

  _value_type _store [ _vsize ] ;

public:

  // some common typedefs for a container type

  typedef std::size_t size_type ;
  typedef _value_type value_type ;
  static const size_type vsize = _vsize ;
  static const int ivsize = _vsize ;      // finessing for g++

  // provide the size as a constexpr

  static constexpr size_type size()
  {
    return vsize ;
  }

  // types used for masks and index vectors. In terms of 'true' SIMD
  // arithmetics, these definitions may not be optimal - especially the
  // definition of a mask as a XEL of bool is questionable - one
  // might consider using a bit field or a sufficiently large integral
  // type. But using a XEL of bool makes processing simple, in
  // a way it's the 'generic' mask type, whereas SIMD masks used by
  // the hardware are the truly 'exotic' types. The problem here is
  // the way C++ encodes booleans - they are usually encoded as some
  // smallish integral type, rather than a single bit.

  // we define both the 'old school' and the 'camel case' variants

  typedef XEL < bool , vsize > mask_type ;
  typedef XEL < int , vsize > index_type ;

  typedef XEL < bool , vsize > MaskType ;
  typedef XEL < int , vsize > IndexType ;

  // operator[] is mapped to ordinary element access on a C vector.
  // This is the only place where we explicitly access _store, the
  // remainder of the code uses operator[].

  value_type & operator[] ( const size_type & i )
  {
    return _store[i] ;
  }

  const value_type & operator[] ( const size_type & i ) const
  {
    return _store[i] ;
  }

  // assignment from a value_type. The assignment is coded as a loop,
  // but it should be obvious to the compiler's loop vectorizer that
  // the loop is a 'SIMD operation in disguise', so here we have the
  // first appearance of 'goading'.

  XEL & operator= ( const value_type & rhs )
  {
    for ( size_type i = 0 ; i < vsize ; i++ )
      (*this) [ i ] = rhs ;
    return *this ;
  }

  // c'tor from value_type. We use the assignment operator for
  // initialization.

  XEL ( const value_type & ini )
  {
    *this = ini ;
  }

  // these two c'tors are left in default mode

  XEL() = default ;
  XEL ( const XEL & ) = default ;

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
    XEL & operator= ( const V < U , VSZ > & rhs ) \
    { \
      for ( size_type i = 0 ; i < vsize ; i++ ) \
        (*this) [ i ] = rhs [ i ] ; \
      return *this ; \
    } \
    template < typename U , template < typename , SIZE_TYPE > class V > \
    XEL ( const V < U , VSZ > & ini ) \
    { \
      *this = ini ; \
    }

  BUILD_FROM_CONTAINER(std::size_t,vsize)
  BUILD_FROM_CONTAINER(int,ivsize)

  #undef BUILD_FROM_CONTAINER

  /*
  // this operator= overload caters for Vc::SimdArray as rhs type.
  // Vc::SimdArray requires additional template arguments - 'an
  // unfortunate implementation detail shining through'.
  // We don't want to refer explicitly to Vc::SimdArray here, so
  // we just code a template which will 'catch' it. In result, we
  // can now easily assign a Vc::SimdArray to a zimt::XEL.
  // The reverse operation can't be coded like this, though, since
  // assignment operators and c'tors have to be coded as members of
  // the 'receiving end'. See 'offload' for an efficient way of
  // moving XEL data to Vc::SimdArrays

  template < typename U ,
             typename V ,
             std::size_t Wt ,
             template < typename , std::size_t ,
                        typename , std::size_t > class W >
  XEL & operator= ( const W < U , vsize , V , Wt > & rhs )
  {
    for ( size_type i = 0 ; i < vsize ; i++ )
      (*this) [ i ] = rhs [ i ] ;
    return *this ;
  }

  // corresponding c'tor:

  template < typename U ,
             typename V ,
             std::size_t Wt ,
             template < typename , std::size_t ,
                        typename , std::size_t > class W >
  XEL ( const W < U , vsize , V , Wt > & ini )
  {
    *this = ini ;
  }
  */

  static const XEL iota()
  {
    XEL result ;
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
    assert ( start + ( vsize - 1 ) * step <= std::size_t ( ceiling ) ) ;

    return ( IndexesFromZero() * int(step) ) + int(start) ;
  }

  // functions Zero and One produce XEL objects filled with
  // 0, or 1, respectively

  static const XEL Zero()
  {
    return XEL ( value_type ( 0 ) ) ;
  }

  static const XEL One()
  {
    return XEL ( value_type ( 1 ) ) ;
  }

  // echo the vector to a std::ostream, read it from an istream

  friend std::ostream & operator<< ( std::ostream & osr ,
                                     XEL it )
  {
    osr << "(" ;
    for ( size_type i = 0 ; i < vsize - 1 ; i++ )
      osr << it [ i ] << ", " ;
    osr << it [ vsize - 1 ] << ")" ;
    return osr ;
  }

  friend std::istream & operator>> ( std::istream & isr ,
                                     XEL it )
  {
    for ( size_type i = 0 ; i < vsize ; i++ )
      isr >> it [ i ] ;
    return isr ;
  }

  // memory access functions, which load and store vector data.
  // We start out with functions transporting data from memory into
  // the XEL. Some of these operations have corresponding
  // c'tors which use the member function to initialize (*this).

  // load uses a simple loop, which is about as easy to recognize as
  // an autovectorizable construct as it gets:

  void load ( const value_type * const p_src )
  {
    for ( size_type i = 0 ; i < vsize ; i++ )
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
    for ( size_type i = 0 ; i < vsize ; i++ )
      (*this) [ i ] = p_src [ indexes [ i ] ] ;
  }

  // c'tor from pointer and indexes, uses gather

  template < typename index_type >
  XEL ( const value_type * const p_src ,
              const index_type & indexes )
  {
    gather ( p_src , indexes ) ;
  }

  // store saves the content of the container to memory

  void store ( value_type * const p_trg ) const
  {
    for ( size_type i = 0 ; i < vsize ; i++ )
      p_trg [ i ] = (*this) [ i ] ;
  }

  // scatter is the reverse operation to gather, see the comments there.

  template < typename index_type >
  void scatter ( value_type * const p_trg ,
                 const index_type & indexes ) const
  {
    for ( size_type i = 0 ; i < vsize ; i++ )
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
  XEL shuffle ( index_type indexes )
  {
    XEL result ;
    for ( size_type i = 0 ; i < vsize ; i++ )
      result [ i ] = (*this) [ indexes [ i ] ] ;
    return result ;
  }

  // operator[] with an index_type argument performs the same
  // operation

  XEL operator[] ( index_type indexes )
  {
    return shuffle ( indexes ) ;
  }

  // apply functions from namespace std to each element in a vector,
  // or to each corresponding set of elements in a set of vectors
  // - going up to three for fma.
  // many standard functions autovectorize well. Note that the
  // autovectorization of standard functions often needs additional
  // compiler flags, like, e.g., -fno-math-errno for clang++, to
  // produce hardware SIMD instructions.

  #define BROADCAST_STD_FUNC(FUNC) \
    friend XEL FUNC ( XEL arg ) \
    { \
      XEL result ; \
      for ( size_type i = 0 ; i < vsize ; i++ ) \
        result [ i ] = FUNC ( arg [ i ] ) ; \
      return result ; \
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
    friend XEL FUNC ( XEL arg1 , \
                            XEL arg2 ) \
    { \
      XEL result ; \
      for ( size_type i = 0 ; i < vsize ; i++ ) \
        result [ i ] = FUNC ( arg1 [ i ] , arg2 [ i ] ) ; \
      return result ; \
    }

  // a short note on atan2: Vc provides a hand-written vectorized version
  // of atan2 which is especially fast and superior to autovectorized code.

  BROADCAST_STD_FUNC2(atan2)
  BROADCAST_STD_FUNC2(pow)

  #undef BROADCAST_STD_FUNC2

  #define BROADCAST_STD_FUNC3(FUNC) \
    friend XEL FUNC ( XEL arg1 , \
                            XEL arg2 , \
                            XEL arg3 ) \
    { \
      XEL result ; \
      for ( size_type i = 0 ; i < vsize ; i++ ) \
        result [ i ] = FUNC ( arg1 [ i ] , arg2 [ i ] , arg3[i] ) ; \
      return result ; \
    }

  BROADCAST_STD_FUNC3(fma)

  #undef BROADCAST_STD_FUNC3

  // macros used for the parameter 'CONSTRAINT' in the definitions
  // further down. Some operations are only allowed for integral types
  // or boolans. This might be enforced by enable_if, here we use a
  // static_assert with a clear error message.
  // TODO: might relax constraints by using 'std::is_convertible'
  // TODO: some operators make sense as simply manipulating the 'raw'
  // bits in the data (e.g. ~ or ^) - I currently limit them to int,
  // but this might be relaxed, casting the data to some unsigned
  // integral format of equal size.

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
  // Only value_type and XEL itself are taken as rhs arguments.

  #define OPEQ_FUNC(OPFUNC,OPEQ,CONSTRAINT) \
    XEL & OPFUNC ( value_type rhs ) \
    { \
      CONSTRAINT \
      for ( size_type i = 0 ; i < vsize ; i++ ) \
        (*this) [ i ] OPEQ rhs ; \
      return *this ; \
    } \
    XEL & OPFUNC ( XEL rhs ) \
    { \
      CONSTRAINT \
      for ( size_type i = 0 ; i < vsize ; i++ ) \
        (*this) [ i ] OPEQ rhs [ i ] ; \
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
    XEL OPFUNC ( XEL rhs ) const \
    { \
      CONSTRAINT \
      XEL help ; \
      for ( size_type i = 0 ; i < vsize ; i++ ) \
        help [ i ] = (*this) [ i ] OP rhs [ i ] ; \
      return help ; \
    } \
    XEL OPFUNC ( value_type rhs ) const \
    { \
      CONSTRAINT \
      XEL help ; \
      for ( size_type i = 0 ; i < vsize ; i++ ) \
        help [ i ] = (*this) [ i ] OP rhs ; \
      return help ; \
    } \
    friend XEL OPFUNC ( value_type lhs , \
                              XEL rhs ) \
    { \
      CONSTRAINT                                   \
      XEL help ; \
      for ( size_type i = 0 ; i < vsize ; i++ ) \
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

  #define OP_FUNC(OPFUNC,OP,CONSTRAINT) \
    XEL OPFUNC() const \
    { \
      XEL help ; \
      for ( size_type i = 0 ; i < vsize ; i++ ) \
        help [ i ] = OP (*this) [ i ] ; \
      return help ; \
    }

  OP_FUNC(operator-,-,)
  OP_FUNC(operator!,!,BOOL_ONLY)
  OP_FUNC(operator~,~,INTEGRAL_ONLY)

  #undef OP_FUNC

  // provide methods to produce a mask on comparing a vector
  // with another vector or a value_type.

  #define COMPARE_FUNC(OP,OPFUNC) \
  friend mask_type OPFUNC ( XEL lhs , \
                            XEL rhs ) \
  { \
    mask_type m ; \
    for ( size_type i = 0 ; i < vsize ; i++ ) \
      m [ i ] = ( lhs [ i ] OP rhs [ i ] ) ; \
    return m ; \
  } \
  friend mask_type OPFUNC ( XEL lhs , \
                            value_type rhs ) \
  { \
    mask_type m ; \
    for ( size_type i = 0 ; i < vsize ; i++ ) \
      m [ i ] = ( lhs [ i ] OP rhs ) ; \
    return m ; \
  } \
  friend mask_type OPFUNC ( value_type lhs , \
                            XEL rhs ) \
  { \
    mask_type m ; \
    for ( size_type i = 0 ; i < vsize ; i++ ) \
      m [ i ] = ( lhs OP rhs [ i ] ) ; \
    return m ; \
  }

  COMPARE_FUNC(<,operator<) ;
  COMPARE_FUNC(<=,operator<=) ;
  COMPARE_FUNC(>,operator>) ;
  COMPARE_FUNC(>=,operator>=) ;
  COMPARE_FUNC(==,operator==) ;
  COMPARE_FUNC(!=,operator!=) ;

  #undef COMPARE_FUNC

  // next we define a masked vector as an object holding two pieces of
  // information: one mask, determining which of the vector's elements
  // will be 'open' to an effect, and one reference to a vector, which
  // will be affected by an assignment or augmented assignment to the
  // masked_type object. The mask is not held by reference - it is
  // typically created right inside an invocation of the V(M) = ...
  // idiom and will only persist when copied to the masked_type object.
  // The compiler will take care of the masks's lifetime and storage and
  // make sure the process is efficient and avoids unnecessary copying.
  // The resulting object will only be viable as long as the referred-to
  // vector is kept 'alive' - it's meant as a construct to be processed
  // in the same scope, as the lhs of an assignment, typically using
  // notation introduced by Vc: a vector's operator() is overloaded to
  // to produce a masked_type when called with a mask_type object, and
  // the resulting masked_type object is then assigned to - the V(M)=...
  // idiom mentioned above.
  // Note that this does not have any effect on those values in 'whither'
  // for which the mask is false. They remain unchanged.

  struct masked_type
  {
    mask_type whether ;   // if the mask is true at whether[i]
    XEL & whither ; // whither[i] will be assigned to

    masked_type ( mask_type _whether ,
                  XEL & _whither )
    : whether ( _whether ) ,
      whither ( _whither )
      { }

    // for the masked vector, we define the complete set of assignments:

    #define OPEQ_FUNC(OPFUNC,OPEQ,CONSTRAINT) \
      XEL & OPFUNC ( value_type rhs ) \
      { \
        CONSTRAINT \
        for ( size_type i = 0 ; i < vsize ; i++ ) \
        { \
          if ( whether [ i ] ) \
            whither [ i ] OPEQ rhs ; \
        } \
        return whither ; \
      } \
      XEL & OPFUNC ( XEL rhs ) \
      { \
        CONSTRAINT \
        for ( size_type i = 0 ; i < vsize ; i++ ) \
        { \
          if ( whether [ i ] ) \
            whither [ i ] OPEQ rhs [ i ] ; \
        } \
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

  // next we have a few functions creating masks - mainly for Vc
  // compatibility.

  static mask_type isnegative ( const XEL & rhs )
  {
    return ( rhs < value_type(0) ) ;
  }

  static mask_type isfinite ( const XEL & rhs )
  {
    for ( std::size_t i = 0 ; i < vsize ; i++ )
    {
      if ( isInf ( rhs[i] ) )
        return false ;
    }
    return true ;
  }

  static mask_type isnan ( const XEL & rhs )
  {
    for ( std::size_t i = 0 ; i < vsize ; i++ )
    {
      if ( isNan ( rhs[i] ) )
        return true ;
    }
    return false ;
  }

  // member functions at_least and at_most. These functions provide the
  // same functionality as max, or min, respectively. Given XEL X
  // and some threshold Y, X.at_least ( Y ) == max ( X , Y )
  // Having the functionality as a member function makes it easy to
  // implement, e.g., min as: min ( X , Y ) { return X.at_most ( Y ) ; }

  #define CLAMP(FNAME,REL) \
    XEL FNAME ( XEL threshold ) const \
    { \
      XEL result ( threshold ) ; \
      for ( std::size_t i = 0 ; i < vsize ; i++ ) \
      { \
        if ( (*this) [ i ] REL threshold ) \
          result [ i ] = (*this) [ i ] ; \
      } \
      return result ; \
    }

  CLAMP(at_least,>)
  CLAMP(at_most,<)

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

#define XEL_DEF_INCLUDED
#endif
