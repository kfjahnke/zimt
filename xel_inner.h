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

/*! \file xel_inner.h

    \brief class definition code of vector-like arithmetic type
*/

/// this header is to fill in the class definition of a fixed-size
/// container type with arithmetic capability. In zimt, there are
/// currently two classes using this header: class xel_t and class
/// simd_type. The inner workings of the generated types are the
/// same, but they are used in different semantic slots: class xel_t
/// is used as a container to hold 'xel' data and their SIMDized
/// equivalents - stuff like pixels - data which have several
/// channels - like rhe R, G, and B channels of an RGB pixel.
/// We might split this header up into several parts to, e.g, remove
/// the masking and other SIMDish stuff from the generated class;
/// for now, both xel_t and simd_type will share the same code.

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

typedef XEL < int , vsize > index_type ;

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

// construction from a std::initializer_list

XEL ( const std::initializer_list < value_type > & rhs )
{
  assert ( rhs.size() == vsize ) ; // TODO: prefer constexpr
  value_type * trg = _store ;
  for ( const auto & src : rhs )
    *trg++ = src ;
}

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
  osr << "{ " ;
  for ( size_type i = 0 ; i < vsize - 1 ; i++ )
    osr << it [ i ] << ", " ;
  osr << it [ vsize - 1 ] << " }" ;
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
// For now I keep them in the common xel code, but they might
// be taken out to a separate file and included only by simd_type

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
      result [ i ] = std::FUNC ( arg1 [ i ] , arg2 [ i ] ) ; \
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
