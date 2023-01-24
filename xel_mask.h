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

/*! \file xel_mask.h

    \brief adds masks to xel_t used as SIMD type
*/

typedef XEL < bool , vsize > MaskType ;
typedef XEL < bool , vsize > mask_type ;

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
