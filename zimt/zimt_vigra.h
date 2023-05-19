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

/*! \file zimt_vigra.h

    \brief code to use vspline data with zimt and vigra with zimt data
*/

#ifndef ZIMT_VIGRA_H

#include <vigra/tinyvector.hxx>
#include <vigra/multi_array.hxx>
#include <zimt/xel.h>
#include <zimt/array.h>

namespace zimt
{
// internally, both vigra::TinyVector and zimt::xel_t are just
// a small C-style array of N T, so we can reinterpret-cast one
// to the other.

template < typename T , int N >
zimt::xel_t < T , N > &
to_zimt ( vigra::TinyVector < T , N > & v )
{
  return reinterpret_cast < zimt::xel_t < T , N > & > ( v ) ;
}

template < typename T , int N >
const zimt::xel_t < T , N > &
to_zimt ( const vigra::TinyVector < T , N > & v )
{
  return reinterpret_cast < const zimt::xel_t < T , N > & > ( v ) ;
}

template < typename T , std::size_t N >
vigra::TinyVector < T , N > &
to_vigra ( zimt::xel_t < T , N > & v )
{
  return reinterpret_cast < vigra::TinyVector < T , N > & > ( v ) ;
}

template < typename T , std::size_t N >
const vigra::TinyVector < T , N > &
to_vigra ( const zimt::xel_t < T , N > & v )
{
  return reinterpret_cast < const vigra::TinyVector < T , N > & > ( v ) ;
}

// next we have an adapter to produce a zimt::view_t from a
// vigra::MultiArrayView of TinyVectors, and of fundamentals.

template < unsigned int D , typename T , int N >
zimt::view_t < D , zimt::xel_t < T , N > >
to_zimt ( vigra::MultiArrayView
            < D , vigra::TinyVector < T , N > > & v )
{
  typedef zimt::xel_t < T , N > dtype ;

  return { (dtype*) v.data() ,
           to_zimt ( v.stride() ) ,
           to_zimt ( v.shape() ) } ;
}

template < unsigned int D , typename T >
zimt::view_t < D , T >
to_zimt ( vigra::MultiArrayView < D , T > & v )
{
  return { v.data() ,
           to_zimt ( v.stride() ) ,
           to_zimt ( v.shape() ) } ;
}

// and the other way round.

template < std::size_t D , typename T >
vigra::MultiArrayView
  < static_cast < unsigned int > ( D ) ,
    T
  >
to_vigra ( zimt::view_t < D , T > & v )
{
  return { to_vigra ( v.shape ) ,
           to_vigra ( v.strides ) ,
           v.data() } ;
}

template < std::size_t D , typename T , std::size_t N >
vigra::MultiArrayView
  < static_cast < unsigned int > ( D ) ,
    vigra::TinyVector < T , static_cast < int > ( N ) >
  >
to_vigra ( zimt::view_t < D , zimt::xel_t < T , N > > & v )
{
  typedef vigra::TinyVector < T , static_cast < int > ( N ) > dtype ;

  return { to_vigra ( v.shape ) ,
           to_vigra ( v.strides ) ,
           (dtype*) v.data() } ;
}

} ; // namespace zimt

#define ZIMT_VIGRA_H
#endif // #ifndef ZIMT_VIGRA_H

