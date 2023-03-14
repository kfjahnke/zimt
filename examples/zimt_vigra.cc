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

/*! \file zimt_vigra.cc

    \brief feeding vigra data to zimt::transform

    vigra::TinyVectors and zimt::xel_t are binary-compatible,
    and vigra::MultiArrayView and zimt::view_t are close relatives,
    sharing the same notion of nD arrays. This makes it quite easy
    to feed vigra data to zimt::transform and relatives, allowing
    legacy code to use zimt for fast multihtreaded SIMD processing
    of vigra data rather than relying on vigra::transformMultiArray.
    In this example, I demonstrate use of simple factory functions to
    create zimt views of vigra arrays and show how they can be
    used to feed the data to a zimt transform with a given zimt
    functor.

*/

#include "../include/zimt_vigra.h"

// Type for typical xel datum consisting of three float values
// and binary-compatible vigra::TinyVector

typedef zimt::xel_t < float , 3 > f3_t ;
typedef vigra::TinyVector < float , 3 > vf3_t ;

typedef typename zimt::vector_traits < float > :: type fv_t ;
typedef zimt::xel_t < fv_t , 3 > fv3_t ;
typedef vigra::TinyVector < fv_t , 3 > vg_vf3_t ;

// simple zimt functor, taking some type T as input and
// producing f3_t as output.

template < typename T >
struct amp13_t
: public zimt::unary_functor < T , f3_t , 16 >
{
  const float factor ;

  amp13_t ( const float & _factor )
  : factor ( _factor )
  { }

  template < typename I , typename O >
  void eval ( const I & in , O & out ) const
  {
    out = in * factor ;
  }
} ;

int main ( int argc , char * argv[] )
{
  // extent of the array we'll process (deliberately odd shape)

  const int w = 1921 , h = 1081 ;

  // we create a vigra::MultiArray holding the data and
  // initialize it

  vigra::MultiArray < 2 , vf3_t > a ( vigra::Shape2 ( w , h ) ) ;
  a = vf3_t { 1.0f , 2.0f , 3.0f } ;

  // create a zimt functor which prouces twice it's input

  amp13_t < f3_t > twice ( 2.0f ) ;

  // now we apply the functor to the array, using as_view to
  // present the data.

  zimt::apply < 2 > ( twice , zimt::as_view ( a ) ) ;

  // let's see a sample of the result

  std::cout << a [ { 100 , 100 } ] << std::endl ;
}
