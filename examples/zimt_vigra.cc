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
    In this example, I demonstrate two simple factory functions to
    create zimt views of vigra arrays and show how they can be
    used to feed the data to a zimt transform with a given zimt
    functor.

*/

#include <vigra/tinyvector.hxx>
#include <vigra/multi_array.hxx>

#include <zimt/zimt.h>

// Type for typical xel datum consisting of three float values
// and binary-compatible vigra::TinyVector

typedef zimt::xel_t < float , 3 > f3_t ;
typedef vigra::TinyVector < float , 3 > vf3_t ;

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

// next we have a few adapter functions to make connecting to
// vigra easier. First an adapter to convert a const reference
// to a vigra::TinyVector to a zimt::xel_t. This will be used
// to convert the strides and shape of the MultiArrayView.
// TODO: this might be used to convert all TinyVectors to
// corresponding xel_t, but may require narrowing the argument
// spectrum.
// TODO: might provide a zimt header for the purpose

template < typename T , int N >
zimt::xel_t < T , N > const &
as_xel ( vigra::TinyVector < T , N > const & v )
{
  return reinterpret_cast < zimt::xel_t < T , N > const & > ( v ) ;
}

// next we have an adapter to produce a zimt::view_t from a
// vigra::MultiArrayView of TinyVectors

template < unsigned int D , typename T , int N >
zimt::view_t < D , zimt::xel_t < T , N > >
as_view
  ( vigra::MultiArrayView < D , vigra::TinyVector < T , N > > & v )
{
  typedef zimt::xel_t < T , N > dtype ;

  return zimt::view_t < D , dtype >
    { (dtype*) v.data() ,
      as_xel ( v.stride() ) ,
      as_xel ( v.shape() ) } ;
}

// now we'll put the code to use

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

  zimt::apply < 2 > ( twice , as_view ( a ) ) ;

  // let's see a sample of the result

  std::cout << a [ { 100 , 100 } ] << std::endl ;
}
