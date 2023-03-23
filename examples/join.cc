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

// Another example demonstrating the direct use of zimt's 'wielding'
// code, which allows what I call 'regular' operations, an abstraction
// of array processing which uses the structure of array processing
// but not necessarily arrays as it's substrate: The processing is
// guided by an n-dimensional coordinate block which is traversed
// along parallel 1D 'lines' or 'strands' which are all parallel to
// one axis of the construct.
// This example uses a get_t functor to put together RGBA pixels
// from four separate float arrays, as you might get them 'natively'
// from some image file formats which don't store the data interleaved.
// The pixels are stored to an interleaved array.

#include <iomanip>
#include <array>
#include "../zimt.h"

// join_t is the 'get_t' we'll use for this example. This functor
// generates the input values to the functor. In this example we'll
// demonstrate how to use a get_t to put together RGBA pixels from
// four separate float arrays holding the channels. We want the
// get_t to do the job so that the act functor can work on rgba
// pixels as input just as if the rgba pixels had been loaded from
// an array of rgba pixels, not four separate channel arrays.

template < typename T ,     // fundamental type
           std::size_t N ,  // channel count
           std::size_t D ,  // dimensions
           std::size_t L >  // lane count
struct join_t
{
  typedef zimt::xel_t < T , N > value_t ;
  typedef zimt::simdized_type < value_t , L > value_v ;
  typedef typename value_v::value_type value_ele_v ;
  typedef zimt::xel_t < long , D > crd_t ;

  // const std::size_t d ; // processing axis - 0 or 1 for a 2D array

  // source of data: N D-dimensional views to the fundamental type

  typedef std::array < zimt::view_t < D , T > , N > src_t ;
  src_t src ;

  zimt::xel_t < const T * , N > pickup ; // source pointers
  zimt::xel_t < long , N > stride ;      // strides of source arrays

  join_t ( const src_t & _src )
  : src ( _src )
  {
    // copy out the strides of the source arrays

    for ( int ch = 0 ; ch < N ; ch++ )
    {
      stride [ ch ] = src [ ch ] . strides [ 0 ] ;
    }
  }

  // init is used to initialize the vectorized value to the value
  // it should hold at the beginning of the peeling run. The discrete
  // coordinate 'crd' gives the location of the first value, and this
  // function infers the start value from it. Here we set the pickup
  // pointers and perform a 'regular gather' to get the first batch
  // of values. Note how we could specialize the code to use load
  // instructions instead of rgather if the stride is 1.

  void init ( value_v & v , const crd_t & crd )
  {
    for ( int ch = 0 ; ch < N ; ch++ )
    {
      pickup [ ch ] = & ( src [ ch ] [ crd ] ) ;
      v [ ch ] . rgather ( pickup [ ch ] , stride [ ch ] ) ;
    }
  }

  // 'capped' variant. This is only needed if the current segment is
  // so short that no vectors can be formed at all. We fill up the
  // target value with the last valid datum.

  void init ( value_v & trg ,
              const crd_t & crd ,
              std::size_t cap )
  {
    for ( int ch = 0 ; ch < N ; ch++ )
    {
      pickup [ ch ] = & ( src [ ch ] [ crd ] ) ;
      for ( std::size_t e = 0 ; e < cap ; e++ )
        trg [ ch ] [ e ] = pickup [ ch ] [ e * stride [ ch ] ] ;
      trg.stuff ( cap ) ;
    }
  }


  // increase modifies it's argument to contain the next value, or
  // next vectorized value, respectively - first we increase the
  // pickup pointers, then we get the data from that location.

  void increase ( value_v & trg )
  {
    for ( int ch = 0 ; ch < N ; ch++ )
    {
      pickup [ ch ] += L * stride [ ch ] ;
      trg [ ch ] . rgather ( pickup [ ch ] , stride [ ch ] ) ;
    }
  }

  // 'capped' variant. This is called after all vectors in the current
  // segment have been processed, so the lanes in trg beyond the cap
  // should hold valid data, and 'stuffing' them with the last datum
  // before the cap is optional.

  void increase ( value_v & trg ,
                  std::size_t cap ,
                  bool _stuff = true )
  {
    for ( int ch = 0 ; ch < N ; ch++ )
    {
      pickup [ ch ] += L * stride [ ch ] ;
      for ( std::size_t e = 0 ; e < cap ; e++ )
        trg [ ch ] [ e ] = pickup [ ch ] [ e * stride [ ch ] ] ;
      if ( _stuff )
        trg.stuff ( cap ) ;
    }
  }
} ;

int main ( int argc , char * argv[] )
{
  std::cout << std::fixed << std::showpoint
            << std::setprecision(1) ;

  typedef join_t < float , 4 , 2 , 16 > j_t ;
  typedef typename j_t::value_t value_t ;

  zimt::array_t < 2 , float > r ( { 1024 , 1024 } ) ;
  zimt::array_t < 2 , float > g ( { 1024 , 1024 } ) ;
  zimt::array_t < 2 , float > b ( { 1024 , 1024 } ) ;
  zimt::array_t < 2 , float > a ( { 1024 , 1024 } ) ;

  std::array < zimt::view_t < 2 , float > , 4 >
    src ( { r , g , b , a } ) ;

  j_t get_rgba ( src ) ;

  zimt::array_t < 2 , value_t > rgba ( { 1024 , 1024 } ) ;

  typedef zimt::pass_through < float , 4 , 16 > act_t ;
  zimt::norm_put_t < act_t , 2 > p ( rgba , 0 ) ;

  r [ { 101 , 203 } ] = 1.0f ;
  g [ { 101 , 203 } ] = 2.0f ;
  b [ { 101 , 203 } ] = 3.0f ;
  a [ { 101 , 203 } ] = 4.0f ;

  zimt::process < act_t , 2 > ( act_t() , rgba , get_rgba , p ) ;

  std::cout << rgba [ { 100 , 203 } ] << std::endl ;
  std::cout << rgba [ { 101 , 203 } ] << std::endl ;
  std::cout << rgba [ { 102 , 203 } ] << std::endl ;
}
