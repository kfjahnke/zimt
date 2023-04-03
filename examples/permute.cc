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
// This example uses a 'permute' object as the get_t passed to
// zimt::process.

#include <iomanip>
#include <array>
#include "../zimt.h"

// currently, part of this example can't be used with Vc and highway,
// because class zimt::array_t does not work with allocators for the
// vectorized data, so vectorized storage using vstorer and vloader
// fails.

#if defined USE_VC or defined USE_HWY
#define EXCLUDE
#endif

void test ( std::size_t a , std::size_t x , std::size_t y , std::size_t z )
{
  // p_t will be the type of get_t for this example: a zimt::permute
  // object permuting three 1D arrays with int values.

  typedef zimt::permute < int , 3 , 16 > p_t ;

  // this permute object produces values which are passed to the 'act'
  // functor. In this example, these values are xel_t of three vectors
  // of int.

  typedef typename p_t::value_t value_t ;
  typedef typename p_t::value_v value_v ;

  // the 1D arrays of per-axis values are held in zimt::arrays

  typedef zimt::array_t < 1 , int > axis_t ;

  // and the permute object needs thre of them in a std::array

  typedef std::array < zimt::view_t < 1 , int > , 3 > grid_t ;

  // TODO: write an overload for 1D arrays accepting a 'naked' size

  typedef zimt::xel_t < std::size_t , 1 > extent_t ;

  // set up the per-axis values

  axis_t ax ( x ) ;
  axis_t ay ( y ) ;
  axis_t az ( z ) ;

  for ( std::size_t i = 0 ; i < x ; i++ )
    ax [ i ] = i + 1 ;

  for ( std::size_t i = 0 ; i < y ; i++ )
    ay [ i ] = i + 1 ;

  for ( std::size_t i = 0 ; i < z ; i++ )
    az [ i ] = i + 1 ;

  // form the 'grid_t' object containing views to the three 1D arrays

  grid_t grid { ax , ay , az } ;

  // create the permute object

  p_t get_abc ( grid , a ) ;

  // and a target array, receiving the output of zimt::process

  zimt::array_t < 3 , value_t > trg ( { x , y , z } ) ;
  zimt::array_t < 3 , value_t > trg2 ( { x , y , z } ) ;

  // the 'act' functor merely passes it's input to it's output

  typedef zimt::pass_through < int , 3 , 16 > act_t ;

  // and the put_t object stores it to the target array

  zimt::norm_put_t < act_t , 3 > p ( trg , a ) ;

  zimt::bill_t bill ;
  bill.axis = a ;

  // showtime!

  zimt::process < act_t , 3 > ( act_t() , trg , get_abc , p , bill ) ;

  // a little flourish (TODO: factor out): repeat the operation,
  // but store to a vbuffer object, then retrieve from same.

  // get the vbuffer array

  auto vbuffer = zimt::get_vector_buffer ( trg , a , 16 ) ;

  // set up and use a vstorer object as put_t for zimt::process

  zimt::vstorer < int , 3 , 3 , 16 > vs ( vbuffer , a ) ;
  zimt::process < act_t , 3 > ( act_t() , trg , get_abc , vs , bill ) ;

  // the data are now in the vbuffer array. now set up and use
  // a vloader object as get_t for zimt::process - as put_t we
  // use a norm_put_t storing to 'trg2'

  zimt::vloader < int , 3 , 3 , 16 > vl ( vbuffer , a ) ;
  zimt::norm_put_t < act_t , 3 > pp ( trg2 , a ) ;
  zimt::process < act_t , 3 > ( act_t() , trg2 , vl , pp , bill ) ;

  // let's look at the result

  auto it = zimt::mcs_t < 3 > ( trg.shape ) ;
  for ( int i = 0 ; i < trg.size() ; i++ )
  {
    auto crd = it() ;
    assert ( trg [ crd ] == ( crd + 1 ) ) ;
    assert ( trg2 [ crd ] == ( crd + 1 ) ) ;
  }
}

int main ( int argc , char * argv[] )
{
  for ( std::size_t x = 1 ; x < 35 ; x++ )
  {
    for ( std::size_t y = 1 ; y < 35 ; y++ )
    {
      for ( std::size_t z = 1 ; z < 35 ; z++ )
      {
        for ( std::size_t a = 0 ; a < 3 ; a++ )
        {
          test ( a , x , y , z ) ;
        }
      }
    }
  }
}
