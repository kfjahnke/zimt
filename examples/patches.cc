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
// This example uses a 2D 'permute' object as the get_t passed to
// zimt::process. This is similar to the input generation in
// permute.cc, only that here we work in 2D and with larger arrays.
// The point of this example is to demonstrate the upper and lower
// limit to processing which can be specified in the 'loading bill'
// passed to zimt::process. These two values specify a window to
// which processing is limited. In this example, we conceptually
// cut up the array into 507X517 patches (or smaller, where a patch
// would exceed the array's extent) and process the patches by
// invocations of zimt::process with corresponding limits in the
// 'loading bill'. After all patches are processed, we expect the
// same result as if we'd processed the whole array at once. The
// test is repeated with different array sizes, and the processing
// axis (or 'hot' axis) is also varied.

#include <array>
#include "../zimt.h"

void test ( std::size_t a , std::size_t x , std::size_t y )
{
  // p_t will be the type of get_t for this example: a zimt::permute
  // object permuting two 1D arrays with int values.

  typedef zimt::permute < int , 2 , 2 , 16 > p_t ;

  // this permute object produces values which are passed to the 'act'
  // functor. In this example, these values are xel_t of two vectors
  // of int.

  typedef typename p_t::value_t value_t ;

  // the 1D arrays of per-axis values are held in zimt::arrays

  typedef zimt::array_t < 1 , int > axis_t ;

  // and the permute object needs two of them in a std::array

  typedef std::array < zimt::view_t < 1 , int > , 2 > grid_t ;

  // TODO: write an overload for 1D arrays accepting a 'naked' size

  typedef zimt::xel_t < std::size_t , 1 > extent_t ;

  // set up the per-axis values

  axis_t ax ( x ) ;
  axis_t ay ( y ) ;

  for ( std::size_t i = 0 ; i < x ; i++ )
    ax [ i ] = i + 1 ;

  for ( std::size_t i = 0 ; i < y ; i++ )
    ay [ i ] = i + 1 ;

  // form the 'grid_t' object containing views to the two 1D arrays

  grid_t grid { ax , ay } ;

  // create the permute object

  p_t get_xy ( grid , a ) ;

  // the shape of the array we'll be working on

  zimt::xel_t < std::size_t , 2 > shape ( { x , y } ) ;

  // the 'act' functor merely passes it's input to it's output

  zimt::pass_through < int , 2 , 16 > act ;

  // a target array, receiving the output of zimt::process

  zimt::array_t < 2 , value_t > trg ( shape ) ;

  // the put_t object stores values to the target array

  zimt::storer < int , 2 , 2 , 16 > p ( trg , a ) ;

  // the loading bill is set up with the 'hot' axis

  zimt::bill_t bill ;
  bill.axis = a ;

  // so far this is pretty much the same as permute.cc, but in 2D.
  // But we'll do the processing in patches: we'll run an outer
  // loop over the patches, modify the bill to set up limits for
  // the window to which the processing is applied, and then call
  // zimt::process repeatedly to process all windows.

  int patch_no = 1 ; // just for the printed output

  // we'll use 507X507 here for the patch size; in a 'real'
  // program we'd rather use a power of two, like 512X512.
  // But the 'odd' shape is better to 'exercise' the entire
  // processing code.

  for ( std::size_t u = 0 ; u < x ; u += 507 )
  {
    std::size_t ue = std::min ( u + 507 , x ) ;
    for ( std::size_t v = 0 ; v < y ; v += 517 )
    {
      std::size_t ve = std::min ( v + 517 , y ) ;

      // set up the limits in 'bill'

      bill.lower_limit = { long(u) , long(v) } ;
      bill.upper_limit = { long(ue) , long(ve) } ;

      // print out the limits

      std::cout << "patch #" << patch_no
                << " lower_limit: " << u << ", " << v
                << " upper_limit: " << ue << ", " << ve
                << std::endl ;

      // showtime!

      zimt::process ( shape , get_xy , act , p , bill ) ;

      // next patch

      ++patch_no ;
    }
  }

  // let's look at the result. If the patches were overlapping,
  // the test would still succeed - we're more interested in
  // whether the entire array has indeed been filled correctly.

  auto it = zimt::mcs_t < 2 > ( shape ) ;
  for ( int i = 0 ; i < trg.size() ; i++ )
  {
    auto crd = it() ;
    assert ( trg [ crd ] == ( crd + 1 ) ) ;
  }
}

// now we'll run the test with a variety of parameters.

int main ( int argc , char * argv[] )
{
  for ( std::size_t x = 405 ; x < 2000 ; x += 387 )
  {
    for ( std::size_t y = 355 ; y < 1500 ; y += 359 )
    {
      for ( std::size_t a = 0 ; a < 2 ; a++ )
      {
        std::cout << "test: hot axis: " << a
                  << " x: " << x
                  << " y: " << y << std::endl ;
        test ( a , x , y ) ;
      }
    }
  }
}
