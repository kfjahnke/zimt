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
// code.
// This example 'picks up' a patch in a small array and copies it to
// a region in a large array. Then it copies out this region to another
// small array and compares the two arrays. What's the point? The small
// array might be a patch which we want to apply to the large array,
// or, on the 'storing' side, we might only care to store part of the
// large array (cropping). With appropriate processing limits and
// get/put offsets, this can be done with zimt::process without the
// need for additional 'external' code. Of course this simple example
// might be coded differently: the easiest would be to form a subarray
// from the large array which covers only the patch area and use
// zimt::process without limits or offsets. The example is more to
// demonstrate correct use of the limits and offsets in the 'loading
// bill'.

#include <zimt/zimt.h>

// we receive the 'hot' axis, and the size of the patch (x, y)

void test ( std::size_t a , std::size_t x , std::size_t y )
{
  // as value type, we'll use a xel_t of one int - this is like
  // working with single integers, but since we're using zimt::process
  // rather than zimt::transform, the code expects xel_t values and
  // rejects 'naked' fundamentals.

  typedef zimt::xel_t < int , 1 > value_t ;

  // the shape of the patch arrays we'll be working on

  zimt::xel_t < std::size_t , 2 > patch_shape ( { x , y } ) ;

  // the shape of the larger array to and from which we'll copy
  // the patch

  zimt::xel_t < std::size_t , 2 > array_shape ( { 2 * x , 2 * y } ) ;

  // the 'act' functor merely passes it's input to it's output

  zimt::pass_through < int , 1 , 16 > act ;

  // we set up the arrays

  zimt::array_t < 2 , value_t > patch_in ( patch_shape ) ;
  zimt::array_t < 2 , value_t > large_array ( array_shape ) ;
  zimt::array_t < 2 , value_t > patch_out ( patch_shape ) ;

  // initialize patch_in with 'something recognizable'

  patch_in.set_data ( value_t ( 42 ) ) ;

  // the get_t object loads data from the source array. By using
  // a suitable 'bill', we'll make zimt::process load from patch_in
  // only, rather than trying to load the entire array.

  zimt::loader < int , 1 , 2 , 16 > get ( patch_in , a ) ;

  // the put_t object stores data to the target array. By using
  // a suitable 'bill', we'll make zimt::process store to patch_out
  // only, rather than storing the entire array.  The c'tor
  // still receives the 'notional' array.

  zimt::storer < int , 1 , 2 , 16 > put ( patch_out , a ) ;

  zimt::bill_t bill ;
  bill.axis = a ;

  // we set up the lower and upper limit for processing to cover
  // only a patch-sized window of the large array

  bill.lower_limit = { 23 , 17 } ;
  bill.upper_limit = { 23 + long(x) , 17 + long(y) } ;

  // and we also set up the offsets passed to the get and put
  // objects - here we pass precisely the negative of the window's
  // lower corner, which will be mapped to (0, 0) in the patch

  bill.get_offset = { -23 , -17 } ;
  bill.put_offset = { -23 , -17 } ;

  // showtime!

  zimt::process ( array_shape , get , act , put , bill ) ;

  // let's look at the result. we expect patch_out to contain the
  // same data as patch_in.

  auto it = zimt::mcs_t < 2 > ( patch_shape ) ;
  for ( int i = 0 ; i < patch_in.size() ; i++ )
  {
    auto crd = it() ;
    assert ( patch_in [ crd ] == patch_out [ crd ] ) ;
  }
}

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
