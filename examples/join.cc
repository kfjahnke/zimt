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
// zimt has a ready-made get_t variant for the purpose, class join_t.
// It's initialized with N arrays holding fundamentals, but it
// produces simdized data with N channels and feeds them into the
// 'act' functor.
// This provides a handy way to process multi-channel data provided
// as several single-channel arrays - it's not necessary to convert
// the data to interleaved format for processing. The reverse
// process - now at the 'receiving end' is also available ready-made
// with class zimt::split_t. That class takes simdized data with
// N channels from the 'act' functor and stores them to N arrays of
// fundamentals. In sum, using these classes - or zimt::loader and
// zimt::storer, zimt can handle interleaved and channel-separated
// data, and the 'act' functor receives simdized xel data for
// processing in all cases, which is most efficient.

#include <iomanip>
#include <array>
#include <zimt/zimt.h>

int main ( int argc , char * argv[] )
{
  std::cout << std::fixed << std::showpoint
            << std::setprecision(1) ;

  // We set up the type of join_t we'll use. It's fundamental
  // data type is float, we will produce 4-channel data in a
  // 2D 'notional' shape, and the lane count for the processing
  // pipeline is set to 16.

  typedef zimt::join_t < float , 4 , 2 , 16 > j_t ;
  typedef typename j_t::value_t value_t ;
  typedef zimt::xel_t < float , 4 > px_t ;

  // just to show that the join_t's value_t is what we expect

  static_assert ( std::is_same < value_t , px_t > :: value ) ;

  // we set up four 2D arrays holding float data

  zimt::array_t < 2 , float > r ( { 3 , 4 } ) ;
  zimt::array_t < 2 , float > g ( { 3 , 4 } ) ;
  zimt::array_t < 2 , float > b ( { 3 , 4 } ) ;
  zimt::array_t < 2 , float > a ( { 3 , 4 } ) ;

  // just to 'see something', we set a few exemplary input values

  r [ { 1 , 3 } ] = 1.0f ;
  g [ { 1 , 3 } ] = 2.0f ;
  b [ { 1 , 3 } ] = 3.0f ;
  a [ { 1 , 3 } ] = 4.0f ;

  // 'loading bill' for the zimt::process run - we take the
  // default.

  zimt::bill_t bill ;

  // class join_t expects a std::array of N view_t which is easily
  // set up with an initializer expression.

  std::array < zimt::view_t < 2 , float > , 4 >
    src ( { r , g , b , a } ) ;

  // now we can construct the join_t object

  j_t get_rgba ( src , bill ) ;

  // we'll have the result of processing stored in an array.
  // note how this array's data type is now px_t, while we feed
  // plain float

  zimt::array_t < 2 , px_t > rgba ( { 3 , 4 } ) ;

  // we won't actually do anything during processing, hence
  // we pick a zimt::pass_through object as 'act' functor

  typedef zimt::pass_through < float , 4 , 16 > act_t ;

  // on the receiving end of the processing chain, we use a
  // zimt::storer object which is set up to store data in
  // the target array 'rgba'

  zimt::storer < float , 4 , 2 , 16 > p ( rgba , bill ) ;

  // now we have all the components set up and call zimt::process

  zimt::process ( rgba.shape , get_rgba , act_t() , p , bill ) ;

  // and finally convince ourselves that the explicitly set
  // input values have made it to the output as the corresponding
  // px_t

  assert ( ( rgba [ { 1 , 3 } ] == px_t ( { 1 , 2 , 3 , 4 } ) ) ) ;

  std::cout << "success! rgba [ { 1 , 3 } ] = "
            << rgba [ { 1 , 3 } ] << std::endl ;
}
