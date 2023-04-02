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

int main ( int argc , char * argv[] )
{
  std::cout << std::fixed << std::showpoint
            << std::setprecision(1) ;

  typedef zimt::join_t < float , 4 , 2 , 16 > j_t ;
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
