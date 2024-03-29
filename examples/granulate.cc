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
// from consecutive values in a 'small' dimension 0 of a float array
// with one dimension more than the 'shape notion' of the 'process'
// invocation. For simplicity's sake, we require a source array with
// no strides in dimension zero (the one where the channels of the
// xels are lined up) and dimension one. So with a pattern of ABC
// for the xel's channels, memory holds ABCABCABC..., and higher
// dimensions (two and up) may have arbitrary strides. The 'notion'
// of the xels in the view which is processed is {ABC}{ABC}{ABC}...
// What do we gain? We can feed 'process' with an appropriately
// shaped and strided array of T, but use an 'act' functor which
// processes xel_t of T instead. If we have a variety of different
// input sources, we can nevertheless use the same act functor,
// which may be a complex pixel pipeline which we don't want to
// produce in different versions just to cater for the variety of
// data sources we need to process. In this context, look at the
// example 'join.cc' as well, which demonstrates yet another input
// 'scenario' which can be routed to an act functor processing xel_t.

#include <iomanip>
#include <array>
#include <zimt/zimt.h>

// join_t is the 'get_t' we'll use for this example. This functor
// generates the input values to the functor. In this example we'll
// demonstrate how to use a get_t to use an adapted view, presenting
// an array of T with a first 'short' dimension for the N channels
// as an array of xel_t. The data don't have to be modified, we can
// just create an 'alternative view'.
// TODO: this class could be fleshed out (remove the constraints)
// and put into get_t.h

template < typename T ,     // fundamental type
           std::size_t N ,  // channel count
           std::size_t D ,  // dimensions of the 'shape notion'
           std::size_t L >  // lane count
struct join_t
: public zimt::unstrided_loader < T , N , D , L >
{
  typedef zimt::xel_t < T , N > value_t ;

  // data source: D+1-dimensional view holding the fundamental type

  typedef zimt::view_t < D + 1 , T > src_t ;

  // adapted data source: D-dimensional view of value_t

  typedef zimt::view_t < D , value_t > in_t ;

  // the c'tor of our base class (zimt::unstrided_loader) must be
  // invoked with the array conforming with the desired 'notion':
  // one dimension less, and value_t instead of T. We use a static
  // member function:

  static in_t convert ( const src_t & src )
  {
    // for this example, we limit the scope of input: the 'extra'
    // dimension must be the first one, and it mustn't be strided.

    assert ( N == src.shape[0] ) ;
    assert ( src.strides[0] == 1 ) ;

    // we also require that the second dimension is unstrided

    assert ( src.strides[1] == N ) ;

    // we calculate adapted shape and strides for the desired
    // new view

    zimt::xel_t < std::size_t , D > _shape ;
    zimt::xel_t < long , D > _strides ;
    for ( std::size_t i = 0 ; i < D ; i++ )
    {
      _shape [ i ] = src.shape [ i + 1 ] ;
      _strides [ i ] = src.strides [ i + 1 ] / N ;
    }

    // and create and return it

    return in_t ( (value_t*) src.origin , _strides , _shape ) ;
  }

  // with the static member function convert, all we need to do is
  // pass the result of 'convert' to the base class c'tor.

  join_t ( const src_t & src , const zimt::bill_t & bill )
  : zimt::unstrided_loader < T , N , D , L > ( convert ( src ) , bill )
  { }
} ;

int main ( int argc , char * argv[] )
{
  std::cout << std::fixed << std::showpoint
            << std::setprecision(1) ;

  typedef join_t < float , 4 , 2 , 16 > j_t ;
  typedef typename j_t::value_t value_t ;

  zimt::bill_t bill ;

  // this is our input: an array of float

  zimt::array_t < 3 , float > src ( { 4 , 1024 , 1024 } ) ;

  j_t get_rgba ( src , bill ) ;

  // this is our output: an array of xel_t<float,4>

  zimt::array_t < 2 , value_t > rgba ( { 1024 , 1024 } ) ;

  // we won't 'act' on the data and on the 'output side' all we do is
  // store them in the target array

  typedef zimt::pass_through < float , 4 , 16 > act_t ;
  zimt::storer < float , 4 , 2 , 16 > p ( rgba , bill ) ;

  // just one sample

  src [ { 0 , 101 , 203 } ] = 1.0f ;
  src [ { 1 , 101 , 203 } ] = 2.0f ;
  src [ { 2 , 101 , 203 } ] = 3.0f ;
  src [ { 3 , 101 , 203 } ] = 4.0f ;

  // let's go

  zimt::process ( rgba.shape , get_rgba , act_t() , p , bill ) ;

  // check that the sample made it to the output

  std::cout << rgba [ { 100 , 203 } ] << std::endl ;
  std::cout << rgba [ { 101 , 203 } ] << std::endl ;
  std::cout << rgba [ { 102 , 203 } ] << std::endl ;
}
