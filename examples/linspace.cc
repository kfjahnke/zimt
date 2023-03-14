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

// example code producing arrays holding linear gradients, just like
// what you get from NumPy's 'linspace'. This example directly uses
// the code in minz.h which makes it quite verbose, because this
// code needs more arguments than zimt::transform and relatives.

#include "../zimt.h"

// linspace_t is the 'get_t' we'll use for this example. This functor
// generates the input values to the functor - and for this example,
// where the 'act' functor does nothing but route it's input to it's
// output, it actually does all the 'work'.
// This example may seem trivial, but it should hint at the flexibility
// of zimt::process and how to use it. And linspace_t can serve as
// a template for more elaborate 'get_t' classes.

template < typename T ,     // elementary type
           std::size_t N ,  // number of channels
           std::size_t L >  // lane count
struct linspace_t
{
  typedef zimt::xel_t < T , N > value_t ;
  typedef zimt::simdized_type < value_t , L > value_v ;
  typedef typename value_v::value_type value_ele_v ;
  typedef zimt::xel_t < long , N > crd_t ;
  typedef zimt::simdized_type < crd_t , L > crd_v ;
  typedef typename crd_v::value_type crd_ele_v ;

  const std::size_t d ;
  const value_t start ;
  const value_t step ;

  // linspace_t's c'tor receives start, step and axis. Note how
  // start and step are N-dimensional; each component gives the
  // intended value for the corresponding axis.

  linspace_t ( const value_t & _start ,
               const value_t & _step ,
               const std::size_t & _d )
  : start ( _start ) ,
    step ( _step ) ,
    d ( _d )
  { }

  // init is used to initialize the vectorized value to the value
  // it should hold at the beginning of the peeling run. The discrete
  // coordinate 'crd' gives the location of the first value, and this
  // function infers the start value from it. The scalar value will
  // not be used until peeling is done, so it isn't initialized here.

  void init ( value_v & cv , const crd_t & crd )
  {
    cv = step * crd + start ;
    cv [ d ] += value_ele_v::iota() * step [ d ] ;
  }

  // initialize the scalar value from the discrete coordinate.
  // This needs to be done once after peeling, the scalar value
  // is not initialized before.

  void init ( value_t & c , const crd_t & crd )
  {
    c = step * crd + start ;
  }

  // increase modifies it's argument to contain the next value, or
  // next vectorized value, respectively

  void increase ( value_t & trg )
  {
    trg [ d ] += step [ d ] ;
  }

  void increase ( value_v & trg )
  {
    trg [ d ] += ( step [ d ] * L ) ;
  }
} ;

int main ( int argc , char * argv[] )
{
  // let's start with a simple 1D linspace.

  {
    typedef zimt::xel_t < float , 1 > delta_t ;
    delta_t start { .5 } ;
    delta_t step { .1 } ;
    linspace_t < float , 1 , 4 > l ( start , step , 0 ) ;
    typedef zimt::pass_through < float , 1 , 4 > act_t ;
    zimt::array_t < 1 , delta_t > a ( 7 ) ;
    zimt::norm_put_t < act_t , 1 > p ( a , 0 ) ;

    zimt::process ( act_t() , a , l , p ) ;

    std::cout << "********** 1D:" << std::endl << std::endl ;

    for ( std::size_t i = 0 ; i < 7 ; i++ )
      std::cout << " " << a [ i ] ;
    std::cout << std::endl << std::endl ;
  }

  // now we'll go 2D

  {
    typedef zimt::xel_t < float , 2 > delta_t ;
    delta_t start { .5 , 0.7 } ;
    delta_t step { .1 , .2 } ;
    linspace_t < float , 2 , 4 > l ( start , step , 0 ) ;
    typedef zimt::pass_through < float , 2 , 4 > act_t ;
    zimt::array_t < 2 , delta_t > a ( { 7 , 5 } ) ;
    zimt::norm_put_t < act_t , 2 > p ( a , 0 ) ;

    zimt::process ( act_t() , a , l , p ) ;

    std::cout << "********** 2D:" << std::endl << std::endl ;

    for ( std::size_t y = 0 ; y < 5 ; y++ )
    {
      for ( std::size_t x = 0 ; x < 7 ; x++ )
        std::cout << " " << a [ { x , y } ] ;
      std::cout << std::endl ;
    }
    std::cout << std::endl ;
  }

  // just to show off, 3D

  {
    typedef zimt::xel_t < float , 3 > delta_t ;
    delta_t start { .5 , 0.7 , -.9 } ;
    delta_t step { .1 , .2 , -.4 } ;
    linspace_t < float , 3 , 4 > l ( start , step , 0 ) ;
    typedef zimt::pass_through < float , 3 , 4 > act_t ;
    zimt::array_t < 3 , delta_t > a ( { 2 , 3 , 4 } ) ;
    zimt::norm_put_t < act_t , 3 > p ( a , 0 ) ;

    std::cout << "********** 3D:" << std::endl << std::endl ;

    zimt::process ( act_t() , a , l , p ) ;
    for ( std::size_t z = 0 ; z < 4; z++ )
    {
      for ( std::size_t y = 0 ; y < 3 ; y++ )
      {
        for ( std::size_t x = 0 ; x < 2 ; x++ )
          std::cout << " " << a [ { x , y , z } ] ;
        std::cout << std::endl ;
      }
      std::cout << std::endl ;
    }
  }
}
