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

#include <zimt/zimt.h>

int main ( int argc , char * argv[] )
{
  zimt::bill_t bill ;

  // let's start with a simple 1D linspace.

  {
    typedef zimt::xel_t < float , 1 > delta_t ;
    delta_t start { .5 } ;
    delta_t step { .1 } ;
    zimt::linspace_t < float , 1 , 1 , 4 > l ( start , step , bill ) ;
    typedef zimt::pass_through < float , 1 , 4 > act_t ;
    zimt::array_t < 1 , delta_t > a ( 7 ) ;
    zimt::storer < float , 1 , 1 , 4 > p ( a , bill ) ;

    zimt::process ( a.shape , l , act_t() ,  p , bill ) ;

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
    zimt::linspace_t < float , 2 , 2 , 4 > l ( start , step , bill ) ;
    typedef zimt::pass_through < float , 2 , 4 > act_t ;
    zimt::array_t < 2 , delta_t > a ( { 7 , 5 } ) ;
    zimt::storer < float , 2 , 2 , 4 > p ( a , bill ) ;

    zimt::process ( a.shape , l , act_t() ,  p , bill ) ;

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
    zimt::linspace_t < float , 3 , 3 , 4 > l ( start , step , bill ) ;
    typedef zimt::pass_through < float , 3 , 4 > act_t ;
    zimt::array_t < 3 , delta_t > a ( { 2 , 3 , 4 } ) ;
    zimt::storer < float , 3 , 3 , 4 > p ( a , bill ) ;

    std::cout << "********** 3D:" << std::endl << std::endl ;

    zimt::process ( a.shape , l , act_t() ,  p , bill ) ;
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

    zimt::process ( a.shape , l , act_t() ,  p , bill ) ;
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
