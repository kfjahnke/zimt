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

    this example program demonstrates how we can employ vigraimpex
    to load image files into zimt arrays and to store zimt arrays to
    image files using vigraimpex.

*/

#include <zimt/zimt_vigra.h>
#include <zimt/zimt.h>
#include <vigra/imageinfo.hxx>
#include <vigra/impex.hxx>

// type for colour pixels: xel_t of three unsigned char

typedef zimt::xel_t < unsigned char , 3 > px_t ;

// simple zimt pixel functor, rotating the colour channels

struct rotate_rgb_t
: public zimt::unary_functor < px_t >
{
  template < typename I , typename O >
  void eval ( const I & in , O & out ) const
  {
    out [ 0 ] = in [ 1 ] ;
    out [ 1 ] = in [ 2 ] ;
    out [ 2 ] = in [ 0 ] ;
  }
} ;

int main ( int argc , char * argv[] )
{
  if ( argc < 2 )
  {
    std::cerr << "pass a coulour image on the command line"
              << std::endl ;
    exit ( 1 ) ;
  }

  // find the image's shape with an imageInfo

  vigra::ImageImportInfo imageInfo ( argv[1] ) ;

  // set up a zimt::array_t with this shape

  zimt::array_t < 2 , px_t >
    a ( zimt::to_zimt ( imageInfo.shape() ) ) ;

  // import the image

  vigra::importImage ( imageInfo , zimt::to_vigra ( a ) ) ;

  // run the channel rotating functor over the array

  zimt::apply ( rotate_rgb_t() , a ) ;

  // create an exportInfo

  vigra::ImageExportInfo eximageInfo ( "rotated.jpg" ) ;

  // and export the array

  vigra::exportImage ( zimt::to_vigra ( a ) , eximageInfo ) ;

  std::cout << "result was stored to 'rotated.jpg'" << std::endl ;
}
