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

/// \file prefilter.cc
///
/// \brief  apply a b-spline prefilter filter to an image
///
/// This is a port of the vspline example program prefilter.cc
///
/// zimt has some code which isn't useful only for b-splines.
/// One example is the application of a digital filter to an array.
/// With zimt's multithreaded SIMD code, this is done efficiently.
/// This example program will apply an n-pole forward/backward
/// recursive filter using a the prefilter poles of a b-spline
/// of a given degree. This is the operation which is needed to
/// convert a b-spline's knot point values to b-spline coefficients.
///
/// compile with:
/// ./examples.sh prefilter.cc
///
/// invoke passing an image file and the spline degree. the result
/// will be written to 'prefilter.tif'. Note that vigraimpex will
/// compress the resulting values into the output format's dynamic
/// range, so if you look at the result, it will - with rising spline
/// degree - approach some shade of middle grey, because the few very
/// large result values (where high frequencies appear in the input)
/// will determine the range, while most result values are nearer the
/// incoming knot point values.


#include <iostream>
#include <stdlib.h>
#include <chrono>

#include <zimt/zimt_vigra.h>
#include <zimt/prefilter.h>

#include <vigra/stdimage.hxx>
#include <vigra/imageinfo.hxx>
#include <vigra/impex.hxx>

// we silently assume we have a colour image

typedef zimt::xel_t < float , 3 > pixel_type; 

// target_type is a 2D array of pixels  

typedef zimt::array_t < 2 , pixel_type > target_type ;

int main ( int argc , char * argv[] )
{
  if ( argc < 3 )
  {
    std::cerr << "pass a colour image file as argument," << std::endl ;
    std::cerr << "followed by the spline degree" << std::endl ;
    exit( -1 ) ;
  }

  // get the image file name
  
  vigra::ImageImportInfo imageInfo ( argv[1] ) ;

  int degree = std::atoi ( argv [ 2 ] ) ;
  std::cout << "degree: " << degree << std::endl ;

  // set up a zimt::array_t with this shape

  zimt::array_t < 2 , pixel_type >
    a ( zimt::to_zimt ( imageInfo.shape() ) ) ;

  // import the image

  vigra::importImage ( imageInfo , zimt::to_vigra ( a ) ) ;

  // we want to time the operation

  std::chrono::system_clock::time_point start
    = std::chrono::system_clock::now() ;

  // apply the filter

  zimt::prefilter
   ( a ,
     a ,
     { zimt::MIRROR , zimt::MIRROR } ,
     degree ) ;

  // store the result with vigra impex

  std::chrono::system_clock::time_point end
    = std::chrono::system_clock::now() ;

  std::cout << "prefilter took "
            << std::chrono::duration_cast<std::chrono::milliseconds>
                ( end - start ) . count()
            << " ms" << std::endl ;

  vigra::ImageExportInfo eximageInfo ( "prefilter.tif" );
  
  std::cout << "storing the target image as 'prefilter.tif'" << std::endl ;
  
  vigra::exportImage ( zimt::to_vigra ( a ) ,
                       eximageInfo
                       .setPixelType("UINT8") ) ;

  // just to make sure the filter code works with fundamentals as well

  zimt::array_t < 2 , float > b ( { 12 , 13 } ) ;
  zimt::prefilter
   ( b ,
     b ,
     { zimt::MIRROR , zimt::MIRROR } ,
     degree ) ;

  exit ( 0 ) ;
}
