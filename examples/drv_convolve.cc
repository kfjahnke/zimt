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

// convolve an image with a kernel. uses vigraimpex and a second TU
// 'convolve.cc' which provides the workhorse code in the form of
// a function project::payload. How project::payload does it's job
// depends on how it's compiled. It can either use a simple one-ISA
// compilation, where SIMD capabilities depend on compiler flags.
// This is the 'classical' zimt approach. This works with all zimt
// SIMD back-ends. Given SRC="drv_convolve.cc convolve.cc", typical
// single-ISA builds would use command lines like:
// g++ $SRC -O3 -I. -lvigraimpex
// g++ $SRC -O3 -mavx2 -I. -DUSE_HWY -lhwy -lvigraimpex
// Alternatively, the payload can use internal dispatching to the best
// SIMD ISA which the CPU executing the code has to offer. To affect
// this, you need to #define MULTI_SIMD_ISA and link with libhwy.
// You don't have to #define USE_HWY - the internal dispatch does
// also work with 'goading' code using zimt's own SIMD emulation.
// But internal dispatch doesn't (yet) work for the Vc or std::simd
// back-ends, they only work with the one-ISA scheme. Typical builds
// for the multi-SIMD-ISA build would be:
// g++ $SRC -O3 -I. -DMULTI_SIMD_ISA -lhwy -lvigraimpex
// g++ $SRC -O3 -mavx2 -I. -DUSE_HWY -DMULTI_SIMD_ISA -lhwy -lvigraimpex
// The convolution code itself doesn't use the back-ends extensively, so
// the run-time of the payload code doesn't vary greatly.

#include <iostream>
#include <stdlib.h>
#include <ctime>
#include <chrono>

#include <vigra/stdimage.hxx>
#include <vigra/imageinfo.hxx>
#include <vigra/impex.hxx>

// now we #include the zimt headers we need:

#include "../zimt/zimt_vigra.h"

// we silently assume we have a colour image

typedef zimt::xel_t < float , 3 > pixel_type; 

// target_type is a 2D array of pixels  

typedef zimt::array_t < 2 , pixel_type > target_type ;

namespace project {

int payload ( zimt::view_t < 2 , pixel_type > & img ,
              const std::vector < zimt::xlf_type > & kernel ) ;

}  // namespace project

// finally we code main, which in turn invokes project::payload.

int main ( int argc , char * argv[] )
{
  if ( argc < 2 )
  {
    std::cerr << "pass a colour image file as argument," << std::endl ;
    std::cerr << "followed by the filter's coefficients" << std::endl ;
    exit( -1 ) ;
  }

  // get the image file name
  
  vigra::ImageImportInfo imageInfo ( argv[1] ) ;

  char * end ;

  std::vector < zimt::xlf_type > kernel ;

  for ( int i = 2 ; i < argc ; i++ )
    kernel.push_back ( zimt::xlf_type ( strtold ( argv [ i ] , &end ) ) ) ;

  // set up a zimt::array_t with this shape

  zimt::array_t < 2 , pixel_type >
    a ( zimt::to_zimt ( imageInfo.shape() ) ) ;

  // import the image

  vigra::importImage ( imageInfo , zimt::to_vigra ( a ) ) ;

  // apply the filter, measure the time the payload code takes

  std::chrono::system_clock::time_point t_start
    = std::chrono::system_clock::now();

  project::payload ( a , kernel ) ;

  std::chrono::system_clock::time_point t_end
    = std::chrono::system_clock::now();

  std::cout << "payload function took "
            << std::chrono::duration_cast<std::chrono::milliseconds>
                 ( t_end - t_start ) . count()
       << " ms" << std::endl ;

  // store the result with vigra impex
  // this can take quite some time!

  vigra::ImageExportInfo eximageInfo ( "convolved.tif" );
  
  std::cout << "storing the target image as 'convolved.tif'" << std::endl ;
  
  vigra::exportImage ( zimt::to_vigra ( a ) ,
                       eximageInfo
                       .setPixelType("UINT8") ) ;
  exit ( 0 ) ;
}


