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

// This example uses a 'linspace_t' get_t to provide 2D coordinates of 
// the sample points of a rectilinear image. The act functor is set up
// to add a third constant coordinate (the distance to the origin),
// yielding a 3D coordinate representing a directional vector in space.
// The directional vector is used to do an 'environment lookup' from
// an environment map.
// This is a rough-and-ready first draft and the environment call goes
// via the filename, which is less efficient. Also, because the code
// directly writes scanlines to the output, it is single-threaded,
// slowing it down even more. But it works. Pass a 2:1 environment
// map (a 'full spherical', 360 degree panorama), decide on the size
// of the output and the output's field of view, and the program will
// produce a rectilinear view and write it toa.tif.

#include <iomanip>
#include <array>
#include <zimt/zimt.h>
#include <zimt/scanlines.h>
#include <OpenImageIO/imageio.h>
#include <OpenImageIO/typedesc.h>
#include <OpenImageIO/texture.h>

using namespace OIIO ;

typedef zimt::xel_t < float , 2 > v2_t ;
typedef zimt::xel_t < float , 3 > v3_t ;

struct echo
: public zimt::unary_functor < v2_t , v2_t , 16 >
{
  template < typename I , typename O >
  void eval ( const I & i , O & o ) const
  {
    std::cout << i << std::endl ;
  }
} ;

struct lookup_t
: public zimt::unary_functor < v2_t , v3_t , 16 >
{
  TextureSystem * ts ;
  TextureOptBatch & batch_options ;
  ustring ufilename ;
  v2_t step ;

  lookup_t ( TextureSystem * _ts ,
             TextureOptBatch & _batch_options ,
             ustring _ufilename ,
             v2_t _step )
  : ts ( _ts ) ,
    batch_options ( _batch_options ) ,
    ufilename ( _ufilename ) ,
    step ( _step )
  { }

  template < typename I , typename O >
  void eval ( const I & _crd , O & px ) const
  {
    out_v crd , delta ;
    crd[0] = _crd[0] ;
    crd[1] = _crd[1] ;
    crd[2] = 1.0f ;
    // auto norm = crd[0] * crd[0] + crd[1] * crd[1] + 1.0f ;
    // crd /= norm ;
    const float * R = (const float*) crd.data() ;
    delta[0] = step[0] ;
    delta[1] = step[0] ;
    delta[2] = 0 ;
    const float * pd = (const float*) delta.data() ;
    float * res = (float*) px.data() ;
    bool success = ts->environment ( ufilename , batch_options ,
                                     Tex::RunMaskOn ,
                                      R , pd , pd , 3 , res ) ;
  }
} ;

// further down, we'll construct two st_line_store_t objects.
// st_line_store_t's c'tor expects a tile loading and a tile
// storing function, but since we're only loading with one and
// storing with the other, we have to pass something for the
// other function: this one, pass. It does nothing.

bool pass ( const float * p_trg ,
            std::size_t nbytes ,
            std::size_t line )
{
  return true ;
}

auto typedesc = TypeDesc::FLOAT ;

int main ( int argc , char * argv[] )
{
  assert ( argc > 4 ) ;
  std::string filename ( argv[1] ) ;
  int w = std::stoi ( argv[2] ) ;
  int h = std::stoi ( argv[3] ) ;
  float v = std::stod ( argv[4] ) * M_PI / 180.0 ;

  std::cout << "filename: " << filename << " w: " << w
            << " h: " << h << " v: " << v << std::endl ;

  float d = 2.0 * tan ( v / 2.0 ) / w ;

  zimt::bill_t bill ;
  bill.njobs = 1 ;
  
  typedef zimt::xel_t < float , 2 > delta_t ;
  delta_t start { ( w - 1 ) / 2.0 , ( h - 1 ) / 2.0  } ;
  start *= d ;
  delta_t step { -d , -d } ;
  zimt::linspace_t < float , 2 , 2 , 16 > ls ( start , step ) ;

  auto * ts = TextureSystem::create() ; 

  auto out = ImageOutput::create ( "a.tif" );
  assert ( out != nullptr ) ;
  ImageSpec ospec ( w , h , 3 , TypeDesc::UINT8 ) ;
  out->open ( "a.tif" , ospec ) ;

  auto store_line = [&] ( const float * p_src ,
                          std::size_t nbytes ,
                          std::size_t line ) -> bool
  {
    // std::cout << "store " << line << " values" << std::endl ;
    return out->write_scanline ( line , 0 , typedesc , p_src );
  } ;

  zimt::line_store_t < float , 3 >
    line_drain ( w , h , pass , store_line ) ;

  TextureOptBatch batch_options ;
  ustring ufilename ( filename ) ;
  lookup_t lookup ( ts , batch_options , ufilename , step ) ;
  zimt::tile_storer < float , 3 , 2 , 16 > tp ( line_drain , bill ) ;
  zimt::process < 2 > ( { w , h } , ls , lookup , tp , bill ) ;
}
