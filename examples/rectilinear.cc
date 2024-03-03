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
// via the filename, which is less efficient. Pass a 2:1 environment
// map (a 'full spherical', 360 degree panorama), decide on the size
// of the output and the output's field of view, and the program will
// produce a rectilinear view and write it to 'a.tif'.

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

// at the heart of the program is the 'act' functor. This one
// receives vectorized 2D grid coordinates. It adds the z component
// to make up a 3D directional vector - here a constant value of
// 1.0, since we're rendering to an upright view perpendicular to
// the viewing axis. We use constant values for the derivatives
// throughout, using the step width in x and y direction in the
// image's center and zero for the derivative of the z component,
// which is true in nominal values but not in normalized ones.
// With the parameter set set uo, we call into OIIO, receiving
// the batch of pixels corresponding to the batch of coordinates.
// the pixels are the act functor's output.

struct lookup_t
: public zimt::unary_functor < v2_t , v3_t , 16 >
{
  TextureSystem * ts ;
  TextureOptBatch & batch_options ;
  TextureSystem::TextureHandle * th ;
  float d ;

  lookup_t ( TextureSystem * _ts ,
             TextureOptBatch & _batch_options ,
             TextureSystem::TextureHandle * _th ,
             float _d )
  : ts ( _ts ) ,
    batch_options ( _batch_options ) ,
    th ( _th ) ,
    d ( _d )
   { }

  template < typename I , typename O >
  void eval ( const I & _crd , O & px ) const
  {
    out_v crd , delta ;

    crd[0] = _crd[0] ;
    crd[1] = _crd[1] ;
    crd[2] = 1.0f ;

    delta[0] = d ;
    delta[1] = d ;
    delta[2] = 0 ;

    const float * R = (const float*) crd.data() ;
    const float * pd = (const float*) delta.data() ;
    float * res = (float*) px.data() ;

    ts->environment ( th , nullptr, batch_options ,
                      Tex::RunMaskOn ,
                      R , pd , pd , 3 , res ) ;
  }
} ;

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

  // common to the entire program: the 'loading bill'

  zimt::bill_t bill ;
  
  // for input, we set up a 'linspace' - a regular grid of 2D
  // coordinates

  typedef zimt::xel_t < float , 2 > delta_t ;
  delta_t start { ( w - 1 ) / 2.0 , ( h - 1 ) / 2.0  } ;
  start *= d ;
  delta_t step { -d , -d } ;
  zimt::linspace_t < float , 2 , 2 , 16 > ls ( start , step ) ;

  // to set up the 'act' functor, we need the OIIO texture system
  // and a few parameters

  auto * ts = TextureSystem::create() ; 
  TextureOptBatch batch_options ;

  // The options for the lookup determine the quality of the output
  // and the time it takes to compute it. Switching MipMapping off
  // for example reduces processing time greatly, and the other
  // commented-out settings also reduce processing time, sacrificing
  // quality for speed. My conclusion is that zimt and OIIO play
  // well together, but I suspect that allowing invariants for
  // options which can be tuned on a per-lane basis might reduce
  // processing load with little loss of quality. If the MIP
  // level could be fixed for the entire run (like when it's switched
  // off altogether) rather than looking at the derivatives, I think
  // processing would speed up nicely. Just guessing, though - I
  // haven't looked at the code.

  // for ( int i = 0 ; i < 16 ; i++ )
  //   batch_options.swidth[i] = batch_options.twidth[i] = 0 ;

  // This is a bit awkward - the batch_options don't accept plain
  // TextureOpt enums (there's a type error) - but that is maybe due to
  // the rather old OIIO I take from debian's packet menegement.

  // batch_options.mipmode = Tex::MipMode ( TextureOpt::MipModeNoMIP ) ;
  // batch_options.interpmode
  //   = Tex::InterpMode ( TextureOpt::InterpBilinear ) ;

  // batch_options.conservative_filter = false ;

  ustring ufilename ( filename ) ;
  auto th = ts->get_texture_handle ( ufilename ) ;
  lookup_t lookup ( ts , batch_options , th , d ) ;

  // finally the data sink - we store to an array, which makes it
  // possible to multithread the code.

  zimt::array_t < 2 , v3_t > trg  ( { w , h } ) ;
  zimt::storer < float , 3 , 2 , 16 > st ( trg , bill ) ;

  // showtime! zimt::process fills the target array with data

  zimt::process < 2 > ( { w , h } , ls , lookup , st , bill ) ;

  // finally we store the data to an image file - note how we have
  // float data in 'trg', and OIIO will convert these on-the-fly to
  // UINT8, as specified in the write_image invocation.

  auto out = ImageOutput::create ( "a.tif" );
  assert ( out != nullptr ) ;
  ImageSpec ospec ( w , h , 3 , TypeDesc::UINT16 ) ;
  out->open ( "a.tif" , ospec ) ;

  out->write_image ( TypeDesc::FLOAT , trg.data() ) ;
  out->close();
}
