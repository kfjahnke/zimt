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

// This example uses a 'gridspace_t' get_t to provide 3D coordinates of 
// the sample points of a rectilinear image 'draped' in space so that
// it represents the target image after application of a set of three
// Euler angles (ywa, pich, roll).
// Pass a 2:1 environment map (a 'full spherical', 360 degree panorama),
// decide on the size of the output and the output's field of view, also
// pass yaw, picth and roll and the program will produce a rectilinear
// view and write it to 'a.tif'. If you don't touch parameterization,
// this will do a high-quality rendition, with proper antialiasing.
// The program isn't blazingly fast, but it does a good job.

#include <iomanip>
#include <array>
#include <zimt/zimt.h>
#include <zimt/scanlines.h>
#include <Imath/ImathVec.h>
#include <Imath/ImathEuler.h>
#include <Imath/ImathQuat.h>
#include <OpenImageIO/imageio.h>
#include <OpenImageIO/typedesc.h>
#include <OpenImageIO/texture.h>

using namespace OIIO ;

typedef zimt::xel_t < float , 2 > v2_t ;
typedef zimt::xel_t < float , 3 > v3_t ;

// helper function to form the norm of a uniform aggregate. note how
// T can be a SIMDized type - the arithmetic can be formulated so that
// it's good for scalars and SIMDized data. TODO: library code?

template < template < typename , std::size_t > class C ,
           typename T , std::size_t N >
T norm ( const C < T , N > & x )
{
  T sqn = x[0] * x[0] ;
  for ( std::size_t i = 1 ; i < N ; i++ )
    sqn += x[i] * x[i] ;
  return sqrt ( sqn ) ;
}

// at the heart of the program is the 'act' functor. This one
// receives vectorized 3D grid coordinates. It calculates the
// normalized coordinate and it's derivatives, then uses these
// values to obtain pixels with oiio's 'environment' function.
// The incoming 3D grid coordinates are spaced uniformly in a
// plane representing the target image after application of
// the given Euler angles. Because they are equidistant with
// the constant offsets given in 'step', it's easy to move from
// one such coordinate to it's two neighbours in canonical image
// coordinates. But the derivatives have to refer to the surface
// of the unit sphere bounding the directional vectors and are
// therefore caclulated after normalization.

struct lookup_t
: public zimt::unary_functor < v3_t , v3_t , 16 >
{
  TextureSystem * ts ;
  TextureOptBatch & batch_options ;
  TextureSystem::TextureHandle * th ;
  const std::array < v3_t , 2 > step ;

  lookup_t ( TextureSystem * _ts ,
             TextureOptBatch & _batch_options ,
             TextureSystem::TextureHandle * _th ,
             const std::array < v3_t , 2 > & _step )
  : ts ( _ts ) ,
    batch_options ( _batch_options ) ,
    th ( _th ) ,
    step ( _step )
   { }

  // the act functor will - used by zimt::process - only ever process
  // vectorized data, but since this is 'pure' stencil code without
  // conditionals/masks, it would also work for single coordinate-to
  // -pixel calculations.

  template < typename I , typename O >
  void eval ( const I & _crd , O & px ) const
  {
    // Incoming, we have a 3D coordinates pertaining to the image plane
    // which is regularly sampled at O + ds * x + dt * y. We 'drape' the
    // image plane at unit distance, the coordinates are scaled to work
    // at this distance. we normalize the coordinate:

    out_v crd = _crd / norm ( _crd ) ;
    
    // Reasonable approximation of the derivatives: We calculate the
    // Difference from a set of coordinates one step away, both in
    // canonical x and canonical y - after normalization.

    out_v ds = _crd + step[0] ;
    ds /= norm ( ds ) ;
    ds -= crd ;
    
    out_v dt = _crd + step[1] ;
    dt /= norm ( dt ) ;
    dt -= crd ;
    
    // crd[0] yields vector of float, whose .data() yields float*.
    // crd[1], crd[2] follow directly after. ditto for ds, dt.
    // ts->environment writes the result directly into px, so we're
    // done once this call returns.
  
    ts->environment ( th , nullptr, batch_options ,
                      Tex::RunMaskOn ,
                      crd[0].data() , ds[0].data() , dt[0].data() ,
                      3 , px[0].data() ) ;
  }
} ;

// we'll use some Imath code, so we need to be able to move from
// zimt's xel to Imath's Vec.

Imath::V3f & toImath ( v3_t & v )
{
  return * ( (Imath::V3f*) (&v) ) ;
}

v3_t & to_zimt ( Imath::V3f & v )
{
  return * ( (v3_t *) (&v) ) ;
}

v3_t to_zimt ( const Imath::V3f & v )
{
  return * ( (v3_t *) (&v) ) ;
}

// apply an Imath rotationl quaternion to a v3_t

void rotate ( v3_t & v , const Imath::Quatf & q )
{
  v = to_zimt ( toImath ( v ) * q ) ;
}

int main ( int argc , char * argv[] )
{
  // collect arguments

  assert ( argc > 7 ) ;
  std::string filename ( argv[1] ) ;
  int w = std::stoi ( argv[2] ) ;
  int h = std::stoi ( argv[3] ) ;
  float v = std::stod ( argv[4] ) ;
  float yaw = std::stod ( argv[5] ) ;
  float pitch = std::stod ( argv[6] ) ;
  float roll = std::stod ( argv[7] ) ;

  std::cout << "filename: " << filename << " width: " << w
            << " height: " << h << " vfov: " << v << std::endl ;
  std::cout << "yaw: " << yaw << " pitch: " << pitch
            << " roll: " << roll << std::endl ;

  // move to radians, reverse yaw and pitch

  v *= M_PI / 180.0 ;
  yaw *= - M_PI / 180.0 ;
  pitch *= - M_PI / 180.0 ;
  roll *= M_PI / 180.0 ;

  // step width in the unrotated image plane
            
  float d = 2.0 * tan ( v / 2.0 ) / w ;

  // common to the entire program: the 'loading bill'

  zimt::bill_t bill ;

  // figure out the corner point to start out from, and initialize
  // the step vectors

  typedef zimt::xel_t < float , 2 > delta_t ;
  v3_t start { d * ( w - 1 ) / 2.0f , d * ( h - 1 ) / 2.0f  , 1.0f } ;
  std::array < v3_t , 2 > step { 0.0f } ;
  step[0][0] = step[1][1] = -d ;

  // set up the rotational quaternion

  Imath::Eulerf angles ( roll , pitch , yaw , Imath::Eulerf::ZXY ) ;
  Imath::Quat q = angles.toQuat() ;

  // and apply it to start and the step vectors

  rotate ( start , q ) ;
  rotate ( step[0] , q ) ;
  rotate ( step[1] , q ) ;

  // for input to zimt::process, we set up a 'gridspace' - a regular grid
  // of 3D coordinates. The 'upper left' corner is at 'start', each step
  // along axis 0 adds step[0] to the coordinate, and each step along
  // axis 1 adds step[1] to the coordinate. The resulting 3D coordinates
  // are used by the 'act' functor (after normalization) to sample the
  // environment texture.

  zimt::gridspace_t < float , 3 , 2 , 16 > ls ( start , step ) ;

  // now we set up the TextureSystem, which is really simple.

  auto * ts = TextureSystem::create() ; 

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

  TextureOptBatch batch_options ;

  // for ( int i = 0 ; i < 16 ; i++ )
  //   batch_options.swidth[i] = batch_options.twidth[i] = 0 ;

  // This is a bit awkward - the batch_options don't accept plain
  // TextureOpt enums (there's a type error) - but that is maybe due to
  // the rather old OIIO I take from debian's packet menegement.

  // batch_options.mipmode = Tex::MipMode ( TextureOpt::MipModeNoMIP ) ;
  // batch_options.interpmode
  //   = Tex::InterpMode ( TextureOpt::InterpBilinear ) ;

  // batch_options.conservative_filter = false ;

  // we obtain the texture handle for most efficient processing of the
  // environment lookup, and codify the act functor as a lookup_t

  ustring ufilename ( filename ) ;
  auto th = ts->get_texture_handle ( ufilename ) ;
  lookup_t lookup ( ts , batch_options , th , step ) ;

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
