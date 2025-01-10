/************************************************************************/
/*                                                                      */
/*    zimt - abstraction layer for SIMD programming                     */
/*                                                                      */
/*            Copyright 2024 by Kay F. Jahnke                           */
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

// Simple utility program producing a cubemap from a lat/lon
// 'environment map'. This modifies 'rectilinear.cc', making the
// initial main into a function and calling it six times with varying
// orientation, to 'look at' all six cube faces in turn and produce
// a square image of the cube face, which is stored at the appropriate
// location in a 1:6 stripe representing the cubemap. The precise
// orientation and sequence of the cubeface images in the stripe
// are chosen to coincide with openEXR cubemap format. For the
// technicalities of the rendition, refer to rectilinear.cc.
// Note that there is now a program 'envutil.cc' which can convert
// both ways and is more comprehensive. For 'real' work use envutil;
// this program is more a zimt demo program.

// TODO: extend this program to deal with other channel counts 

#include <array>

#include <zimt/zimt.h>

#include <Imath/ImathVec.h>
#include <Imath/ImathEuler.h>
#include <Imath/ImathQuat.h>

#include <OpenImageIO/texture.h>

using namespace OIIO ;

typedef decltype (TextureSystem::create()) ts_ptr ;

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
  ts_ptr ts ;
  TextureOptBatch & batch_options ;
  TextureSystem::TextureHandle * th ;
  const zimt::xel_t < v3_t , 2 > step ;

  // construct the 'act' functor. The first three parameters are
  // needed for the environment lookup - we use the TextureHandle
  // because we're only looking up values from a single texture,
  // and passing the handle is faster than passing the image file's
  // name.
  // 'step' is a pair of short 3D vectors. If the view is head-on
  // to the environment map - so, the viewing axis is x,y,z = 0,0,1
  // in IMath coordinate space - these two vectors would each only
  // contain one nonzero component - the first one would have an x
  // value and the second an y yalue. The step refers to the progress
  // of the coordinates in the 'draped' image - it's constituent sample
  // points distributed in space in a plane one unit step from the
  // coordinate system's origin. With a head-on view, obviously the
  // steps will be either horizontal - so with increasing x - or
  // vertical - with increasing y. But if the view is rotated (with
  // the Euler angles passed in), the steps also have to be rotated,
  // and, depending on the rotation, all three components may hold
  // non-zero values.
  // Note that 'step' refers to steps in the image's coordinate
  // system (the 'canonical' coordinates). Applying step[0] once
  // is akin to increasing the colum, applying step[1] is akin to
  // increasing the line. Note also that the 'canonical' image
  // coordinates increase from the image's top to it's bottom, while
  // the spatial y axis in IMath points upward, so you may notice
  // signs changes on the y axis in the code. The image's x axis and
  // the IMath x axis coincide.

  lookup_t ( ts_ptr _ts ,
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
    // Incoming, we have 3D coordinates pertaining to the image plane
    // which is regularly sampled at UL + step[0] * x + step[1] * y,
    // Where UL is the upper left. We have 'draped' the image plane
    // at unit distance, and the coordinates are scaled to work at this
    // distance. we normalize the coordinate to move to points on the
    // unit sphere - equivalent to the directional vectors we want to
    // feed to the environment lookup.
    // TODO: since we normalize anyway, moving from image coordinates
    // to model space coordinates is futile. We can sample the image
    // e.g. at unit steps and at a corresponding distance. Any scale
    // will do - also twice unit steps, as I use them now in latlon.cc

    in_v crd = _crd / norm ( _crd ) ;
    
    // Reasonable approximation of the derivatives: We calculate the
    // difference from a set of coordinates one step away, both in
    // canonical x and canonical y - after normalization. So 'step'
    // is applied to the incoming coordinates (before normalization),
    // And then the 'neighbouring' coordinates are normalized before
    // the difference to the normalized lookup coordinates is calculated.

    in_v ds = _crd + step[0] ;
    ds /= norm ( ds ) ;
    ds -= crd ;
    
    in_v dt = _crd + step[1] ;
    dt /= norm ( dt ) ;
    dt -= crd ;

    // Note the simplicity of interfacing the zimt code with the OIIO
    // code, and note also that there is a good chance that the compiler
    // will not even 'go through memory' to transfer zimt's SIMD data
    // to what will later on be OIIO's equivalent SIMD data, and
    // instead do the entire operation in registers, making the code
    // as fast as possible. How well the integration succeeds will
    // depend on the compiler used, and the specific SIMD backend
    // used by zimt.
    // Not also that the zimt::process 'driver' code which calls the
    // 'act' functor repeatedly is multithreaded: there will be several
    // worker threads which cooperate to get through all the lookups
    // neded for the entire target image. All of the parcelling of
    // the data, distributing the workload to threads, generation of
    // incoming coordinates, and disposal of the resulting data to RAM
    // is handled by zimt and is it's 'native domain'. It leaves you
    // to write the 'interesting' code. As of this writing, using data()
    // on zimt data types only works with the highway and zimt backends.
    // For the Vc and std::simd backend, the SIMDized objects must be
    // unpacked to memory to be re-loaded by the OIIO code, and the
    // result has to be re-loaded by the zimt objects. Again, with a
    // bit of luck, the compiler may 'get it', but the 'direct way'
    // is more promising. Using 'zimt's own' backend is relying on
    // autovectorization and may be quite slow. I found that processing
    // time varies quite a bit without a pattern I could recognize.
    // Note that highway code can be made faster by passing additional
    // compiler flags specifying the architecture more precisely than
    // the 'blanket' -march=native option we use here, which is also
    // specific to intel CPUs. If you're working on a different
    // architecture, you will need to pass different compiler options
    // to get 'native' binary SIMD code.
    // Note, finally, how zimt gives you the choice of four different
    // SIMD backends by simply setting a preprocessor #define.

#if defined USE_VC or defined USE_STDSIMD

    // to interface with zimt's Vc and std::simd backends, we need to
    // extract the data from the SIMDized objects and re-package the
    // ouptut as a SIMDized object. The compiler will likely optimize
    // this away and work the entire operation in registers, so let's
    // call this a 'semantic manoevre'.

    float scratch [ 4 * 3 * 16 ] ;

    crd.store ( scratch ) ;
    ds.store ( scratch + 3 * 16 ) ;
    dt.store ( scratch + 6 * 16 ) ;

    ts->environment ( th , nullptr, batch_options ,
                      Tex::RunMaskOn ,
                      scratch , scratch + 3 * 16 , scratch + 6 * 16 ,
                      3 , scratch + 9 * 16 ) ;

    px.load ( scratch + 9 * 16 ) ;

#else

    // the highway and zimt's own backend have an internal representation
    // as a C vector of fundamentals, so we van use data() on them, making
    // the code even simpler - though the code above would work just the
    // same.

    ts->environment ( th , nullptr, batch_options ,
                      Tex::RunMaskOn ,
                      crd[0].data() , ds[0].data() , dt[0].data() ,
                      3 , px[0].data() ) ;

#endif
  }
} ;

// we'll use some Imath code in the next section, so we need to be able
// to move from zimt's 'xel' data to Imath's Vec. This requires merely
// a cast - the layout is the same.

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

// apply an Imath rotational quaternion to a v3_t. This is an important
// bit of support we receive from IMath - it accepts Euler angles, and
// it can represent them as a rotational quaternion. This makes the
// formulation of the rotations we need in this program very succinct:
// we simply apply the quaternion to both the image's upper left
// corner - the (0,0) sample of the image - and to the step vectors.

void rotate ( v3_t & v , const Imath::Quatf & q )
{
  v = to_zimt ( toImath ( v ) * q ) ;
}

void extract ( ts_ptr ts ,
               TextureSystem::TextureHandle * th ,
               int w , int h ,
               float v , float yaw , float pitch , float roll ,
               zimt::view_t < 2 , v3_t > trg )
{
  // step width in the unrotated image plane
            
  float d = 2.0 * tan ( v / 2.0 ) / w ;

  // common to the entire program: the 'loading bill'

  zimt::bill_t bill ;

  // figure out the corner point to start out from, and initialize
  // the step vectors. x and y start out positive and decrease,
  // and z is at positive untit distance.

  typedef zimt::xel_t < float , 2 > delta_t ;

  v3_t start { d * ( w - 1 ) / 2.0f ,
               d * ( h - 1 ) / 2.0f  ,
               1.0f } ;

  std::array < v3_t , 2 > step { 0.0f } ;

  step[0][0] = - d ;
  step[1][1] = - d ;

  // set up the rotational quaternion

  Imath::Eulerf angles ( roll , pitch , yaw , Imath::Eulerf::ZXY ) ;
  Imath::Quat q = angles.toQuat() ;

  // and apply it to start and the step vectors

  rotate ( start , q ) ;
  rotate ( step[0] , q ) ;
  rotate ( step[1] , q ) ;

  // for input to zimt::process, we set up a 'gridspace' providing evenly
  // spaced 3D coordinates. The 'upper left' corner is at 'start', each
  // step along axis 0 adds step[0] to the coordinate, and each step along
  // axis 1 adds step[1] to the coordinate. The resulting 3D coordinates
  // are used by the 'act' functor (after normalization) to sample the
  // environment texture. Note that the gridspace_t object does not
  // actually contain any coordinates - like an array would - but that
  // it is instead a construct yielding values as needed: it's a
  // generator. This is much more efficient than having an array
  // of coordinates at the source - in a way it's like pulling the
  // input 'out of thin air'.

  zimt::gridspace_t < float , 3 , 2 , 16 > ls ( start , step ) ;

  // The options for the lookup determine the quality of the output
  // and the time it takes to compute it. Switching MipMapping off
  // for example reduces processing time greatly, and the other
  // settings also reduce processing time, sacrificing quality for
  // speed. Please refer to the OIIO documentation for the batch
  // lookup options - here, I simply use the defaults.

  TextureOptBatch batch_options ;

  // Note that the batch options aren't fully initialized (this is
  // a bug), so as a workaround I initialize some of them manually:

  for ( int i = 0 ; i < 16 ; i++ )
    batch_options.swidth[i] = batch_options.twidth[i] = 1 ;
  
  for ( int i = 0 ; i < 16 ; i++ )
    batch_options.sblur[i] = batch_options.tblur[i] = 0 ;

  lookup_t lookup ( ts , batch_options , th , step ) ;

  // finally the data sink - we store to a zimt array, which makes it
  // possible to multithread the code - rather than writing scanlines,
  // which is also possible, but can't be multithreaded with zimt,
  // because that can't guarantee that the scanlines are produced
  // in strictly ascending order.

  zimt::xel_t < std::size_t , 2 > shape { w , h } ;

  // zimt::array_t < 2 , v3_t > trg  ( shape ) ;
  zimt::storer < float , 3 , 2 , 16 > st ( trg , bill ) ;

  // showtime! zimt::process fills the target array with data.
  // The arguments are, in order:
  // - the 'notional' shape of the operation
  // - the source of input to the 'act' functor
  // - the act functor itself
  // - the data sink, receiving rendered pixels
  // - the 'loading bill' providing shared zimt parameterization

  zimt::process ( shape , ls , lookup , st , bill ) ;
}

int main ( int argc , char * argv[] )
{
  // collect arguments
  // TODO: consider 'normal' argument syntax.

  if ( argc != 4 )
  {
    std::cerr << "usage: cubemap <latlon> <width> <cubemap>" << std::endl ;
    std::cerr << "latlon must contain a lat/lon environment map" << std::endl ;
    std::cerr << "  a.k.a. full spherical panorama image" << std::endl ;
    std::cerr << "width is the width of a single cube face" << std::endl ;
    std::cerr << "cubemap is the filename for the output, a 1:6" << std::endl ;
    std::cerr << "  stripe with the six cube faces, compatible" << std::endl ;
    std::cerr << "  with openEXR cubemap environment format." << std::endl ;
    exit ( -1 ) ;
  }

  std::string environment ( argv[1] ) ;
  int w = std::stoi ( argv[2] ) ;
  std::string output ( argv[3] ) ;

  std::cout << "environment: " << environment
            << " cubeface width: " << w
            << " output: " << output << std::endl ;

  // now we set up the TextureSystem, which is really simple.

  ts_ptr ts = TextureSystem::create() ; 

  // we're only dealing with one environment here, let's get
  // it's handle to speed up access further down the line

  ustring uenvironment ( environment ) ;
  auto th = ts->get_texture_handle ( uenvironment ) ;

  // let's see if we can glean information about this environment
  // using this TextureSystem function:
  
  // virtual bool get_texture_info(ustring filename, int subimage,
  //          ustring dataname, TypeDesc datatype, void *data)

  char * p_texturetype ;
  bool success = ts->get_texture_info
    ( uenvironment , 0 , ustring("texturetype") ,
      TypeDesc::STRING , &p_texturetype ) ;

  if ( success )
    std::cout << "gleaned texture type '" << p_texturetype
              << "'" << std::endl ;

  int resolution[2] ;
  success = ts->get_texture_info
    ( uenvironment , 0 , ustring("resolution") ,
      TypeDesc(TypeDesc::INT,2) , resolution ) ;  

  assert ( success ) ;
  int width = resolution[0] ;
  int height = resolution[1] ;
  if ( width != height * 2 )
  {
    std::cerr << "expecting a lat/lon environment in 2:1 aspect ratio"
              << std::endl ;
    std::cerr << "input has extent " << width << " X " << height
              << std::endl ;
    exit ( -1 ) ;
  }

  float v = M_PI_2 ;

  zimt::xel_t < std::size_t , 2 > shape { w , w } ;
  zimt::xel_t < long , 2 > strides { 1 , w } ;

  zimt::array_t < 2 , v3_t > stripe ( { w , w * 6 } ) ;

  // now we render the six cube faces. Note how we create the target 'view'
  // with increasing base address, visiting the six square sections of the
  // stripe. The varying yaw, pitch and roll values orient the camera so as
  // to capture the cube face with the correct orientation.

  {
    zimt::view_t < 2 , v3_t > trg ( stripe.data() + 0 * ( w * w ) ,
                                    strides , shape ) ;
    extract ( ts , th , w , w , v , M_PI_2 , 0 , 0 , trg ) ;
  }
  {
    zimt::view_t < 2 , v3_t > trg ( stripe.data() + 1 * ( w * w ) ,
                                    strides , shape ) ;
    extract ( ts , th , w , w , v , - M_PI_2 , 0 , 0 , trg ) ;
  }
  {
    zimt::view_t < 2 , v3_t > trg ( stripe.data() + 2 * ( w * w ) ,
                                    strides , shape ) ;
    extract ( ts , th , w , w , v , 0 , - M_PI_2 , - M_PI , trg ) ;
  }
  {
    zimt::view_t < 2 , v3_t > trg ( stripe.data() + 3 * ( w * w ) ,
                                    strides , shape ) ;
    extract ( ts , th , w , w , v , 0 , M_PI_2 , 0- M_PI , trg ) ;
  }
  {
    zimt::view_t < 2 , v3_t > trg ( stripe.data() + 4 * ( w * w ) ,
                                    strides , shape ) ;
    extract ( ts , th , w , w , v , 0 , 0 , 0 , trg ) ;
  }
  {
    zimt::view_t < 2 , v3_t > trg ( stripe.data() + 5 * ( w * w ) ,
                                    strides , shape ) ;
    extract ( ts , th , w , w , v , M_PI , 0 , 0 , trg ) ;
  }

  // finally we store the data to an image file - note how we have
  // float data in 'trg', and OIIO will convert these on-the-fly to
  // UINT16, as specified in the write_image invocation.
  // Note that the target will receive image data in the same colour
  // space as the input. If you feed, e.g. openEXR, and store to JPEG,
  // the image will look too dark, because the linear RGB data are
  // stored as if they were sRGB.

  auto out = ImageOutput::create ( output );
  assert ( out != nullptr ) ;
  ImageSpec ospec ( w , w * 6 , 3 , TypeDesc::UINT16 ) ;
  out->open ( output , ospec ) ;

  out->write_image ( TypeDesc::FLOAT , stripe.data() ) ;
  out->close();
}
