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

// Simple utility program producing a lat/lon 'environment map' from
// a cubemap. TODO work in progress

#include <array>

#include <zimt/zimt.h>

#include <Imath/ImathVec.h>
#include <Imath/ImathEuler.h>
#include <Imath/ImathQuat.h>

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

// enum encoding the sequence of cube face images in the cubemap

enum
{
  CM_LEFT ,
  CM_RIGHT ,
  CM_TOP ,
  CM_BOTTOM ,
  CM_FRONT ,
  CM_BACK
} ;

// sixfold_t contains data for a six-square sky box with support
// around the cube faces to allow for proper filtering and
// interpolation. The content is set up following conventions
// of openEXR environment maps. cubemaps on openEXR conatin six
// square images concatenated vertically. The sequence of the
// images, from top to bottom, is: left, right, top, bottom,
// front, back. The top and bottom image are oriented so that
// they align vertically with the 'back' image.
// The sixfold_t object also combines the six images in one
// array, but it adds a frame of additional 'support' pixels
// around the square images to allow for interpolation with
// interpolators needing support around the interpolation locus.
// The supporting frame around each square image is chosen to
// yield a total per-cube-face size which is a multiple of the
// tile size, to allow for mip-mapping. With a bit of indexing
// magic, we'll be able to form 'pick-up' coordinates pertaining
// directly to the entire array held in the sixfold_t, which is
// more efficient than combining pixel values from per-cube-face
// lookups - we can code the entire process in SIMD code. The
// frames around the cube faces make it possible to do this with
// mip-mapping and correct anti-aliasing, using OIIO's planar
// texture lookup. This latter part is not yet implemented, for
// now I aim at getting the geometry and logic right and use a
// simple nearest-neighbour lookup to get pixel values. Another
// issue to be implemented is population of the frame with suitable
// pixel data.

struct sixfold_t
{
  const std::size_t face_width ;
  const std::size_t outer_width ;
  const std::size_t tile_size ;
  std::size_t frame ;

  zimt::array_t < 2 , v3_t > store ;

  zimt::view_t < 2 , v3_t > front ;
  zimt::view_t < 2 , v3_t > back ;
  zimt::view_t < 2 , v3_t > left ;
  zimt::view_t < 2 , v3_t > right ;
  zimt::view_t < 2 , v3_t > top ;
  zimt::view_t < 2 , v3_t > bottom ;

  std::vector < float * > p_face_v ;

  sixfold_t ( std::size_t _face_width ,
              std::size_t _tile_size = 64 )
  : face_width ( _face_width ) ,
    tile_size ( _tile_size ) ,
    outer_width ( ( 1UL + _face_width / _tile_size ) * _tile_size ) ,
    store ( {     ( 1UL + _face_width / _tile_size ) * _tile_size ,
              6 * ( 1UL + _face_width / _tile_size ) * _tile_size } )
  {
    // make sure tile size is a power of two

    assert ( ( tile_size & ( tile_size - 1 ) ) == 0 ) ;

    // we want some headroom for sure, let's say four pixels

    assert ( outer_width >= ( tile_size + 4 ) ) ;
    std::cout << "outer_width: " << outer_width << std::endl ;

    frame = ( outer_width - face_width ) / 2 ;
    std::cout << "frame: " << frame << std::endl ;

    // find the location of the first cube face's upper left corner
    // in the 'store' array

    v3_t * p = store.data() ;
    std::ptrdiff_t offset = outer_width * outer_width ;
    std::cout << "offset: " << offset << std::endl ;

    p += frame * store.strides[0] ;
    p += frame * store.strides[1] ;

    // and save the pointer in the array 'p_face_v'
  
    p_face_v.push_back ( (float*) p ) ;

    typedef zimt::xel_t < std::size_t , 2 > shape_type ;
    shape_type face_shape { face_width , face_width } ;

    // now we set up views to the individual cube faces inside
    // the 'store' array, so that we can access each cube face
    // individually, if we need to do so. This feature is not
    // currently used in this program - here we use the pointers
    // in p_face_v.

    left.shallow_copy ( { p , store.strides , face_shape } ) ;

    p += offset ;
    p_face_v.push_back ( (float*) p ) ;
    right.shallow_copy ( { p , store.strides , face_shape } ) ;

    p += offset ;
    p_face_v.push_back ( (float*) p ) ;
    top.shallow_copy ( { p , store.strides , face_shape } ) ;

    p += offset ;
    p_face_v.push_back ( (float*) p ) ;
    bottom.shallow_copy ( { p , store.strides , face_shape } ) ;

    p += offset ;
    p_face_v.push_back ( (float*) p ) ;
    front.shallow_copy ( { p , store.strides , face_shape } ) ;

    p += offset ;
    p_face_v.push_back ( (float*) p ) ;
    back.shallow_copy ( { p , store.strides , face_shape } ) ;
  }

  // function to read the image data from disk
  // (via an OIIO-provided inp)

  void load ( std::unique_ptr<ImageInput> & inp )
  {
    // paranoid.

    assert ( inp != nullptr ) ;

    const ImageSpec &spec = inp->spec() ;
    int xres = spec.width ;
    int yres = spec.height ;

    std::cout << "spec.width: " << spec.width << std::endl ;

    assert ( xres == face_width ) ;
    assert ( yres == 6 * face_width ) ;

    // for now, we only process three channels, even if the input
    // conatins more.

    int nchannels = spec.nchannels ;

    std::cout << "store.strides: " << store.strides << std::endl ;

    // read the six cube face images from the 1:6 stripe into the
    // appropriate slots in the sixfold_t object's 'store' array

    for ( int face = 0 ; face < 6 ; face++ )
    {
      auto * p_trg = p_face_v [ face ] ;

      // for reference: OIIO's read_scanlines' signature

      // virtual bool read_scanlines ( int subimage, int miplevel,
      //                               int ybegin, int yend,
      //                               int z, int chbegin, int chend,
      //                               TypeDesc format, void *data,
      //                               stride_t xstride = AutoStride,
      //                               stride_t ystride = AutoStride)

      // note how we read face_width scanlines in one go, using
      // appropriate strides to place the image data inside the
      // larger 'store' array, converting to float as we go along.
      // THe channels are capped at three, discarding alpha, z, etc.

      inp->read_scanlines ( 0 , 0 ,
                            face * face_width , (face+1) * face_width ,
                            0 , 0 , 3 , // TODO cater for more channels
                            TypeDesc::FLOAT , p_trg ,
                            3 * 4 ,
                            3 * 4 * store.strides[1] ) ;
    }
  }
} ;

// convert_t is the functor used as 'act' functor for zimt::process.
// This functor accesses the data in the sixfold_t object, yielding
// pixel data for incoming 2D lat/lon coordinates. It implements a
// typical processing pipeline, using three distinct stages: first,
// the incoming 2D lat/lon coordinates are converted to 3D 'ray'
// coordinates, then these are used to figure out the corresponding
// cube face and the 2D in-face coordinates, and finally these values
// are used to obtain pixel data from the sixfold_t object.
// Note how we code the entire process as a SIMD operation: we're
// not handling individual coordinates or pixels, but their 'SIMDized'
// equivalents.

struct convert_t
: public zimt::unary_functor < v2_t , v3_t , 16 >
{
  // source of pixel data

  const sixfold_t & cubemap ;

  // some SIMDized types we'll use

  typedef typename zimt::simdized_type < float , 16 > f_v ;
  typedef typename zimt::simdized_type < v2_t , 16 > crd2_v ;
  typedef typename zimt::simdized_type < v3_t , 16 > crd3_v ;
  typedef typename zimt::simdized_type < v3_t , 16 > px_v ;

  // convert_t's c'tor obtains a const reference to the sixfold_t
  // object holding pixel data

  convert_t ( const sixfold_t & _cubemap )
  : cubemap ( _cubemap )
  { }

  // code from lux to convert lat/lon to 3D ray. Note that I am using
  // lux coordinate convention (book order: x is right, y down and
  // z forward). I use a template here, the code is good for scalar
  // values and SIMD data alike.

  template < typename in_type , typename out_type >
  void ll_to_ray ( const in_type & in ,
                   out_type & out ) const
  {
    auto const & x1 ( in[0] ) ;
    auto const & y1 ( in[1] ) ;

    auto & z2 ( out[0] ) ;
    auto & x2 ( out[1] ) ;
    auto & y2 ( out[2] ) ;

    z2 = cos ( x1 ) * cos ( y1 ) ;
    x2 = sin ( x1 ) * cos ( y1 ) ;
    y2 = sin ( y1 ) ;
  }

  // given a 3D 'ray' coordinate, find the corresponding cube face
  // and the in-face coordinate - note the two references which take
  // the result values.

  void ray_to_cubeface ( const crd3_v & c ,
                         f_v & face ,
                         crd2_v & in_face ) const
  {
    // form three masks with relations of the numerical values of
    // the 'ray' coordinate. These are sufficient to find out which
    // component has the largest absolut value (the 'dominant' one,
    // along the 'dominant' axis

    auto m1 = ( abs ( c[0] ) >= abs ( c[1] ) ) ;
    auto m2 = ( abs ( c[0] ) >= abs ( c[2] ) ) ;
    auto m3 = ( abs ( c[1] ) >= abs ( c[2] ) ) ;

    // instead of the code lower down, an alternative strategy to
    // figure out the cube face would be to first test whether
    // the z axis is dominant, and if not, derive the cube face
    // directly from the longitude with a modulo operation. But
    // this depends on actually having lat/lon coordinates. We do
    // have them here, but I prefer to code not just for this
    // situation but to have a general way of working from 3D
    // coordinates to face index plus in-face coordinates.

    // masks which are true where a specific axis is 'dominant'.
    // the three 'dom' masks are mutually exclusive. dom0 is for
    // axis 0 etc.

    auto dom0 = m1 & m2 ;
    auto dom1 = ( ! m1 ) & m3 ; 
    auto dom2 = ( ! m2 ) & ( ! m3 ) ;

    // Now we can assign face indexes, stored in f. If the coordinate
    // value is negative along the dominant axis, we're looking at
    // the face opposite and assign a face value one higher.

    face = CM_FRONT ;
    face ( dom0 & ( c[0] < 0 ) ) = CM_BACK ;
    face ( dom1 ) = CM_LEFT ;
    face ( dom1 & ( c[1] < 0 ) ) = CM_RIGHT ;
    face ( dom2 ) = CM_TOP ;
    face ( dom2 & ( c[2] < 0 ) ) = CM_BOTTOM ;

    // find the in-face coordinates, start for dom0 (x axis)
    // we divide the two non-dominant coordinate values by the
    // dominant one. One of the axes comes out just right when
    // dividing by the absolute value - e.g. the vertical axis
    // is just as upright for all the four cube faces around
    // the center. The other axis is divided by the 'major'
    // coordinate value as-is; the resulting coordinate runs
    // one way for positive major values and backwards for
    // negative ones. Note that we might capture the absolute
    // values (which we've used before) in variables, but the
    // compiler will recognize the common subexpressions and
    // do it for us.

    // since we have - expensive - divisions here, we might
    // first test whether the dom mask is populated at all.
    // an alternative to the division is to use a multiplication
    // with the reciprocal value of the absolute value followed
    // by a multiplication with the sign of c[i] for those ops
    // which don't use the absolute values. This may be faster,
    // saving three divisions, but the code is clearer like this.

    in_face[0] ( dom0 ) = - c[1] / c[0] ;
    in_face[1] ( dom0 ) = - c[2] / abs ( c[0] ) ;

    in_face[0] ( dom1 ) =   c[0] / c[1] ;
    in_face[1] ( dom1 ) = - c[2] / abs ( c[1] ) ;

    // the top and bottom images could each be oriented in four
    // different ways. The orientation I expect in this program
    // is openEXR's cubemap format, where the top and bottom
    // image align with the 'back' image (the last one down).

    in_face[0] ( dom2 ) =   c[1] / abs ( c[2] ) ;
    in_face[1] ( dom2 ) = - c[0] / c[2] ;
  }

  // given the cube face and the in-face coordinate, extract the
  // corresponding pixel values from the internal representation
  // of the cubemap held in the sixfold_t object

  void cubemap_to_pixel ( const f_v & face ,
                          crd2_v in_face ,
                          px_v & px ) const
  {
    // for now, we'll simply do a nearest-neighbour lookup.

    typedef zimt::xel_t < long , 2 > index_type ;
    typedef zimt::simdized_type < index_type , 16 > index_v ;

    // each cube face spans a coordinate range of (-1,1), dy
    // is the distance in in-square units to go to the same
    // location in the next square down. This trick allows us
    // to use a single environment lookup from a single texture
    // even though we have six discrete images on the texture.

    in_face += 1.0f ; // in_face is now in [0,2]

    const double dx = 2.0 * double ( cubemap.frame )
                          / double ( cubemap.face_width ) ;

    const double dy = 2.0 * double ( cubemap.outer_width )
                          / double ( cubemap.face_width ) ;

    // add per-facet offsets to the 2D in-face coordinate
  
    in_face[0] += dx ;
    in_face[1] += face * dy + dx ;

    // scale up to cubemap image coordinates

    in_face *= ( cubemap.face_width / 2.0 ) ;

    // now do the texture lookup

    // for the time being, we do a straight NN pick-up. This way
    // , we establish that the geometry works out as expected, and
    // we can add the proper texture lookup later on as a refinement,
    // so given an evaluator ev (TODO) we derive the pixel value.

    // convert the in-face coordinates to integer - using simple
    // truncation for now.

    index_v idx { in_face[0] , in_face[1] } ;

    // calculate corresponding offsets, using the strides, and
    // obtain pixel data by gathering with this set of offsets.
    // Note how the first multiplication broadcasts the pair of
    // strides to the pair of int vectors

    const auto ofs = ( idx * cubemap.store.strides ) . sum() * 3 ;
    const auto * p = (float*) ( cubemap.store.data() ) ;

    px.gather ( p , ofs ) ;
  }

  // 'eval' function which will be called by zimt::process.
  // We might just put all the code from the three functions above
  // together, but I prefer to have three distinct steps, which may
  // come in handy later on when the code is put to use beyond the
  // immediate aim to code the cubemap-to-lat/lon conversion

  // incoming: 2D coordinate as lat/lon in radians. Internally, we
  // first calculate 3D directional vector, then the index of the
  // cube face and then the 2D in-face coordinate, where we pick up
  // the pixel value from the cube face image. We form 2D pick-up
  // coordinates into the array held i the sixfold_t object, which
  // contains the images 'embedded' in additional support space,
  // which offers 'headroom' for interpolators which need support
  // and also provides each cube face image with just so much frame
  // that the sixth of the total array it inhabits can be tiled
  // exactly with the given tile size. This is to enable correct
  // mip-mapping for filtered texture lookup.

  void eval ( const crd2_v & lat_lon , px_v & px )
  {
    // convert 2D spherical coordinate to 3D 'c'
  
    crd3_v crd3 ;
    ll_to_ray ( lat_lon , crd3 ) ;

    // find the cube face and in-face coordinate for 'c'

    f_v face ;
    crd2_v in_face ;
    ray_to_cubeface ( crd3 , face , in_face ) ;

    // use this information to obtain pixel values

    cubemap_to_pixel ( face , in_face , px ) ;
  }

} ;

int main ( int argc , char * argv[] )
{
  // collect arguments
  // TODO: consider 'normal' argument syntax.

  if ( argc != 4 )
  {
    std::cerr << "usage: skybox <cubemap> <height> <latlon>" << std::endl ;
    std::cerr << "cubemap must contain a 1:6 single-image cubemap" << std::endl ;
    std::cerr << "height is the height of the output" << std::endl ;
    std::cerr << "latlon is the filename for the output, a full" << std::endl ;
    std::cerr << "spherical in openEXR latlon environment format." << std::endl ;
    exit ( -1 ) ;
  }

  std::string skybox ( argv[1] ) ;
  int height = std::stoi ( argv[2] ) ;
  std::string latlon ( argv[3] ) ;

  std::cout << "skybox: " << skybox
            << " output height: " << height
            << " output: " << latlon << std::endl ;

  auto inp = ImageInput::open ( skybox ) ;
  assert ( inp != nullptr ) ;

  const ImageSpec &spec = inp->spec() ;
  int xres = spec.width ;
  int yres = spec.height ;

  std::cout << "spec.width: " << spec.width << std::endl ;
  assert ( spec.width * 6 == spec.height ) ;
  
  int nchannels = spec.nchannels ;
  // assert ( nchannels == 3 ) ;

  sixfold_t sf ( spec.width ) ;
  sf.load ( inp ) ;

  {
    auto sfi = ImageOutput::create ( "internal.exr" );
    assert ( sfi != nullptr ) ;
    ImageSpec ospec ( sf.store.shape[0] , sf.store.shape[1] ,
                      3 , TypeDesc::FLOAT ) ;
    sfi->open ( "internal.exr" , ospec ) ;

    sfi->write_image ( TypeDesc::FLOAT, sf.store.data() ) ;
    sfi->close();
  }

  convert_t act ( sf ) ;

  zimt::array_t < 2, v3_t > trg ( { 2 * height , height } ) ;

  // set up a linspace_t over the theta/phi sample points as get_t

  typedef zimt::xel_t < float , 2 > delta_t ;

  double d = 2.0 * M_PI / trg.shape[0] ;

  delta_t start { M_PI - d / 2.0 , M_PI_2 - d / 2.0 } ;
  delta_t step { -d , -d } ;

  zimt::bill_t bill ;
  bill.njobs = 1 ;
  
  zimt::linspace_t < float , 2 , 2 , 16 > linspace
    ( start , step , bill ) ;

  // set up a zimt::storer to trg

  zimt::storer < float , 3 , 2 , 16 > st ( trg , bill ) ;

  // call zimt::process
  
  zimt::process ( trg.shape , linspace , act , st , bill ) ;

  // finally we store the data to an image file - note how we have
  // float data in 'trg', and OIIO will convert these on-the-fly to
  // UINT16, as specified in the write_image invocation.
  // Note that the target will receive image data in the same colour
  // space as the input. If you feed, e.g. openEXR, and store to JPEG,
  // the image will look too dark, because the linear RGB data are
  // stored as if they were sRGB.

  auto out = ImageOutput::create ( latlon );
  assert ( out != nullptr ) ;
  ImageSpec ospec ( trg.shape[0] , trg.shape[1] , 3 , TypeDesc::FLOAT ) ;
  out->open ( latlon , ospec ) ;

  out->write_image ( TypeDesc::FLOAT , trg.data() ) ;
  out->close();
}
