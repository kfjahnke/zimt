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

// utility program producing a lat/lon 'environment map' from a cubemap.

// As an example program for zimt, this program is maybe too complex
// and not entirely on the topic, but it does demonstrate the use
// of a good range of zimt features in a 'real' program.
// AFAICT there are two formats used to represent a complete 360X180
// degree environment. The more common one is a lat/lon environment,
// which captures the environment in a single image in spherical
// projection. Using this format is quite straightforward, but it
// requires using transcendental functions to move between the
// lat/lon spherical coordinates and 3D 'ray' geometry. The second
// format, which - I have been told - is less common, is the
// 'cubemap' or 'skybox' format. It captures the environment in
// six square images representing the faces of a virtual cube
// surrounding the origin. For viewing purposes, this format has
// some advantages and some disadvantages. On the plus side is the
// fact that, since the cube faces are stored in rectilinear projection,
// reprojection to a rectilinear view can be done without transcendental
// functions. On the negative side, handling the six discrete images,
// which requires picking the right one to pick up image information
// to go to a specific target location in the output, requires a fair
// amount of logic, and the sequence and (for the top and bottom square)
// orientation of the cube faces isn't obvious and leaves room for error.
// There are standards, though - openEXR defines a cubemap format with
// specific sequence and orientations, and this is the format I use
// in this program. openEXR offers a tool to convert between the two
// types of environment, but I found that it doesn't seem to work
// correctly - see this issue:
// https://github.com/AcademySoftwareFoundation/openexr/issues/1675
// Since I am processing both types of environment representation in
// lux, I have a special interest in them and writing some image
// processing code in zimt+OpenImageIO offers a good opportunity to
// deepen my understanding and come up with efficient ways of dealing
// with both formats. I started out with 'cubemap.cc', which converts
// lat/lon format to openEXR-compatible cubemap format. To pick up
// data from a lat/lon environment map, there is ready-made code in
// OpenImageIO. I use this code to good effect, producing output
// with a very 'proper' anisotropic anti-aliasing filter. The
// reverse transformation - from a cubemap to lat/lon format - is
// what I'm coding in this program. Here, employing OpenImageIO for
// the task of picking up data from the environment is not available
// out-of-the-box - OpenImageIO does not support this format. This
// may well be because it is less used and harder to handle. In my
// opinion, the greatest problem with this format is the fact that the
// six cube face images each cover precisely ninety degrees. This is
// sufficient to regenerate the environment, but it's awkward, due
// to the fact that near the edges there is not enough correct
// support in the individual images to use good interpolators.
// Instead, to use such interpolators, this support has to be
// gleaned from adjoining cube faces, with proper reprojection.
// The second stumbling stone - when trying to use such cubemaps
// with OpenImageIO - is the fact that to use mip-mapping on them
// needs even larger support around the cube faces (unless they happen
// to have a size which is a multiple of the tile size) so as to
// avoid mixing data from cube faces which are located next to each
// other in the 1:6 stripe but have no other relation - there is
// a hard discontiuity from one image's bottom to the next image's
// top. The third stumbling stone is the fact that OpenImageIO's
// texture system code is file-based, and the data I produce from
// the 'raw' cubemap input to deal with the first two issues can't
// simply be fed back to OIIO's texture system, because they are
// in memory, whereas the texture system wants them on disk. I'd
// like to find a way to feed them directly, but for now I think
// I may use an intermediate image on disk for the task. As it
// stands, this program doesn't yet use OIIO's texture system but
// relies on bilinear interpolation, which is adequate for the
// format conversion as long as the resolution of input and output
// are roughly the same. So there is some work waiting to be done.
//
// I have mentioned that I have dealt with the two first issues,
// so I'll give a quick outline here - the code is amply commented
// to explain the details. I start out by setting up a single array
// of pixel data which has enough 'headroom' to accomodate the cube
// faces plus a surrounding frame of support pixels which is large
// enough to allow for good interpolators and mip-mapping, so each
// cube face is embedded in a square section of the array which has
// a multiple of the tile size as it's extent. I proceed to import
// the cube face data, which I surround initially with a single-pixel
// frame of mirrored data to give enough support for bilinear
// interpolation even near the edges. Then I fill in the support frames
// using bilinear interpolation. The array is now filled with six square
// images which have more than ninety degrees field of view - the part
// beyond ninety degrees generated from adjoining cube faces. This is
// the texture from which I pick output data. Due to the ample support,
// it could be subjected to filters with large-ish support, and the
// array of six 'widened' cube faces might even stand as a useful
// format by itself, which would be quite easy to describe formally
// because it's a derivative of the openEXR cubemap format.
// I mentioned above that I haven't found a way to directly use
// OpenImageIO's texture system code on the data in RAM, so for the
// time being, I pick output data with bilinear interpolation to
// populate the target image in lat/lon format. Having coded the
// process to this point gives me a 'decent' quality and I can now
// ascertain that my code is geometrically correct, and I can also
// test how much image degradation I get with repeated conversions
// from one environment format to the other, using 'cubemap.cc'
// for the reverse process.
//
// There's an issue: currently some of the compilations with g++ fail.

#include <array>
#include <zimt/zimt.h>
#include <OpenImageIO/texture.h>

using namespace OIIO ;

// zimt types for 2D and 3D coordinates and pixels

typedef zimt::xel_t < int , 2 > v2i_t ;
typedef zimt::xel_t < long , 2 > index_type ;

typedef zimt::xel_t < float , 2 > v2_t ;
typedef zimt::xel_t < float , 3 > v3_t ;

// some SIMDized types we'll use. I use 16 SIMD lanes for now.

#define LANES 16

typedef zimt::simdized_type < float , LANES > f_v ;
typedef zimt::simdized_type < v2_t ,  LANES > crd2_v ;
typedef zimt::simdized_type < v2i_t , LANES > v2i_v ;
typedef zimt::simdized_type < v3_t ,  LANES > crd3_v ;
typedef zimt::simdized_type < v3_t ,  LANES > px_v ;
typedef zimt::simdized_type < index_type , LANES > index_v ;

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
// of openEXR environment maps. cubemaps on openEXR contain six
// square images concatenated vertically. The sequence of the
// images, from top to bottom, is: left, right, top, bottom,
// front, back. The top and bottom images are oriented so that
// they align vertically with the 'back' image (lux aligns with
// the front image).
// The sixfold_t object also combines the six images in one
// array, but it adds a frame of additional 'support' pixels
// around each square image to allow for interpolation with
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
// simple bilinear interpolation to get pixel values.

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
    std::cout << "face_width: " << face_width << std::endl ;
    std::cout << "outer_width: " << outer_width << std::endl ;

    frame = ( outer_width - face_width ) / 2 ;
    std::cout << "frame: " << frame << std::endl ;

    // find the location of the first cube face's upper left corner
    // in the 'store' array

    v3_t * p = store.data() ;
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

    std::ptrdiff_t offset = outer_width * outer_width ;

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
    // contains more.

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

  // these two will use the next class, implementation below

  void mirror_around() ;
  void fill_support ( int degree ) ;
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

enum { RIGHT , DOWN , FORWARD } ;

struct convert_t
: public zimt::unary_functor < v2_t , v3_t , LANES >
{
  // source of pixel data

  const sixfold_t & cubemap ;
  const int degree ;

  // convert_t's c'tor obtains a const reference to the sixfold_t
  // object holding pixel data

  convert_t ( const sixfold_t & _cubemap ,
              const int & _degree )
  : cubemap ( _cubemap ) ,
    degree ( _degree )
  { }

  // code from lux to convert lat/lon to 3D ray. Note that I am using
  // lux coordinate convention (book order: x is right, y down and
  // z forward). I use a template here, the code is good for scalar
  // values and SIMD data alike.

  template < typename in_type , typename out_type >
  void ll_to_ray ( const in_type & in ,
                   out_type & out ) const
  {
    // incoming, we have lon/lat coordinates - we use the convention
    // of axis order x - y - z, hence

    auto const & lon ( in[0] ) ;
    auto const & lat ( in[1] ) ;

    // outgoing, we have a directional vector.

    auto & right ( out[RIGHT] ) ;
    auto & down ( out[DOWN] ) ;
    auto & forward ( out[FORWARD] ) ;

    // we measure angles so that a view to directly ahead (0,0,1)
    // corresponds to latitude and longitude zero. longitude increases
    // into positive values when the view moves toward the right and
    // latitude increases into positive values when the view moves
    // downwards from the view straight ahead.
    // The code benefits from using sincos, where available.

#if defined USE_HWY or defined USE_VC

    f_v sinlat , coslat , sinlon , coslon ;
    sincos ( lat , sinlat , coslat ) ;
    sincos ( lon , sinlon , coslon ) ;

#else

     f_v sinlat = sin ( lat ) ;
     f_v coslat = cos ( lat ) ;
     f_v sinlon = sin ( lon ) ;
     f_v coslon = cos ( lon ) ;

#endif

    // the x component, pointing to the right in lux, is zero at
    // longitude zero, which is affected by the sine term. The
    // cosine term affects a scaling of this 'raw' value which
    // is one for latitude zero and decreases both ways.

    right = sinlon * coslat ;

    // The z component, pointing forward, is one at longitude and
    // latitude zero, and decreases with both increasing and decreasing
    // longitude and latitude.

    forward = coslon * coslat ;

    // The y component, pointing down, is zero for the view straight
    // ahead and increases with the latitude. For latitudes above
    // the equator, we'll see negative values, and positive values
    // for views into the 'southern hemisphere'. This component is not
    // affected by the longitude.

    down = sinlat ;
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
    // component has the largest absolute value (the 'dominant' one,
    // along the 'dominant' axis)

    auto m1 = ( abs ( c[RIGHT] ) >= abs ( c[DOWN] ) ) ;
    auto m2 = ( abs ( c[RIGHT] ) >= abs ( c[FORWARD] ) ) ;
    auto m3 = ( abs ( c[DOWN] )  >= abs ( c[FORWARD] ) ) ;

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

    // where the numerical value along the x axis (pointing right)
    // is larger than the other two, 'dom' will be true.

    auto dom = m1 & m2 ;

    // Now we can assign face indexes, stored in f. If the coordinate
    // value is negative along the dominant axis, we're looking at
    // the face opposite and assign a face value one higher. Note
    // how this SIMD code does the job for LANES coordinates in
    // parallel, avoiding conditionals and using masking instead.
    // we also find in-face coordinates:
    // we divide the two non-dominant coordinate values by the
    // dominant one. One of the axes comes out just right when
    // dividing by the absolute value - e.g. the vertical axis
    // points downwards for all the four cube faces around
    // the center. The other axis is divided by the 'major'
    // coordinate value as-is; the resulting coordinate runs
    // one way for positive major values and backwards for
    // negative ones. Note that we might capture the absolute
    // values (which we've used before) in variables, but the
    // compiler will recognize the common subexpressions and
    // do it for us.

    if ( any_of ( dom ) )
    {
      // extract in-face coordinates for the right and left cube
      // face. the derivation of the x coordinate uses opposites for
      // the two faces, the direction of the y coordinate is equal

      face = CM_RIGHT ;
      face ( dom & ( c[RIGHT] < 0 ) ) = CM_LEFT ;

      in_face[0] ( dom ) = - c[FORWARD] / c[RIGHT] ;
      in_face[1] ( dom ) = c[DOWN] / abs ( c[RIGHT] ) ;
    }

    // now set dom true where the y axis (pointing down) has the
    // largest numerical value

    dom = ( ! m1 ) & m3 ; 

    if ( any_of ( dom ) )
    {
      // same for the top and bottom cube faces - here the x coordinate
      // corresponds to the right 3D axis, the y coordinate depends on
      // which of the faces we're looking at (hence no abs)
      // the top and bottom images could each be oriented in four
      // different ways. The orientation I expect in this program
      // is openEXR's cubemap format, where the top and bottom
      // image align with the 'back' image (the last one down).
      // For lux conventions (the top and bottom image aligning
      // with the front cube face) swap the signs in both expressions.

      face ( dom ) = CM_BOTTOM ;
      face ( dom & ( c[DOWN] < 0 ) ) = CM_TOP ;

      // lux convention:
      // in_face[0] ( dom ) =   c[RIGHT] / abs ( c[DOWN] ) ;
      // in_face[1] ( dom ) = - c[FORWARD] / c[DOWN] ;

      in_face[0] ( dom ) = - c[RIGHT] / abs ( c[DOWN] ) ;
      in_face[1] ( dom ) =   c[FORWARD] / c[DOWN] ;

    }

    // set dom true where the z axis (pointing forward) has the
    // largest numerical value
    
    dom = ( ! m2 ) & ( ! m3 ) ;

    if ( any_of ( dom ) )
    {
      // finally the front and back faces

      face ( dom ) = CM_FRONT ;
      face ( dom & ( c[FORWARD] < 0 ) ) = CM_BACK ;

      in_face[0] ( dom ) = c[RIGHT] / c[FORWARD] ;
      in_face[1] ( dom ) = c[DOWN] / abs ( c[FORWARD] ) ;
    }
  }

  // given the cube face and the in-face coordinate, extract the
  // corresponding pixel values from the internal representation
  // of the cubemap held in the sixfold_t object

  void cubemap_to_pixel ( const f_v & face ,
                          crd2_v in_face ,
                          px_v & px ,
                          const int & degree ) const
  {
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

    // now do the texture lookup. an incoming angle of precisely
    // zero degrees must map to -0.5, the left edge of the leftmost
    // pixel, hence:

    in_face -= .5f ;

    if ( degree == 0 )
    {
      // simple nearest-neighbour lookup. This is not currently used.

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
    else if ( degree == 1 )
    {
      const auto * p = (float*) ( cubemap.store.data() ) ;
      px_v px2 ,help ;

      index_v low { in_face[0] , in_face[1] } ;
      auto diff = in_face - crd2_v ( low ) ;

      index_v idx { low[0] , low[1] } ;
      auto ofs = ( idx * cubemap.store.strides ) . sum() * 3 ;
      px.gather ( p , ofs ) ;
      px *= ( 1.0f - diff[0] ) ;

      idx[0] += 1 ; ;
      ofs = ( idx * cubemap.store.strides ) . sum() * 3 ;
      help.gather ( p , ofs ) ;
      px += help * diff[0] ;

      px *= ( 1.0f - diff[1] ) ;

      idx[0] -= 1 ; ;
      idx[1] += 1 ; ;
      ofs = ( idx * cubemap.store.strides ) . sum() * 3 ;
      px2.gather ( p , ofs ) ;
      px2 *= ( 1.0f - diff[0] ) ;

      idx[0] += 1 ; ;
      ofs = ( idx * cubemap.store.strides ) . sum() * 3 ;
      help.gather ( p , ofs ) ;
      px2 += help * diff[0] ;

      px += px2 * diff[1] ;
    }
  }

  // 'eval' function which will be called by zimt::process.
  // We might just put all the code from the three functions above
  // together, but I prefer to have three distinct steps, which may
  // come in handy later on when the code is put to use beyond the
  // immediate aim to code the cubemap-to-lat/lon conversion

  // incoming: 2D coordinate as lat/lon in radians. Internally, we
  // first calculate a 3D directional vector, then the index of the
  // cube face and then the 2D in-face coordinate, where we pick up
  // the pixel value from the cube face image. We form 2D pick-up
  // coordinates into the array held in the sixfold_t object, which
  // contains the images 'embedded' in additional support space,
  // which offers 'headroom' for interpolators which need support
  // and also provides each cube face image with just so much frame
  // that the sixth of the total array it inhabits can be tiled
  // exactly with the given tile size. This is to enable correct
  // mip-mapping for filtered texture lookup.

  void eval ( const crd2_v & lat_lon , px_v & px )
  {
    // convert 2D spherical coordinate to 3D ray coordinate 'crd3'
  
    crd3_v crd3 ;
    ll_to_ray ( lat_lon , crd3 ) ;

    // find the cube face and in-face coordinate for 'c'

    f_v face ;
    crd2_v in_face ;
    ray_to_cubeface ( crd3 , face , in_face ) ;

    // use this information to obtain pixel values

    cubemap_to_pixel ( face , in_face , px , degree ) ;
  }

} ;

// this functor is used to fill the frame of support pixels in the
// array in the sixfold_t object. incoming, we have 2D image coordinates,
// which, for the purpose at hand, will lie outside the cube face.
// But we can still convert these image coordinates to planar
// coordinates in 'model space' and then further to 'pickup'
// coordinates into the array in the sixfold_t object.

struct fill_frame_t
: public zimt::unary_functor < v2i_t , v3_t , LANES >
{
  const convert_t & convert ;
  const sixfold_t & sf ;
  const int face ;
  const int degree ;

  fill_frame_t ( const convert_t & _convert ,
                 const int & _face ,
                 const int & _degree )
  : convert ( _convert ) ,
    sf ( _convert.cubemap ) ,
    face ( _face ) ,
    degree ( _degree )
  { }

  void eval ( const v2i_v & crd2 , px_v & px )
  {
    float third = 1.0f ;
    crd3_v crd3 ;
    auto shift = sf.frame + ( sf.face_width - 1.0f ) / 2.0f ;
    auto scale = 1.0f / float ( sf.face_width / 2.0f ) ;

    // since 'face' is const, this case switch should be optimized away.
    // here, we move from discrete image coordinates to 3D coordinates
    // scaled to 'model space', on a plane one unit from the origin.

    switch ( face )
    {
      case CM_FRONT :
        crd3[RIGHT]   =   ( crd2[RIGHT] - shift ) * scale ;
        crd3[DOWN]    =   ( crd2[DOWN]  - shift ) * scale ;
        crd3[FORWARD] =   1.0f ;
        break ;
      case CM_BACK :
        crd3[RIGHT]   = - ( crd2[RIGHT] - shift ) * scale ;
        crd3[DOWN]    =   ( crd2[DOWN]  - shift ) * scale ;
        crd3[FORWARD] = - 1.0f ;
        break ;
      case CM_RIGHT :
        crd3[RIGHT] =     1.0f ;
        crd3[DOWN] =      ( crd2[DOWN]  - shift ) * scale ;
        crd3[FORWARD] = - ( crd2[RIGHT] - shift ) * scale ;
        break ;
      case CM_LEFT :
        crd3[RIGHT] =   - 1.0f ;
        crd3[DOWN] =      ( crd2[DOWN]  - shift ) * scale ;
        crd3[FORWARD] =   ( crd2[RIGHT] - shift ) * scale ;
        break ;
      // for bottom and top, note that we're using openEXR convention.
      // to use lux convention, invert the signs.
      case CM_BOTTOM :
        crd3[RIGHT] =   - ( crd2[RIGHT] - shift ) * scale ;
        crd3[DOWN] =      1.0f ;
        crd3[FORWARD] =   ( crd2[DOWN]  - shift ) * scale ;
        break ;
      case CM_TOP :
        crd3[RIGHT] =   - ( crd2[RIGHT] - shift ) * scale ;
        crd3[DOWN] =    - 1.0f ;
        crd3[FORWARD] = - ( crd2[DOWN]  - shift ) * scale ;
        break ;
     } ;

    // next we use the 3D coordinates we have just obtained to
    // find a cube face and in-face coordinates - this will
    // be a different cube face to 'face', and it will be one
    // which actually can provide data for location we're filling
    // in. So we re-project the content from adjoining cube faces
    // into the support area around the cube face we're currently
    // surrounding with a support frame.

    f_v fv ;
    crd2_v in_face ;
    convert.ray_to_cubeface ( crd3 , fv , in_face ) ;

    // finally we use this information to obtain pixel values,
    // which are written to the target location.

    convert.cubemap_to_pixel ( fv , in_face , px , degree ) ;
  }
} ;

// After the cube faces have been read from disk, they are surrounded
// by black (or even undefined) pixels. We want to provide minimal
// support, the support's quality is not crucial, but it should not
// be black, but rather like mirroring on the edge, which this function
// does - it produces a one-pixel-wide frame with mirrored pixels
// aound each of the cube faces.

void sixfold_t::mirror_around()
{
  auto * p_base = store.data() ;

  for ( int face = 0 ; face < 6 ; face++ )
  {
    // get a pointer to the upper left of the cube face 'proper'

    auto * p_frame = p_base + face * outer_width * store.strides[1]
                            + frame * store.strides[1]
                            + frame ;

    // get a zimt view to the current cube face

    zimt::view_t < 2 , v3_t > cubeface
      ( p_frame , store.strides , { face_width , face_width } ) ;

    // we use 2D discrete coordinates

    typedef zimt::xel_t < long , 2 > ix_t ;

    // mirror the horizontal edge

    for ( long x = -1L ; x <= long ( face_width ) ; x++ )
    {
      ix_t src { x ,  0L } ;
      ix_t trg { x , -1L } ;
      cubeface [ trg ] = cubeface [ src ] ;
      src [ 1 ] = face_width - 1L ;
      trg [ 1 ] = face_width ;
      cubeface [ trg ] = cubeface [ src ] ;
    }

    // and the vertical edge

    for ( long y = -1L ; y <= long ( face_width ) ; y++ )
    {
      ix_t src {  0L , y } ;
      ix_t trg { -1L , y } ;
      cubeface [ trg ] = cubeface [ src ] ;
      src [ 0 ] = face_width - 1L ;
      trg [ 0 ] = face_width ;
      cubeface [ trg ] = cubeface [ src ] ;
    }
  }
}

// fill_support uses the fill_frame_t functor to populate the
// frame of support. The structure of the code is similar to the
// previous function, iterating over the six sections of the
// array and manipulating each in turn. But here we fill in
// the entire surrounding frame, not just a pixel-wide line.

void sixfold_t::fill_support ( int degree )
{
  auto * p_base = store.data() ;

  for ( int face = 0 ; face < 6 ; face++ )
  {
    convert_t cvt ( *this , degree ) ;
    fill_frame_t fill_frame ( cvt , face , degree ) ;
  
    // get a pointer to the upper left of the cube face plus it's
    // surrounding frame

    auto * p_frame = p_base + face * outer_width * store.strides[1] ;
    
    // we form a view to the current section of the array in the
    // sixfold_t object

    zimt::view_t < 2 , v3_t >
      framed ( p_frame , store.strides ,
               { outer_width , outer_width } ) ;

    // we'll use a 'loading bill' to narrow the filling-in down
    // to the areas which are outside the cube face.

    zimt::bill_t bill ;
    
    // now we fill in the lower an upper limits. This is a good
    // demonstration of how these parameters can be put to use.
    // We use a 'notional' shape which encompasses the entire
    // section, but the iteration will only visit those coordinates
    // which are in the range given by the limits. Note how we
    // use zimt::transform without a 'source' argument: this idiom
    // uses a 'get_crd' object to generate input, namely the
    // discrete coordinates which we iterate over.

    // fill in the stripe above the cube face

    bill.lower_limit = { 0 ,           0 } ;
    bill.upper_limit = { long(outer_width) , long(frame) } ;
    zimt::transform ( fill_frame , framed , bill  ) ;

    // fill in the stripe below the cube face

    bill.lower_limit = { 0 ,           long(frame + face_width) } ;
    bill.upper_limit = { long(outer_width) , long(outer_width) } ;
    zimt::transform ( fill_frame , framed , bill  ) ;

    // fill in the stripe to the left of the cube face

    bill.lower_limit = { 0 ,     0 } ;
    bill.upper_limit = { long(frame) , long(outer_width) } ;
    zimt::transform ( fill_frame , framed , bill  ) ;

    // fill in the stripe to the right of the cube face

    bill.lower_limit = { long(frame + face_width) , 0 } ;
    bill.upper_limit = { long(outer_width) , long(outer_width) } ;
    zimt::transform ( fill_frame , framed , bill  ) ;
  }
}

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

  // we expect an image with 1:6 aspect ratio. We don't check
  // for metadata specific to environments - it can be anything,
  // but the aspect ratio must be correct.

  const ImageSpec &spec = inp->spec() ;
  int xres = spec.width ;
  int yres = spec.height ;

  std::cout << "spec.width: " << spec.width << std::endl ;
  assert ( spec.width * 6 == spec.height ) ;

  // for now, we process three channels only, even if the input
  // has more

  int nchannels = spec.nchannels ;

  // we set up the sixfold_t object, 'preparing the ground'
  // to pull in the image data

  sixfold_t sf ( spec.width ) ;

  // load the cube faces into slots in the array in the
  // sixfold_t object

  sf.load ( inp ) ;

  // now we build up the 'support'. The square images for the
  // cube faces are cut off at the ninety-degree point, and to
  // interpolate in the vicinity of the cube faces' margin, we
  // need some support in good quality. Further out from the
  // margins, the quality of the support data isn't so critical -
  // these pixels only come to play when the cube face is
  // mip-mapped. But going over these pixels again and filling
  // them in with values from bilinear interpolation over the
  // data in the neighbouring cube faces does no harm - we might
  // reduce the target area to some thinner stripe near the
  // edge, but since this is only working over a small-ish part
  // of the data, for now, we'll just redo the lot a few times.

  // we start out mirroring out the square's 1-pixel edge

  sf.mirror_around() ;

  // to refine the result, we generate support twice
  // with bilinear interpolation. This might be done in a
  // thinner frame around the cube face proper, since the
  // data further out are good enough for their purpose
  // (namely, mip-mapping). Doing it twice may even be
  // overkill - I can't see much of a difference between
  // using one and two runs.

  sf.fill_support ( 1 ) ;
  sf.fill_support ( 1 ) ;

  // to have a look at the internal representation with properly
  // set up support area, uncomment this bit:

//   {
//     auto sfi = ImageOutput::create ( "internal3.exr" );
//     assert ( sfi != nullptr ) ;
//     ImageSpec ospec ( sf.store.shape[0] , sf.store.shape[1] ,
//                       3 , TypeDesc::HALF ) ;
//     sfi->open ( "internal3.exr" , ospec ) ;
//   
//     sfi->write_image ( TypeDesc::FLOAT, sf.store.data() ) ;
//     sfi->close();
//   }

  // with proper support near the edges, we can now run the
  // actual payload code - the conversion to lon/lat - with
  // bilinear interpolation, which is a step up from the
  // nearest-neighbour interpolation we had until now and
  // is good enough if the resolution of input and output
  // don't differ too much. But the substrate we have
  // generated is also good for mip-mapping, so we have the
  // option to switch to interpolation methods with larger
  // support, like OIIO's texture access with anisotropic
  // antaliasing filter. We'd need the derivatives for that,
  // though, which is an extra complication.
  // The internal representation - as it is now - is also
  // useful for other interpolation schemes which need
  // support - if the frame is wide enough, running a b-spline
  // prefilter is quite feasible, because the disturbances due
  // to the imperfections further out from the areas of interest
  // will be negligible.
  // Note also that with the conversion of an incoming 2D or 3D
  // coordinate to a single 2D coordinate pertaining to the
  // internal representation (rather than the pair of cube face
  // index and in-face coordinate) picking up the result is also
  // more efficient and may outperform the current method used
  // in lux, making cubemaps even better suited as environments.
  // It may be a good idea to store the entire internal
  // representation - cubefaces plus support - to an image file
  // for faster access, avoiding the production of the support
  // area from outher cube faces, at the cost of saving a few
  // extra pixels. The process as it stands now seems to produce
  // support which is just as good as support which is created
  // by rendering cube face images with slightly more than ninety
  // degrees (as it can be done in lux), so we can process the
  // 'orthodox' format and yet avoid it's shortcomings.

  // time to do the remaining work.

  // set up convert_t using bilinear interpolation (argument 1).
  // This is the functor which takes lon/lat coordinates and
  // produces pixel data from the internal representation of the
  // cubemap held in the sixfold_t object

  convert_t act ( sf , 1 ) ;

  // The target array will receive the output pixel data

  zimt::array_t < 2, v3_t > trg ( { 2 * height , height } ) ;

  // set up a linspace_t over the lon/lat sample points as get_t
  // (a.k.a input generator). d is the step width from one sample
  // to the next:

  double d = 2.0 * M_PI / trg.shape[0] ;

  v2_t start { - M_PI + d / 2.0 , - M_PI_2 + d / 2.0 } ;
  v2_t step { d , d } ;

  zimt::linspace_t < float , 2 , 2 , LANES > linspace ( start , step ) ;

  // set up a zimt::storer writing to to the target array

  zimt::storer < float , 3 , 2 , LANES > st ( trg ) ;

  // showtime! call zimt::process.
  
  zimt::process ( trg.shape , linspace , act , st ) ;

  // finally we store the data to an image file - note how we have
  // float data in 'trg', and OIIO will convert these on-the-fly to
  // HALF, as specified in the write_image invocation.
  // Note that the target will receive image data in the same colour
  // space as the input. If you feed, e.g. openEXR, and store to JPEG,
  // the image will look too dark, because the linear RGB data are
  // stored as if they were sRGB.

  auto out = ImageOutput::create ( latlon );
  assert ( out != nullptr ) ;
  ImageSpec ospec ( trg.shape[0] , trg.shape[1] , 3 , TypeDesc::HALF ) ;
  out->open ( latlon , ospec ) ;

  out->write_image ( TypeDesc::FLOAT , trg.data() ) ;
  out->close();
}
