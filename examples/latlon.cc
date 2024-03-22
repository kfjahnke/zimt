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
// The program is evolving from a simple 'proof-of-geometry' to a state
// which is already quite sophisticated: An internal representation
// holding the cube face data, plus 'support' area generated from the
// cube faces and surrounding them in the internal representation, is
// set up. It can do the 'pick-up' for arbitrary lat/lon or 3D 'ray'
// coordinates from a single image (rather than six discrete images
// of which the correct one has to be chosen for each pick-up).
// The last thing which I haven't done yet is to use OIIO's texture
// lookup on the internal representation, which is now feasible because
// the IR's geometry supports mip-mapping - for now the process uses
// bilinear interpolation only.

#include <array>

#include <zimt/zimt.h>

#include <Imath/ImathVec.h>
#include <Imath/ImathEuler.h>
#include <Imath/ImathQuat.h>

#include <OpenImageIO/texture.h>

using namespace OIIO ;

typedef zimt::xel_t < float , 2 > v2_t ;
typedef zimt::xel_t < int , 2 > v2i_t ;
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
    std::cout << "face_width: " << face_width << std::endl ;
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

  // will use the next class, implementation below

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
: public zimt::unary_functor < v2_t , v3_t , 16 >
{
  // source of pixel data

  const sixfold_t & cubemap ;
  const int degree ;

  // some SIMDized types we'll use

  typedef typename zimt::simdized_type < float , 16 > f_v ;
  typedef typename zimt::simdized_type < v2_t , 16 > crd2_v ;
  typedef typename zimt::simdized_type < v3_t , 16 > crd3_v ;
  typedef typename zimt::simdized_type < v3_t , 16 > px_v ;

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

    typedef typename zimt::simdized_type < float , 16 > f_v ;

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
    // ahead and increases both with increasing latitude. For latitudes
    // above the equator, we'll see negative values, and positive values
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
    // component has the largest absolut value (the 'dominant' one,
    // along the 'dominant' axis

    auto m1 = ( abs ( c[RIGHT] ) >= abs ( c[DOWN] ) ) ;
    auto m2 = ( abs ( c[RIGHT] ) >= abs ( c[FORWARD] ) ) ;
    auto m3 = ( abs ( c[DOWN] ) >= abs ( c[FORWARD] ) ) ;

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
    // is larger than the other two, dom0 will be true.

    auto dom0 = m1 & m2 ;

    // dom1 is true where the y axis (pointing down) has the largest
    // numerical value, and dom2 pertains to the za axis (forward)

    auto dom1 = ( ! m1 ) & m3 ; 
    auto dom2 = ( ! m2 ) & ( ! m3 ) ;

    // Now we can assign face indexes, stored in f. If the coordinate
    // value is negative along the dominant axis, we're looking at
    // the face opposite and assign a face value one higher.

    face = CM_RIGHT ;
    face ( dom0 & ( c[RIGHT] < 0 ) ) = CM_LEFT ;
    face ( dom1 ) = CM_BOTTOM ;
    face ( dom1 & ( c[DOWN] < 0 ) ) = CM_TOP ;
    face ( dom2 ) = CM_FRONT ;
    face ( dom2 & ( c[FORWARD] < 0 ) ) = CM_BACK ;

    // find the in-face coordinates, start for dom0 (x axis)
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

    // since we have - expensive - divisions here, we might
    // first test whether the dom mask is populated at all.
    // an alternative to the division is to use a multiplication
    // with the reciprocal value of the absolute value followed
    // by a multiplication with the sign of c[i] for those ops
    // which don't use the absolute values. This may be faster,
    // saving three divisions, but the code is clearer like this.

    // extract in-face coordinates for the right and left cube
    // face. the derivation of the x coordinate uses opposites for
    // the two faces, the direction of the y coordinate is equal

    in_face[0] ( dom0 ) = - c[FORWARD] / c[RIGHT] ;
    in_face[1] ( dom0 ) = c[DOWN] / abs ( c[RIGHT] ) ;

    // same for the top and bottom cube faces - here the x coordinate
    // corresponds to the right 3D axis, the y coordinate depends on
    // which of the faces we're looking at (hence no abs)
    // the top and bottom images could each be oriented in four
    // different ways. The orientation I expect in this program
    // is openEXR's cubemap format, where the top and bottom
    // image align with the 'back' image (the last one down).
    // For lux conventions (the top and bottom image aligning
    // with the front cube face) swap the signs in both expressions.

    // lux convention:
    // in_face[0] ( dom1 ) =   c[RIGHT] / abs ( c[DOWN] ) ;
    // in_face[1] ( dom1 ) = - c[FORWARD] / c[DOWN] ;

    in_face[0] ( dom1 ) = - c[RIGHT] / abs ( c[DOWN] ) ;
    in_face[1] ( dom1 ) =   c[FORWARD] / c[DOWN] ;

    // finally the front and back faces

    in_face[0] ( dom2 ) = c[RIGHT] / c[FORWARD] ;
    in_face[1] ( dom2 ) = c[DOWN] / abs ( c[FORWARD] ) ;
  }

  // given the cube face and the in-face coordinate, extract the
  // corresponding pixel values from the internal representation
  // of the cubemap held in the sixfold_t object

  void cubemap_to_pixel ( const f_v & face ,
                          crd2_v in_face ,
                          px_v & px ,
                          const int & degree ) const
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

    if ( degree == 0 )
    {
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
      typedef typename zimt::simdized_type < v2_t , 16 > crd2_v ;
      typedef typename zimt::simdized_type < v3_t , 16 > px_v ;

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

    cubemap_to_pixel ( face , in_face , px , degree ) ;
  }

} ;

struct fill_frame_t
: public zimt::unary_functor < v2i_t , v3_t , 16 >
{
  const convert_t & convert ;
  const sixfold_t & sf ;
  const int face ;
  const int degree ;

  // some SIMDized types we'll use

  typedef typename zimt::simdized_type < float , 16 > f_v ;
  typedef typename zimt::simdized_type < v2_t , 16 > crd2_v ;
  typedef typename zimt::simdized_type < v2i_t , 16 > v2i_v ;
  typedef typename zimt::simdized_type < v3_t , 16 > crd3_v ;
  typedef typename zimt::simdized_type < v3_t , 16 > px_v ;

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

    // move from image coordinates to ray coordinates. Note how we
    // have incoming coordinates from outside the cube face proper,
    // pertaining to the 'store' array.
    // add the component for the third axis


    // std::cout << "crd2 " << crd2 << std::endl ;
    // std::cout << "sf.frame " << sf.frame << std::endl ;
    // std::cout << "sf.face_width " << sf.face_width << std::endl ;
    // std::cout << "crd3 " << crd3 << std::endl ;
    // find the cube face and in-face coordinate for 'c'

    f_v fv ;
    crd2_v in_face ;
    convert.ray_to_cubeface ( crd3 , fv , in_face ) ;

    // std::cout << "face " << face << std::endl ;
    // std::cout << "in_face " << in_face << std::endl ;
    // use this information to obtain pixel values

    convert.cubemap_to_pixel ( fv , in_face , px , degree ) ;
    // std::cout << "px " << px << std::endl ;
  }
} ;

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
    // auto to_lower = ( frame + face_width ) * store.strides[1] ;
    // auto to_right = ( frame + face_width ) * store.strides[0] ;
    
    zimt::bill_t bill ;
    
    zimt::view_t < 2 , v3_t >
      framed ( p_frame , store.strides ,
               { outer_width , outer_width } ) ;

    bill.lower_limit = { 0 ,           0 } ;
    bill.upper_limit = { long(outer_width) , long(frame) } ;
    zimt::transform ( fill_frame , framed , bill  ) ;

    bill.lower_limit = { 0 ,           long(frame + face_width) } ;
    bill.upper_limit = { long(outer_width) , long(outer_width) } ;
    zimt::transform ( fill_frame , framed , bill  ) ;

    bill.lower_limit = { 0 ,     0 } ;
    bill.upper_limit = { long(frame) , long(outer_width) } ;
    zimt::transform ( fill_frame , framed , bill  ) ;

    bill.lower_limit = { long(frame + face_width) , 0 } ;
    bill.upper_limit = { long(outer_width) , long(outer_width) } ;
    zimt::transform ( fill_frame , framed , bill  ) ;
  }
}

void sixfold_t::mirror_around()
{
  auto * p_base = store.data() ;

  for ( int face = 0 ; face < 6 ; face++ )
  {
    // get a pointer to the upper left of the cube face 'proper'

    auto * p_frame = p_base + face * outer_width * store.strides[1]
                            + frame * store.strides[1] + frame ;
    zimt::view_t < 2 , v3_t > cubeface
      ( p_frame , store.strides , { face_width , face_width } ) ;

    typedef zimt::xel_t < long , 2 > ix_t ;
    // v3_t red { 255.0f , 0.0f , 0.0f } ;
    // v3_t green { 0.0f , 255.0f , 0.0f } ;

    for ( long x = -1L ; x <= long ( face_width ) ; x++ )
    {
      ix_t src { x ,  0L } ;
      ix_t trg { x , -1L } ;
      cubeface [ trg ] = cubeface [ src ] ;
      // std::cout << src << " -> " << trg << std::endl ;
      src [ 1 ] = face_width - 1L ;
      trg [ 1 ] = face_width ;
      cubeface [ trg ] = cubeface [ src ] ;
      // std::cout << src << " -> " << trg << std::endl ;
    }

    for ( long y = -1L ; y <= long ( face_width ) ; y++ )
    {
      ix_t src {  0L , y } ;
      ix_t trg { -1L , y } ;
      cubeface [ trg ] = cubeface [ src ] ;
      // std::cout << src << " -> " << trg << std::endl ;
      src [ 0 ] = face_width - 1L ;
      trg [ 0 ] = face_width ;
      cubeface [ trg ] = cubeface [ src ] ;
      // std::cout << src << " -> " << trg << std::endl ;
    }
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

  const ImageSpec &spec = inp->spec() ;
  int xres = spec.width ;
  int yres = spec.height ;

  std::cout << "spec.width: " << spec.width << std::endl ;
  assert ( spec.width * 6 == spec.height ) ;
  
  int nchannels = spec.nchannels ;
  // assert ( nchannels == 3 ) ;

  sixfold_t sf ( spec.width ) ;

  // load the cube faces into slots in the shared array

  sf.load ( inp ) ;

  // now we build up the 'support'. The square images for the
  // cube faces are cut off at the ninety-degree point, and to
  // interpolate in the vicinity of the cube faces' margin, we
  // need some support in good quality. Further out from the
  // margins, the quality of the support dta isn't so critical -
  // these pixels only come to play when the cube face is
  // mip-mapped. But going over these pixels again and filling
  // them in with values from bilinear interpolation over the
  // data in the neighbouring cube faces does no harm - we might
  // reduce the target area to some thinner stripe near the
  // edge, but since this is only working over a small-ish part
  // of the data, for now, we'll just redo the lot.

  // we start out mirroring out the square's 1-pixel edge

  sf.mirror_around() ;

  // now we do a nerarest-neighbour pickup into the square,
  // fetching data from neighbouring squares

  sf.fill_support ( 0 ) ;

  // to refine the result, we repeat te process several times
  // with bilinear interpolation. This might be done in a
  // thinner frame around the cube face proper, since the
  // data further out are good enough for their purpose
  // (namely, mip-mapping)

  sf.fill_support ( 1 ) ;
  sf.fill_support ( 1 ) ;
  sf.fill_support ( 1 ) ;

  // to have a look at the internal representation with properly
  // set up support area, uncomment this bit:

  // {
  //   auto sfi = ImageOutput::create ( "internal.exr" );
  //   assert ( sfi != nullptr ) ;
  //   ImageSpec ospec ( sf.store.shape[0] , sf.store.shape[1] ,
  //                     3 , TypeDesc::HALF ) ;
  //   sfi->open ( "internal.exr" , ospec ) ;
  // 
  //   sfi->write_image ( TypeDesc::FLOAT, sf.store.data() ) ;
  //   sfi->close();
  // }

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
  // area from outher cube faces.


  convert_t act ( sf , 1 ) ; // convert using bilinear interpolation

  zimt::array_t < 2, v3_t > trg ( { 2 * height , height } ) ;

  // set up a linspace_t over the theta/phi sample points as get_t

  typedef zimt::xel_t < float , 2 > delta_t ;

  double d = 2.0 * M_PI / trg.shape[0] ;

  delta_t start { - M_PI + d / 2.0 , - M_PI_2 + d / 2.0 } ;
  delta_t step { d , d } ;

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
