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
// You can store this intermediate in a file called internal.exr
// by setting the global save_ir to true.
// I mentioned above that I haven't found a way to directly use
// OpenImageIO's texture system code on the data in RAM, so for the
// time being, I pick output data with bilinear interpolation to
// populate the target image in lat/lon format. Having coded the
// process to this point gives me a 'decent' quality and I can now
// ascertain that my code is geometrically correct, and I can also
// test how much image degradation I get with repeated conversions
// from one environment format to the other, using 'cubemap.cc'
// for the reverse process. The forward-backward conversion is
// 'quite hard on the data' - looking at the re-generated lat/lon
// image compared to the original lat/lon image, both artifacts
// due to filtering and due to interpolation are apparent. One way
// to improve the fidelity greatly is by rendering to a cubemap
// with twice the size as intermediate. Processing speed of this
// program is fast, probably mostly bound by I/O, but then it's
// using bilinear interpolation only. Mind you, the entire process
// is coded to use multithreaded SIMD code, so it does at least
// exploit the CPU resources properly. There is little difference
// in performance between the four SIMD back-ends and the two
// compilers I test regularly.

#include <array>
#include <zimt/zimt.h>
#include <OpenImageIO/texture.h>

using namespace OIIO ;

// zimt types for 2D and 3D coordinates and pixels

typedef zimt::xel_t < int , 2 > v2i_t ;
typedef zimt::xel_t < long , 2 > index_type ;
typedef zimt::xel_t < std::size_t , 2 > shape_type ;

typedef zimt::xel_t < float , 2 > v2_t ;
typedef zimt::xel_t < float , 3 > v3_t ;

// some SIMDized types we'll use. I use 16 SIMD lanes for now.

#define LANES 16

typedef zimt::simdized_type < float , LANES > f_v ;
typedef zimt::simdized_type < int , LANES > i_v ;
typedef zimt::simdized_type < v2_t ,  LANES > crd2_v ;
typedef zimt::simdized_type < v2i_t , LANES > v2i_v ;
typedef zimt::simdized_type < v3_t ,  LANES > crd3_v ;
typedef zimt::simdized_type < index_type , LANES > index_v ;

// enum encoding the sequence of cube face images in the cubemap
// This is the sequence used for openEXR cubmap format. The top
// and bottom squares are oriented so as to align with the back
// image. Of course, the labels are debatable: my understanding
// of 'front' is 'aligned with the image center'. If one were to
// associate 'front' with the wrap-around point of the full
// spherical, the labels would be different.

typedef enum
{
  CM_LEFT ,
  CM_RIGHT ,
  CM_TOP ,
  CM_BOTTOM ,
  CM_FRONT ,
  CM_BACK
} face_index_t ;

// we use lux coordinate system convention. I call it 'latin book
// order': if you have a stack of prints in front of you and read
// them, your eyes move first left to right inside the line, then
// top to bottom from line to line, then, moving to the next pages,
// forward in the stack. Using this order also makes the first two
// compnents agree with normal image indexing conventions, namely
// x is to the right and y down. Note that I put the fastest-moving
// index first, which is 'fortran' style, whereas C/C++ use the
// opposite order for nD arrays.

enum { RIGHT , DOWN , FORWARD } ;

// some globals, used to trigger output of intermediate or test
// images.

bool save_ir = false ;
bool regenerate = false ;

// helper function to save a zimt array to an image file. I am
// working with openEXR images hare, hence the HALF data type.
// If the output format can't produce HALF, it will use the
// 'next-best' thing. Note that input and output should agree
// on the colour space. If you stay within one format, that's
// not an issue.

template < std::size_t nchannels >
void save_array ( const std::string & filename ,
                  const zimt::view_t
                    < 2 ,
                      zimt::xel_t < float , nchannels >
                    > & pixels )
{
  auto out = ImageOutput::create ( filename );
  assert ( out != nullptr ) ;
  ImageSpec ospec ( pixels.shape[0] , pixels.shape[1] ,
                    nchannels , TypeDesc::HALF ) ;
  out->open ( filename , ospec ) ;

  auto success = out->write_image ( TypeDesc::FLOAT , pixels.data() ) ;
  assert ( success ) ;
  out->close();
}

// coordinate transformations, coded as templates in zimt 'act'
// functor style, returning the result via a reference argument

// code to convert lat/lon to 3D ray. Note that I am using lux
// coordinate convention ('book order': x is right, y down and
// z forward). I use a template here, the code is good for scalar
// values and SIMD data alike. The lat/lon values coming in are
// angles in radians, and the resulting 'ray' coordinates are
// in model space units. We're using an implicit radius of 1.0
// for the conversion from spherical to cartesian coordinates,
// so the output has unit length.

struct ll_to_ray_t
: public zimt::unary_functor < v2_t , v3_t , LANES >
{
  template < typename in_type , typename out_type >
  void eval ( const in_type & in ,
              out_type & out ) const
  {
    // incoming, we have lon/lat coordinates.

    auto const & lon ( in[0] ) ;
    auto const & lat ( in[1] ) ;

    // outgoing, we have a directional vector.

    auto & right   ( out [ RIGHT   ] ) ;
    auto & down    ( out [ DOWN    ] ) ;
    auto & forward ( out [ FORWARD ] ) ;

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
} ;

// code to move from 3D ray coordinates to lat/lon. This is the
// reverse operation to ll_to_ray above and follows the same
// conventions. Incoming, we have 3D ray coordinates, and
// outgoing, 2D lon/lat coordinates. Note how the use of atan2
// allow us to take rays in any scale.
// The output is in [-pi, pi], and it's zero for the view
// 'straight ahead' in lux convention, which coincides with the
// center of the full spherical image.

struct ray_to_ll_t
: public zimt::unary_functor < v3_t , v2_t , LANES >
{
  template < typename in_type , typename out_type >
  void eval ( const in_type & in ,
              out_type & out ) const
  {
    // incoming, we have a 3D directional vector
    
    auto const & right ( in[RIGHT] ) ;
    auto const & down ( in[DOWN] ) ;
    auto const & forward ( in[FORWARD] ) ;

    // outgoing, we have a 2D lat/lon coordinate.

    auto & lon ( out[0] ) ;
    auto & lat ( out[1] ) ;

    auto s = sqrt ( right * right + forward * forward ) ;
    lat = atan2 ( down , s ) ;
    lon = atan2 ( right , forward ) ;
  }
} ;

// this structure is used to calculate the metrics of a sixfold_t.
// These values depend on four input values: the tile size, the
// size of a cube face image, as found in the input image(s), the
// horizontal field of view of a cube face image, and a minimum
// value for the size of the 'support'. The first two are in units of
// pixels, the third is a floating point value. The support size
// (in pixel units) is the minimal width of the surrounding frame
// which we need to set up good-quality interpolators.

struct metrics_t
{
  const std::size_t tile_width ;
  const std::size_t face_width ;
  const std::size_t support_min ;
  std::size_t inherent_support ;
  std::size_t additional_support ;
  const double face_hfov ;
  double outer_fov ;
  std::size_t frame_width ;
  std::size_t n_tiles ;
  std::size_t outer_width ;
  double model_to_px ;
  double px_to_model ;
  double section_size ;
  double ref90 ;

  // the c'tor only needs one argument: the size of an individual
  // cube face image. This is the size covering the entire image,
  // whose field of view may be ninety degrees or more. The default
  // is ninety degrees, as we would see with standard openEXR
  // cubemaps, but we want to cater for cube face images which
  // already contain some support themselves. A minimal support of
  // four is ample for most direct interpolators.

  metrics_t ( std::size_t _face_width ,
              double _face_hfov = M_PI_2 ,
              std::size_t _support_min = 4UL ,
              std::size_t _tile_width = 64UL
            )
  : face_width ( _face_width ) ,
    face_hfov ( _face_hfov ) ,
    support_min ( _support_min ) ,
    tile_width ( _tile_width )
  {
    // first make sure that certain minimal requirements are met

    // we want even face size for now TODO can we use odd sizes?

    assert ( ( face_width & 1 ) == 0 ) ;

    // the cube face images must have at least 90 degrees fov

    assert ( face_hfov >= M_PI_2 ) ;

    // the tile size must be at least one

    assert ( tile_width > 0 ) ;

    // the tile size must be a power of two

    assert ( ( tile_width & ( tile_width - 1 ) ) == 0 ) ;

    // given the face image's field of view, how much support does
    // the face image already contain? We start out by calculating
    // The cube face image's diameter (in model space units) and the
    // 'overscan' - by how much the diameter exceeds the 2.0 diameter
    // which occurs with a cube face image of precisely ninety degrees
    // field of view

    double overscan = 0.0 ;
    double diameter = 2.0 ;
    inherent_support = 0 ;

    // calculate the diameter in model space units. The diameter
    // for a ninety degree face image would be precisely 2, and
    // if the partial image has larger filed of view, it will be
    // larger.

    if ( face_hfov > M_PI_2 )
    {
      diameter = 2.0 * tan ( face_hfov / 2.0 ) ;
    }

    // calculate scaling factors from model space units to pixel
    // units and from pixel units to model space units.

    model_to_px = double ( face_width ) / diameter ;
    px_to_model = diameter / double ( face_width ) ;

    // The overscan is the distance, in model space units, from
    // the cube face image's edge to the edge of it's central
    // section, holding the ninety degrees wide cube face proper.
    // it's the same as (diameter / 2 - 1). Note that this value
    // may not coincide with a discrete number of pixels for
    // cube face images with more than ninety degrees fov.

    if ( face_hfov > M_PI_2 )
    {      
      overscan = tan ( face_hfov / 2.0 ) - 1.0 ;

      // how wide is the overscan, expressed in pixel units?

      double px_overscan = model_to_px * overscan ;

      // truncate to integer to receive the inherent support in
      // pixel units. If thsi value is as large or larger than the
      // required support, we needn't add any additional space.

      inherent_support = px_overscan ;

      // we are conservative with the 'inherent support': the UL
      // point of the ninety-degrees-proper central part is not
      // on a pixel boundary, so rounding may land us either side.
      // If we're not conservative here, we may end up with too
      // little support.

      if ( inherent_support > 0 )
        inherent_support-- ;
    }

    // if there is more inherent support than the minimal support
    // required, we don't need to provide additional support. If
    // the inherent support is too small, we do need additional
    // support.
  
    if ( inherent_support >= support_min )
      additional_support = 0 ;
    else
      additional_support = support_min - inherent_support ;

    // given the additional support we need - if any - how many tiles
    // are needed to contain a cube face image and it's support?

    std::size_t px_min = face_width + 2 * additional_support ;

    // if the user passes a tile size of one, n_tiles will end up
    // equal to px_min.

    n_tiles = px_min / tile_width ;
    if ( n_tiles * tile_width < px_min )
      n_tiles++ ;

    // this gives us the 'outer width': the size, in pixels, which
    // the required number of tiles will occupy, and the frame size,
    // the number of pixels from the edge of the IR section to
    // the first pixel originating from the cube face image.

    outer_width = n_tiles * tile_width ;
    frame_width = ( outer_width - face_width ) / 2UL ;

    // paranoid

    assert ( ( 2UL * frame_width + face_width ) == outer_width ) ;

    // the central part of each partial image, which covers
    // precisely ninety degrees - the cube face proper - is at
    // a specific distance from the edge of the total section.
    // we know that it's precisely 1.0 from the cube face image's
    // center in model space units. So first we convert the
    // 'outer width' to model space units

    section_size = px_to_model * outer_width ;

    // then subtract 2.0 - the space occupied by the central
    // section of the partial image - the 'cube face proper',
    // spanning ninety degrees - and divide by two:

    ref90 = ( section_size - 2.0 ) / 2.0 ;

    // outer_fov: the field of view covered by the complete
    // 1/6 section of the IR. To fill the entire section with
    // image data from some other source, a rectilinear image
    // with precisely this field of view has to be placed in
    // the corresponding section. This is the same code as
    // for the filling-in of the cubemap with ninety-degree
    // cube faces, only the per-cubeface fov value has to be
    // adapted. If the source is e.g. a full spherical, the
    // resulting sixfold will be very close to what can be
    // 'regenerated' with code in this program from a cubemap
    // with 90-degree faces.

    outer_fov = 2.0 * atan ( section_size / 2.0 ) ;
  }

  // get_pickup_coordinate receives the in-face coordinate and
  // the face index and produces the corresponding coordinate,
  // in pixel units, which is equivalent in the total internal
  // representation image. So this coordinate can be used to
  // 'pick up' the pixel value from the IR image by using the
  // desired interpolation method, and it can be picked up
  // with a single interpolator invocation without any case
  // switching or conditionals to choose the correct cube face
  // image - all the images are combined in the IR.

  // The function is coded as a template to allow for scalar
  // and SIMDized pickups alike. Incoming we have face index(es)
  // and in-face coordinate(s) in model space units, and outgoing
  // we have a coordinate in pixel units pertaining to the entire
  // IR image. Note that the outgoing value is in pixel units,
  // but it's not discrete.

  template < typename face_index_t , typename crd_t >
  void get_pickup_coordinate ( const face_index_t & face_index ,
                               const crd_t & in_face_coordinate ,
                               crd_t & target ) const
  {
    target =    in_face_coordinate // in (-1,1)
              + 1.0                // now in (0,2)
              + ref90 ;            // distance from section UL

    // we could add the per-face offset here, but see below:

    // target[1] += face_index * section_size ;

    // move from model space units to pixel units. This yields us
    // a coordinate in pixel units pertaining to the section.

    target *= model_to_px ;

    // add the per-section offset - Doing this after the move to
    // pixel units makes the calculation more precise, because
    // we can derive the offset from integer values.

    target[1] += f_v ( face_index * int(outer_width) ) ;

    // Subtract 0.5 - we look at pixels as small squares with an
    // extent of 1 pixel unit, and an incoming coordinate which
    // is precisely on the margin of the cube face (a value of
    // +/- 1) has to be mapped to the outermost pixel's margin as
    // well.
    // Note that the output is now in the range (-0.5, width-0.5)
    // and using this with interpolators needing support may
    // access pixels outside the 'ninety degrees proper'.
    // We provide support to cater for that, but its'
    // good to keep the fact in mind. Even nearest-neighbour
    // pickup might fall outside the 'ninety degrees proper'
    // due to small imprecisions and subsequent rounding.

    target -= .5 ;
  }

  // variant of get_pickup_coordinate which yields the pickup
  // coordinate in texture units from with extent in [0,1].

  template < typename face_index_t , typename crd_t >
  void get_pickup_coordinate_tx ( const face_index_t & face_index ,
                                  const crd_t & in_face_coordinate ,
                                  crd_t & target ) const
  {
    target =    in_face_coordinate // in (-1,1)
              + 1.0                // now in (0,2)
              + ref90 ;            // distance from section UL

    // move from model space units to pixel units. This yields us
    // a coordinate in pixel units pertaining to the section.

    target *= model_to_px ;

    // add the per-section offset - Doing this after the move to
    // pixel units makes the calculation more precise, because
    // we can derive the offset from integer values.

    target[1] += f_v ( face_index * int(outer_width) ) ;

    // move to texture coordinates in [0,1]

    target[0] /= outer_width ;
    target[1] /= 6 * outer_width ;
  }
} ;

// sixfold_t contains data for a six-square sky box with support
// around the cube faces to allow for proper filtering and
// interpolation. The content is set up following conventions
// of openEXR environment maps. cubemaps in openEXR contain six
// square images concatenated vertically. The sequence of the
// images, from top to bottom, is: left, right, top, bottom,
// front, back. The top and bottom images are oriented so that
// they align vertically with the 'back' image (lux aligns with
// the front image) - but note that labels like 'front' are
// somewhat arbitrary - I associate 'front' with the center of
// a full spherical image and 'left' with the first image in the
// cubemap.
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
// lookups - and we can code the entire process in SIMD code. The
// frames around the cube faces make it possible to do this with
// mip-mapping and correct anti-aliasing, using OIIO's planar
// texture lookup. This latter part is not yet implemented, for
// now I aim at getting the geometry and logic right and use a
// simple bilinear interpolation to get pixel values.
// The template argument 'nchannels' is the number of channels
// in the image. We accept up to four in main - RGBA should
// come in with associated alpha.

template < int nchannels >
struct sixfold_t
{
  // shorthand for pixell and SIMDized pixels

  typedef zimt::xel_t < float , nchannels > px_t ;
  typedef zimt::simdized_type < px_t , LANES > px_v ;

  // we keep a separate object holding the metrics. This object
  // is quick to compute and light-weight, so we can generate one
  // and use it to inform const members in the sixfold_t.

  const metrics_t metrics ;

  // These references will bind to members of the metrics object.
  // This is merely for cenvenience.

  const std::size_t & face_width ;
  const std::size_t & outer_width ;
  const std::size_t & tile_width ;
  const std::size_t & frame_width ;
  const double & px_to_model ;
  const double & model_to_px ;
  TextureSystem * ts ;
  TextureSystem::TextureHandle * th ;

  // This array holds all the image data. We mark the array as const,
  // but the data it holds are not const. Later on, in comments, I'll
  // refer to this array as the 'IR': the internal representation.

  zimt::array_t < 2 , px_t > store ;

  // We can only create a zimt::array_t if we know it's extent.
  // So we use a static function calculating the size via a
  // metrics object. The calculation is more involved than what
  // we want to pack into an initializer expression.

  static shape_type get_store_shape ( std::size_t _face_width )
  {
    metrics_t metrics ( _face_width ) ;
    shape_type shape { metrics.outer_width , 6 * metrics.outer_width } ;
    return shape ;
  }

  sixfold_t ( std::size_t _face_width ,
              std::size_t _tile_width = 64 )
  : metrics ( _face_width ) ,
    face_width ( metrics.face_width ) ,
    tile_width ( metrics.tile_width ) ,
    outer_width ( metrics.outer_width ) ,
    frame_width ( metrics.frame_width ) ,
    px_to_model ( metrics.px_to_model ) ,
    model_to_px ( metrics.model_to_px ) ,
    store ( get_store_shape ( _face_width ) ) ,
    ts ( nullptr ) ,
    th ( nullptr )
  { }

  // for now, just the TextureSystem and handle, no options

  void set_ts ( TextureSystem * _ts ,
                TextureSystem::TextureHandle * _th )
  {
    ts = _ts ;
    th = _th ;
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

    // the calling code should already have looked at the image and
    // gleaned the face width, but here we check again:

    assert ( xres == face_width ) ;
    assert ( yres == 6 * face_width ) ;

    // find the location of the first cube face's upper left corner
    // in the 'store' array. The frame of additional support around
    // the cube face image is filled in later on.

    px_t * p_ul = store.data() ;
    p_ul += ( frame_width * store.strides ) . sum() ;

    // what's the offset to the same location in the next section?

    std::ptrdiff_t offset = outer_width * outer_width ;

    // read the six cube face images from the 1:6 stripe into the
    // appropriate slots in the sixfold_t object's 'store' array

    for ( int face = 0 ; face < 6 ; face++ )
    {
      auto * p_trg = p_ul + face * offset ;

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
      // The channels are capped at nchannels. We ask OIIO to
      // provide float data.

      auto success =
      inp->read_scanlines ( 0 , 0 ,
                            face * face_width , (face+1) * face_width ,
                            0 , 0 , nchannels ,
                            TypeDesc::FLOAT , p_trg ,
                            nchannels * 4 ,
                            nchannels * 4 * store.strides[1] ) ;
      assert ( success ) ;
    }
  }

  // given a 3D 'ray' coordinate, find the corresponding cube face
  // and the in-face coordinate - note the two references which take
  // the result values. The incoming 'ray' coordinate does not have
  // to be normalized.

  void ray_to_cubeface ( const crd3_v & c ,
                         i_v & face ,
                         crd2_v & in_face ) const
  {
    // form three masks with relations of the numerical values of
    // the 'ray' coordinate. These are sufficient to find out which
    // component has the largest absolute value (the 'dominant' one,
    // along the 'dominant' axis)

    auto m1 = ( abs ( c[RIGHT] ) >= abs ( c[DOWN] ) ) ;
    auto m2 = ( abs ( c[RIGHT] ) >= abs ( c[FORWARD] ) ) ;
    auto m3 = ( abs ( c[DOWN] )  >= abs ( c[FORWARD] ) ) ;

    // form a mask which is true where a specific axis is 'dominant'.
    // We start out looking at the x axis: where the numerical value
    // along the x axis (pointing right) is larger than the other two,
    // 'dom' will be true.

    auto dom = m1 & m2 ;

    // Now we can assign face indexes, stored in f. If the coordinate
    // value is negative along the dominant axis, we're looking at
    // the face opposite and assign a face index one higher. Note
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
    // do it for us. While it's generally preferable to avoid
    // conditionals in inner-loop code, I use conditionals here
    // because most of the time all coordinates will 'land' in
    // the same cube face, so for two cases, the rather expensive
    // code to calculate the face index and in-face coordinate
    // can be omitted. TODO: One might test whether omitting the
    // conditionals is actually slower.

    if ( any_of ( dom ) )
    {
      // extract in-face coordinates for the right and left cube
      // face. the derivation of the x coordinate uses opposites for
      // the two faces, the direction of the y coordinate is equal.
      // Note that some lanes in c[RIGHT] may be zero and result in
      // an Inf result, but these will never be the ones which end
      // up in the result, because only those where c[RIGHT] is
      // 'dominant' will 'make it through', and where c[RIGHT] is
      // dominant, it's certainly not zero. But we rely on the
      // system not to throw a division-by-zero exception, which
      // would spoil our scheme.

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
      // with the 'front' cube face) swap the signs in both expressions.

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

  // variant which takes a given face vector. This is used to
  // approximate the first derivative, which is done by subtracting
  // the result of the coordinate transformation of incoming
  // coordinates which were offset by one sampling step in either
  // canonical direction. We have to look at the same cube face,
  // to avoid the possibility that the cube face changes between
  // the calculation of the result for the actual cordinate and
  // it's offsetted neighbours, which would likely result in a
  // large discontinuity in the in-face coordinate, spoiling the
  // result.
  // So, incoming, we have a 3D ray coordinate and a set of cube
  // face indices, and we'll get in-face coordinates as output.

  void ray_to_cubeface_fixed ( const crd3_v & c ,
                               const i_v & face ,
                               crd2_v & in_face ) const
  {
    // form a mask which is true where a specific axis is 'dominant'.
    // since we have the face indices already, this is simple: it's
    // the face indices shifted to the right to remove their least
    // significant bit, which codes for the sign along the dominant
    // axis.

    auto dom_v = face >> 1 ;

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
    // do it for us. While it's generally preferable to avoid
    // conditionals in inner-loop code, I use conditionals here
    // because most of the time all coordinates will 'land' in
    // the same cube face, so for two cases, the rather expensive
    // code to calculate the face index and in-face coordinate
    // can be omitted. TODO: One might test whether omitting the
    // conditionals is actually slower.

    auto dom = ( dom_v == 0 ) ;
    if ( any_of ( dom ) )
    {
      // extract in-face coordinates for the right and left cube
      // face. the derivation of the x coordinate uses opposites for
      // the two faces, the direction of the y coordinate is equal.
      // Note that some lanes in c[RIGHT] may be zero and result in
      // an Inf result, but these will never be the ones which end
      // up in the result, because only those where c[RIGHT] is
      // 'dominant' will 'make it through', and where c[RIGHT] is
      // dominant, it's certainly not zero. But we rely on the
      // system not to throw a division-by-zero exception, which
      // would spoil our scheme.
      // Since we know the face value already, all we need here is
      // the deivision by the dominant component.

      in_face[0] ( dom ) = - c[FORWARD] / c[RIGHT] ;
      in_face[1] ( dom ) = c[DOWN] / abs ( c[RIGHT] ) ;
    }

    // now set dom true where the y axis (pointing down) has the
    // largest numerical value

    dom = ( dom_v == 1 ) ; 

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
      // with the 'front' cube face) swap the signs in both expressions.

      // lux convention:
      // in_face[0] ( dom ) =   c[RIGHT] / abs ( c[DOWN] ) ;
      // in_face[1] ( dom ) = - c[FORWARD] / c[DOWN] ;

      in_face[0] ( dom ) = - c[RIGHT] / abs ( c[DOWN] ) ;
      in_face[1] ( dom ) =   c[FORWARD] / c[DOWN] ;

    }

    // set dom true where the z axis (pointing forward) has the
    // largest numerical value
    
    dom = ( dom_v == 2 ) ;

    if ( any_of ( dom ) )
    {
      // finally the front and back faces

      in_face[0] ( dom ) = c[RIGHT] / c[FORWARD] ;
      in_face[1] ( dom ) = c[DOWN] / abs ( c[FORWARD] ) ;
    }
  }

  // given the cube face and the in-face coordinate, extract the
  // corresponding pixel values from the internal representation
  // of the cubemap held in the sixfold_t object. We have two
  // variants here, the first one using neraest neighbour
  // interpolation, the second bilinear. The first one is currently
  // unused. The pick-up is not guaranteed to look up pixel data
  // strictly inside the 90-degree cube face 'proper' but may
  // glean some information from the support frame, so this has
  // to be present. Initially we provide a one-pixel-wide
  // support frame of mirrored pixels (if nexessary - if the
  // incoming partial images have 'inherent support' because
  // they span more than ninety degrees, this is not necessary)
  // which is enough for bilinear interpolation. Once we have
  // filled in the support frame, we can use interpolators with
  // wider support.

  void cubemap_to_pixel ( const i_v & face ,
                          crd2_v in_face ,
                          px_v & px ,
                          const int & degree ) const
  {
    crd2_v pickup ;
    metrics.get_pickup_coordinate ( face , in_face , pickup ) ;

    if ( degree == 0 )
    {
      // simple nearest-neighbour lookup. This is not currently used,
      // but it's instructive.

      // convert the in-face coordinates to integer. If the in-face
      // coordinate is right on the edge, the pick-up may fall to
      // pixels outside the 'ninety degrees poper'.

      index_v idx { round ( pickup[0] ) , round ( pickup[1] ) } ;

      // calculate corresponding offsets, using the strides, and
      // obtain pixel data by gathering with this set of offsets.
      // Note how the first multiplication broadcasts the pair of
      // strides to the pair of int vectors

      const auto ofs = ( idx * store.strides ) . sum() * nchannels ;
      const auto * p = (float*) ( store.data() ) ;

      px.gather ( p , ofs ) ;
    }
    else if ( degree == 1 )
    {
      // bilinear interpolation. Since we have incoming coordinates
      // in the range of (-0.5, width-0,5) relative to the 'ninety
      // degrees proper', this interpolator may 'look at' pixels
      // outside the 'ninety degrees proper'. So we need sufficient
      // support here: pickup coordinates with negative values will,
      // for example, look at pixels just outside the 'ninety degree
      // zone'. Note that we want correct support, rather than using
      // the next-best thing (like mirroring on the edge), and we'll
      // generate this support. Then we can be sure that the output
      // is optimal and doesn't carry in any artifacts from the edges.

      const auto * p = (float*) ( store.data() ) ;
      px_v px2 ,help ;

      // find the floor of the pickup coordinate by truncation

      index_v low { pickup[0] , pickup[1] } ;

      // how far is the pick-up coordinate from the floor value?

      auto diff = pickup - crd2_v ( low ) ;

      // gather data pertaining to the pixels at the 'low' coordinate

      index_v idx { low[0] , low[1] } ;
      auto ofs = ( idx * store.strides ) . sum() * nchannels ;
      px.gather ( p , ofs ) ;

      // and weight them according to distance from the 'low' value

      auto one = f_v::One() ;

      px *= ( one - diff[0] ) ;

      // repeat the process for the low coordinate's neighbours

      idx[0] += 1 ; ;
      ofs = ( idx * store.strides ) . sum() * nchannels ;
      help.gather ( p , ofs ) ;
      px += help * diff[0] ;

      // the first partial sum is also weighted, now according to
      // vertical distance

      px *= ( one - diff[1] ) ;

      idx[0] -= 1 ; ;
      idx[1] += 1 ; ;
      ofs = ( idx * store.strides ) . sum() * nchannels ;
      px2.gather ( p , ofs ) ;
      px2 *= ( one - diff[0] ) ;

      idx[0] += 1 ; ;
      ofs = ( idx * store.strides ) . sum() * nchannels ;
      help.gather ( p , ofs ) ;
      px2 += help * diff[0] ;

      px += px2 * diff[1] ;
    }
  }

  void get_filtered_px ( crd2_v pickup ,
                         px_v & px ,
                         const f_v & dxu ,
                         const f_v & dyu ,
                         const f_v & dxv ,
                         const f_v & dyv ,
                         TextureOptBatch & batch_options ) const
  {
     // virtual bool texture (    TextureHandle *texture_handle,
                               // Perthread *thread_info,
                               // TextureOptBatch &options,
                               // Tex::RunMask mask,
                               // const float *s, const float *t,
                               // const float *dsdx, const float *dtdx,
                               // const float *dsdy, const float *dtdy,
                               // int nchannels,
                               // float *result,
                               // float *dresultds = nullptr,
                               // float *dresultdt = nullptr)

    bool result =
    ts->texture ( th , nullptr , batch_options , Tex::RunMaskOn ,
                  pickup[0].data() , pickup[1].data() ,
                  dxu.data() , dxv.data() ,
                  dyu.data() , dyv.data() ,
                  nchannels , (float*) ( px[0].data() ) ) ;

    // std::cout << "pickup: " << pickup << std::endl ;
    // std::cout << "px: " << px << std::endl ;

    assert ( result ) ;
  }

  void _get_filtered_px ( crd2_v pickup ,
                         px_v & px ,
                         const f_v & dxu ,
                         const f_v & dyu ,
                         const f_v & dxv ,
                         const f_v & dyv ) const
  {
    // std::cout << "*** dxu " << dxu << std::endl ;
    // std::cout << "*** dyu " << dyu << std::endl ;
    // std::cout << "*** dxv " << dxv << std::endl ;
    // std::cout << "*** dyv " << dyv << std::endl ;
    auto dx = abs ( dxu ) . at_least ( abs ( dxv ) ) ;
    auto dy = abs ( dyu ) . at_least ( abs ( dyv ) ) ;

    // auto dx = dxu ;
    // dx ( dxv > dxu ) = dxv ;
    // auto dy = dyu ;
    // dy ( dyv > dyu ) = dyv ;
    
    // std::cout << "dx " << dx << std::endl ;
    // std::cout << "dy " << dy << std::endl ;

    auto x0 = pickup[0] - dx / 2.0f ;
    auto x1 = pickup[0] + dx / 2.0f ;
    x1 ( x1 <= x0 ) = x0 + .000001f ;

    auto y0 = pickup[1] - dy / 2.0f ;
    auto y1 = pickup[1] + dy / 2.0f ;
    y1 ( y1 <= y0 ) = y0 + .000001f ;

    auto x = x0 ;
    auto xd = i_v ( ceil ( x0 ) ) ;
    xd ( x0 == ceil ( x0 ) ) += 1 ;

    f_v w = 0.0f ;
    px = 0.0f ;

    // TODO: comes out wrong if x1 < ceuil ( x0 )

    while ( any_of ( x < x1 ) )
    {
      auto xlim = f_v ( xd ) . at_most ( x1 ) ;
      auto sx = xlim - x ;
      sx ( x >= x1 ) = 0.0f ;

      auto y = y0 ;
      auto yd = i_v ( ceil ( y0 ) ) ;
      yd ( y0 == ceil ( y0 ) ) += 1 ;
  
      while ( any_of ( y < y1 ) )
      {
        auto ylim = f_v ( yd ) . at_most ( y1 ) ;
        auto sy = ylim - y ;
        sy ( y >= y1 ) = 0.0f ;

        index_v xylow = { xd - 1 , yd - 1 } ;
        auto ofs = ( xylow * store.strides ) . sum() * nchannels ;
        auto p = (float*) store.data() ;
        px_v pxp ;
        pxp.gather ( p , ofs ) ;

        auto wp = sx * sy ;
        wp ( wp <= 0.0f ) = .000001f ;
        px += pxp * wp ;
        w += wp ;

        yd += 1 ;
        y = f_v ( yd ) . at_most ( y1 ) ;
      }
      xd += 1 ;
      x = f_v ( xd ) . at_most ( x1 ) ;
    }
   assert ( all_of ( w > 0.0f ) ) ;
   px /= w ;
  }

  // variant taking derivatives of the in-face coordinate, which
  // are approximated by calculating the difference to a canonical
  // (target image) coordinate one sample step to the right (u)
  // or below (v), respectively. The derivatives are in texture
  // units aready, and we also convert the pickup coordinate to
  // texture units.

  void cubemap_to_pixel ( const i_v & face ,
                          crd2_v in_face ,
                          px_v & px ,
                          const f_v & dxu ,
                          const f_v & dyu ,
                          const f_v & dxv ,
                          const f_v & dyv ,
                          TextureOptBatch & bo ) const
  {
    crd2_v pickup ;
    metrics.get_pickup_coordinate_tx ( face , in_face , pickup ) ;

    // // move the pickup to texture coordinates in [0,1]
    // 
    // pickup[0] /= store.shape[0] ;
    // pickup[1] /= store.shape[1] ;

    get_filtered_px ( pickup , px , dxu , dyu , dxv , dyv , bo) ;
  }

  // After the cube faces have been read from disk, they are surrounded
  // by black (or even undefined) pixels. We want to provide minimal
  // support, this support's quality is not crucial, but it should not
  // be black, but rather like mirroring on the edge, which this function
  // does - it produces a one-pixel-wide frame with mirrored pixels
  // around each of the cube faces. In the next step, we want to fill
  // in the support frame around the cube faces proper, and we may have
  // to access image data close to the margin. Rather than implementing
  // the mirroring on the edge by manipulating the coordinate of the
  // pick-up (e.g. clamping it) having the one-pixel-wide minimal
  // support allows us to do the next stage without looking at the
  // coordinates: we can be sure that the pick-up will not exceed
  // the support area.

  void mirror_around()
  {
    auto * p_base = store.data() ;

    for ( int face = 0 ; face < 6 ; face++ )
    {
      // get a pointer to the upper left of the cube face 'proper'

      auto * p_frame = p_base + face * outer_width * store.strides[1]
                              + frame_width * store.strides[1]
                              + frame_width * store.strides[0] ;

      // get a zimt view to the current cube face

      zimt::view_t < 2 , px_t > cubeface
        ( p_frame , store.strides , { face_width , face_width } ) ;

      // we use 2D discrete coordinates

      typedef zimt::xel_t < int , 2 > ix_t ;

      // mirror the horizontal edges

      for ( int x = -1 ; x <= int ( face_width ) ; x++ )
      {
        ix_t src { x ,  0 } ;
        ix_t trg { x , -1 } ;
        cubeface [ trg ] = cubeface [ src ] ;
        src [ 1 ] = face_width - 1 ;
        trg [ 1 ] = face_width ;
        cubeface [ trg ] = cubeface [ src ] ;
      }

      // and the vertical edges

      for ( int y = -1 ; y <= int ( face_width ) ; y++ )
      {
        ix_t src {  0 , y } ;
        ix_t trg { -1 , y } ;
        cubeface [ trg ] = cubeface [ src ] ;
        src [ 0 ] = face_width - 1 ;
        trg [ 0 ] = face_width ;
        cubeface [ trg ] = cubeface [ src ] ;
      }
    }
  }

  // this member function will use class fill_frame_t, so it's
  // implementation follows below that class definition.

  void fill_support ( int degree ) ;
} ;

// this functor template converts incoming in-face coordinates
// to ray coordinates for a given face index, which is passed
// as a template argument - so the sixfold 'if constexpr ...' is
// not a conditional, it's just a handy way of putting the code
// into a single function without having to write partial template
// specializations for the six possible face indices.

template < face_index_t F , int nchannels >
struct ir_to_ray
: public zimt::unary_functor < v2_t , v3_t , LANES >
{
  const sixfold_t<nchannels> & sf ;

  ir_to_ray ( const sixfold_t<nchannels> & _sf )
  : sf ( _sf )
  { }

  template < typename I , typename O >
  void eval ( const I & crd2 , O & crd3 )
  {
    if constexpr ( F == CM_FRONT )
    {
      crd3[RIGHT]   =   crd2[RIGHT] ;
      crd3[DOWN]    =   crd2[DOWN]  ;
      crd3[FORWARD] =   1.0f ;
    }
    else if constexpr ( F == CM_BACK )
    {
      crd3[RIGHT]   = - crd2[RIGHT] ;
      crd3[DOWN]    =   crd2[DOWN]  ;
      crd3[FORWARD] = - 1.0f ;
    }
    else if constexpr ( F == CM_RIGHT )
    {
      crd3[RIGHT] =     1.0f ;
      crd3[DOWN] =      crd2[DOWN]  ;
      crd3[FORWARD] = - crd2[RIGHT] ;
    }
    else if constexpr ( F == CM_LEFT )
    {
      crd3[RIGHT] =   - 1.0f ;
      crd3[DOWN] =      crd2[DOWN]  ;
      crd3[FORWARD] =   crd2[RIGHT] ;
    }

    // for bottom and top, note that we're using openEXR convention.
    // to use lux convention, invert the signs.

    else if constexpr ( F == CM_BOTTOM )
    {
      crd3[RIGHT] =   - crd2[RIGHT] ;
      crd3[DOWN] =      1.0f ;
      crd3[FORWARD] =   crd2[DOWN]  ;
    }
    else if constexpr ( F == CM_TOP )
    {
      crd3[RIGHT] =   - crd2[RIGHT] ;
      crd3[DOWN] =    - 1.0f ;
      crd3[FORWARD] = - crd2[DOWN]  ;
    }
  }
} ;

// this functor converts incoming 2D coordinates pertaining
// to the entire IR array to 3D ray coordinates. This is
// the general form - if the face index is known beforehand,
// instantiate the template above for a more specific functor.
// We expect the incoming coordinates to be centered - the
// origin is at the center of the IR image.
// This functor can serve to populate the IR image: set up a
// functor yielding model space coordinates pertaining to pixels
// in the IR image, pass these model space coordinates to this
// functor, receiving ray coordinates, then glean pixel values
// for the given ray by evaluating some functor taking ray
// coordinates and yielding pixels.

template < int nchannels >
struct ir_to_ray_gen
: public zimt::unary_functor < v2_t , v3_t , LANES >
{
  const sixfold_t < nchannels > & sf ;

  ir_to_ray_gen ( const sixfold_t < nchannels > & _sf )
  : sf ( _sf )
  { }

  // incoming, we have 2D model space coordinates, with the origin
  // at the (total!) IR image's center.

  template < typename I , typename O >
  void eval ( const I & _crd2 , O & crd3 )
  {
    I crd2 ( _crd2 ) ;

    // move the vertical origin to the top of the first section

    crd2[1] += 3.0 * sf.metrics.section_size ;

    // The numerical constants for the cube faces/sections are set
    // up so that a simple division of the y coordinate yields the
    // corresponding section index.

    i_v section ( crd2[1] / sf.metrics.section_size ) ;

    // Subtracting the offset to the section's beginning produces
    // the vertical in-face coordinate, and since the incoming 2D
    // coordinate is centered, we already have the horizontal
    // in-face coordinate.

    crd2[1] -= section * sf.metrics.section_size ;
    crd2[1] -= ( sf.metrics.section_size / 2.0 ) ;

    // the numerical constants can also yield the 'dominant' axis
    // by dividing the value by two (another property which is
    // deliberate):

    i_v dom ( section >> 1 ) ;

    // again we use a conditional to avoid lengthy calculations
    // when there aren't any populated lanes for the given predicate

    if ( any_of ( dom == 0 ) )
    {
      auto m = ( section == CM_RIGHT ) ;
      if ( any_of ( m ) )
      {
        crd3[RIGHT](m) =     1.0f ;
        crd3[DOWN](m) =      crd2[DOWN] ;
        crd3[FORWARD](m) = - crd2[RIGHT] ;
      }
      m = ( section == CM_LEFT ) ;
      if ( any_of ( m ) )
      {
        crd3[RIGHT](m) =   - 1.0f ;
        crd3[DOWN](m) =      crd2[DOWN] ;
        crd3[FORWARD](m) =   crd2[RIGHT] ;
      }
    }
    if ( any_of ( dom == 1 ) )
    {
      auto m = ( section == CM_BOTTOM ) ;
      if ( any_of ( m ) )
      {
        crd3[RIGHT](m) =   - crd2[RIGHT] ;
        crd3[DOWN](m) =      1.0f ;
        crd3[FORWARD](m) =   crd2[DOWN] ;
      }
      m = ( section == CM_TOP ) ;
      if ( any_of ( m ) )
      {
        crd3[RIGHT](m) =   - crd2[RIGHT] ;
        crd3[DOWN](m) =    - 1.0f ;
        crd3[FORWARD](m) = - crd2[DOWN] ;
      }
    }
    if ( any_of ( dom == 2 ) )
    {
      auto m = ( section == CM_FRONT ) ;
      if ( any_of ( m ) )
      {
        crd3[RIGHT](m)   =   crd2[RIGHT] ;
        crd3[DOWN](m)    =   crd2[DOWN] ;
        crd3[FORWARD](m) =   1.0f ;
      }
      m = ( section == CM_BACK ) ;
      if ( any_of ( m ) )
      {
        crd3[RIGHT](m)   = - crd2[RIGHT] ;
        crd3[DOWN](m)    =   crd2[DOWN] ;
        crd3[FORWARD](m) = - 1.0f ;
      }
    }
  }
} ;

// this functor is used to fill the frame of support pixels in the
// array in the sixfold_t object. incoming, we have 2D coordinates,
// which, for the purpose at hand, will lie outside the cube face.
// But we can still convert these image coordinates to planar
// coordinates in 'model space' and then further to 'pickup'
// coordinates into the array in the sixfold_t object. When we
// pick up data from the neighbouring cube faces, we produce
// support around the cube face proper, 'regenerating' what would
// have been there in the first place, if we had had partial images
// with larger field of view. Due to the geometric transformation
// and interpolation, the regenerated data in the frame will not
// be 'as good' as genuine image data, but we'll never actually
// 'look at' the regenerated data: they only serve as support for
// filtering and mip-mapping, and for that purpose, they are certainly
// 'good enough'.
// The eval functor could of course also produce pixel values for
// 2D coordinates inside the cube face image. Note that we have 
// discrete incoming coordinates.

template < int nchannels >
struct fill_frame_t
: public zimt::unary_functor
    < v2i_t , zimt::xel_t < float , nchannels > , LANES >
{
  const sixfold_t < nchannels > & sf ;
  const int face ;
  const int degree ;
  const int ithird ;

  typedef zimt::xel_t < float , nchannels > px_t ;
  typedef zimt::simdized_type < px_t , LANES > px_v ;

  fill_frame_t ( const sixfold_t < nchannels > & _cubemap ,
                 const int & _face ,
                 const int & _degree )
  : sf ( _cubemap ) ,
    face ( _face ) ,
    degree ( _degree ) ,
    ithird ( _cubemap.model_to_px * 2 )
  { }

  void eval ( const v2i_v & crd2 , px_v & px ) const
  {
    crd3_v crd3 ;

    // since 'face' is const, this case switch should be optimized away.
    // here, we move from discrete image coordinates to float 3D
    // coordinates, but we don't scale to model coordinates and do the
    // arithmetic in integer until the final conversion to float:
    // the scale does not matter, because the next processing step
    // is insensitive to scale.
    // Note that we're obtaining crd2 values from a linspace_t which
    // provides readily shifted and doubled coordinates, hence the
    // factor of two in the initialization of ithird.

    switch ( face )
    {
      case CM_FRONT :
        crd3[RIGHT]   =   crd2[RIGHT] ;
        crd3[DOWN]    =   crd2[DOWN] ;
        crd3[FORWARD] =   ithird ;
        break ;
      case CM_BACK :
        crd3[RIGHT]   = - crd2[RIGHT] ;
        crd3[DOWN]    =   crd2[DOWN] ;
        crd3[FORWARD] = - ithird ;
        break ;
      case CM_RIGHT :
        crd3[RIGHT] =     ithird ;
        crd3[DOWN] =      crd2[DOWN] ;
        crd3[FORWARD] = - crd2[RIGHT] ;
        break ;
      case CM_LEFT :
        crd3[RIGHT] =   - ithird ;
        crd3[DOWN] =      crd2[DOWN] ;
        crd3[FORWARD] =   crd2[RIGHT] ;
        break ;
      // for bottom and top, note that we're using openEXR convention.
      // to use lux convention, invert the signs.
      case CM_BOTTOM :
        crd3[RIGHT] =   - crd2[RIGHT] ;
        crd3[DOWN] =      ithird ;
        crd3[FORWARD] =   crd2[DOWN] ;
        break ;
      case CM_TOP :
        crd3[RIGHT] =   - crd2[RIGHT] ;
        crd3[DOWN] =    - ithird ;
        crd3[FORWARD] = - crd2[DOWN] ;
        break ;
     } ;

    // next we use the 3D coordinates we have just obtained to
    // find a source cube face and in-face coordinates - this will
    // be a different cube face to 'face', and it will be one
    // which actually can provide data for location we're filling
    // in. So we re-project the content from adjoining cube faces
    // into the support area around the cube face we're currently
    // surrounding with a support frame.
    // Note how we use sixfold_t's member functions for most of
    // the work - gleaning pixel values from the sixfold_t object
    // works the same for both purposes. The difference here is
    // the source of the 3D ray coordinates: here they originate
    // from target locations on the support frame, whereas the
    // eval member function in ll_to_px_t produces them from
    // incoming lat/lon coordinates.

    i_v fv ;
    crd2_v in_face ;
    sf.ray_to_cubeface ( crd3 , fv , in_face ) ;

    // finally we use this information to obtain pixel values,
    // which are written to the target location.

    sf.cubemap_to_pixel ( fv , in_face , px , degree ) ;
  }
} ;

// fill_support uses the fill_frame_t functor to populate the
// frame of support. The structure of the code is similar to
// 'mirror_around', iterating over the six sections of the
// array and manipulating each in turn. But here we fill in
// the entire surrounding frame, not just a pixel-wide line,
// and we pick up data from neighbouring cube faces.

template < int nchannels >
void sixfold_t < nchannels > :: fill_support ( int degree )
{
  auto * p_base = store.data() ;

  for ( int face = 0 ; face < 6 ; face++ )
  {
    // set up the 'gleaning' functor

    fill_frame_t<nchannels> fill_frame ( *this , face , degree ) ;
  
    // get a pointer to the upper left of the section of the IR

    auto * p_frame = p_base + face * outer_width * store.strides[1] ;
    
    // we form a view to the current section of the array in the
    // sixfold_t object. The shape of this view is the 'notional
    // shape' zimt::process will work with.

    zimt::view_t < 2 , px_t >
      section ( p_frame , store.strides ,
               { outer_width , outer_width } ) ;

    auto & shp = section.shape ;

    // we'll use a 'loading bill' to narrow the filling-in down
    // to the areas which are outside the cube face.

    zimt::bill_t bill ;
    
    // now we fill in the lower an upper limits. This is a good
    // demonstration of how these parameters can be put to use.
    // We use a 'notional' shape which encompasses the entire
    // section, but the iteration will only visit those coordinates
    // which are in the range given by the limits.
    // our linspace_t will generate doubled coordinates -
    // scaling up by a factor of two is irrelevant for the next
    // stage of the processing, and it allows us to code the
    // sampling mathematics entirely in int until the stage
    // where we do the unavoidable division to project the
    // ray onto a plane. Without the doubling, we'd have to
    // start out at ( outer_width -1 ) / 2, which can't be
    // expressed precisely in int.

    auto ishift = outer_width - 1 ;
    zimt::linspace_t < int , 2 , 2 , LANES > ls ( -ishift , 2 ) ;

    zimt::storer < float , nchannels , 2 , LANES > st ( section ) ;
    
    // fill in the stripe above the cube face

    bill.lower_limit = { 0 , 0 } ;
    bill.upper_limit = { long(outer_width) , long(frame_width) } ;
    zimt::process ( shp , ls , fill_frame , st , bill ) ;

    // fill in the stripe below the cube face

    bill.lower_limit = { 0 , long(frame_width + face_width) } ;
    bill.upper_limit = { long(outer_width) , long(outer_width) } ;
    zimt::process ( shp , ls , fill_frame , st , bill ) ;

    // fill in the stripe to the left of the cube face

    bill.lower_limit = { 0 , 0 } ;
    bill.upper_limit = { long(frame_width) , long(outer_width) } ;
    zimt::process ( shp , ls , fill_frame , st , bill ) ;

    // fill in the stripe to the right of the cube face

    bill.lower_limit = { long(frame_width + face_width) , 0 } ;
    bill.upper_limit = { long(outer_width) , long(outer_width) } ;
    zimt::process ( shp , ls , fill_frame , st , bill ) ;
  }
}

// ll_to_px_t is the functor used as 'act' functor for zimt::process
// to produce pixel values for lat/lon coordinates.
// This functor accesses the data in the sixfold_t object, yielding
// pixel data for incoming 2D lat/lon coordinates. It implements a
// typical processing pipeline, using three distinct stages: first,
// the incoming 2D lat/lon coordinates are converted to 3D 'ray'
// coordinates, then these are used to figure out the corresponding
// cube face and the 2D in-face coordinates, and finally these values
// are used to obtain pixel data from the sixfold_t object.
// Note how we code the entire process as a SIMD operation: we're
// not handling individual coordinates or pixels, but their 'SIMDized'
// equivalents. With this functor, we can glean pixel values for
// arbitrary discrete 2D coordinates located on any of the six
// planes which surround the origin at 1.0 units distance in model
// space.

template < int nchannels >
struct ll_to_px_t
: public zimt::unary_functor
   < v2_t , zimt::xel_t < float , nchannels > , LANES >
{
  // some types

  typedef zimt::xel_t < float , nchannels > px_t ;
  typedef zimt::simdized_type < px_t , LANES > px_v ;

  // infrastructure
  
  const ll_to_ray_t ll_to_ray ;

  // for lookup with OIIO's texture system, we need batch options.
  // I'd keep them in the code which actually uses them, but the
  // OIIO 'texture' function expects an lvalue, so I have to pass
  // them in from here by reference.

  TextureOptBatch batch_options ;

  // source of pixel data

  const sixfold_t < nchannels > & cubemap ;
  const int degree ;
  v2_t delta ;

  // ll_to_px_t's c'tor obtains a const reference to the sixfold_t
  // object holding pixel data, the degree of the interpolator and
  // the delta of s sampling step, to form derivatives.

  ll_to_px_t ( const sixfold_t < nchannels > & _cubemap ,
               const int & _degree ,
               const v2_t & _delta )
  : cubemap ( _cubemap ) ,
    degree ( _degree ) ,
    delta ( _delta ) ,
    ll_to_ray() // g++ is picky.
  {
    // let's not be too conservative for now

    batch_options.conservative_filter = false ;
  }

  // 'eval' function which will be called by zimt::process.
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
    ll_to_ray.eval ( lat_lon , crd3 ) ;

    // find the cube face and in-face coordinate for 'c'

    i_v face ;
    crd2_v in_face ;
    cubemap.ray_to_cubeface ( crd3 , face , in_face ) ;

    if ( degree == -1 )
    {
      // for this interpolation, we need the derivatives, in
      // pixel coordinates pertaining to the IR image. We use
      // the same approach as in cubemap.cc: we obtain source
      // image coordinates for two points, each one step in
      // one of the canonical sirections away. Then we subtract
      // the unmodified pickup coordinate.

      // get a copy of the incoming 2D lat/lon coordinates and
      // add the sample step width to the horizontal component

      crd2_v dx1 = lat_lon ;
      dx1[0] += delta[0] ;

      // convert the results to 3D ray coordinates

      crd3_v dx1_3 ;
      ll_to_ray.eval ( dx1 , dx1_3 ) ;

      // and then to in-face coordinates, using the same face
      // indices we gleaned above. This is important: if we were
      // to compare in-face coordinates from different cube faces,
      // we'd get totally wrong results.

      crd2_v dx1_if ;
      cubemap.ray_to_cubeface_fixed ( dx1_3 , face , dx1_if ) ;

      // now we can calculate the approximation of the derivative
      // by forming the difference from the in-face coordinate of
      // the pick-up location. We get two components, which we
      // handle separately, and we scale to pixel coordinates.

      auto dxu = ( dx1_if[0] - in_face[0] ) * cubemap.model_to_px ;
      auto dyu = ( dx1_if[1] - in_face[1] ) * cubemap.model_to_px ;

      // we repeat the process for a coordinate one sample step away
      // along the vertical axis

      crd2_v dy1 = lat_lon ;
      dy1[1] += delta[1] ;
      crd3_v dy1_3 ;
      ll_to_ray.eval ( dy1 , dy1_3 ) ;
      crd2_v dy1_if ;
      cubemap.ray_to_cubeface_fixed ( dy1_3 , face , dy1_if ) ;

      auto dxv = ( dy1_if[0] - in_face[0] ) * cubemap.model_to_px ;
      auto dyv = ( dy1_if[1] - in_face[1] ) * cubemap.model_to_px ;

      // now we can call the cubemap_to_pixel variant which takes
      // derivatives

      // move the derivatives to texture coordinates in [0,1]

      dxu /= cubemap.store.shape[0] ;
      dxv /= cubemap.store.shape[0] ;
      dyu /= cubemap.store.shape[1] ;
      dyv /= cubemap.store.shape[1] ;

      cubemap.cubemap_to_pixel ( face , in_face , px ,
                                 dxu , dyu , dxv , dyv ,
                                 batch_options ) ;
    }
    else
    {
      // use this information to obtain pixel values

      cubemap.cubemap_to_pixel ( face , in_face , px , degree ) ;
    }
  }

} ;

// This functor 'looks at' a full spherical image. It receives
// 2D lat/lon coordinates and yields pixel values. Pick-up is done
// with bilinear interpolation.

template < int nchannels >
struct eval_latlon
: public zimt::unary_functor
           < v2_t , zimt::xel_t < float , nchannels > , LANES >
{
  typedef zimt::xel_t < float , nchannels > px_t ;
  typedef zimt::simdized_type < px_t , 16 > px_v ;

  // latlon contains a view to an image in 2:1 aspect ratio in
  // spherical projection.

  const zimt::view_t < 2 , px_t > & latlon ;

  // scaling factor to move from model space coordinates to
  // image coordinates

  const double scale ;

  eval_latlon ( const zimt::view_t < 2 , px_t > & _latlon )
  : latlon ( _latlon ) ,
    scale ( _latlon.shape[1] / M_PI )
  { }

  template < typename I , typename O >
  void eval ( const I & _in , O & out )
  {
    // we have incoming model space coordinates, in[0] is the
    // longitude in the range of [-pi,pi] and in[1] is the
    // latitude in the range of [-pi/2,pi/2]. First we move
    // to image coordinates.

    const zimt::xel_t < double , 2 > shift { M_PI , M_PI_2 } ;

    auto in = ( _in + shift ) * scale ;

    // if the coordinate is now precisely zero, this corresponds
    // to a pick-up point at the top or left margin of the UL
    // pixel, which is 0.5 away from it's center

    in -= .5 ;

    // now we move to discrete values for the index calculations
    // the nearest discrete coordinate with components smaller
    // than or equal to 'in' is found by applying 'floor'. We
    // store the result in 'uli' for 'upper left index'.

    v2i_v uli { floor ( in[0] ) , floor ( in[1] ) } ;

    // the distance of this coordinate, both in horizontal and
    // vertical direction, is stored in 'wr' ('weight right'),
    // and it will be used to weight the right-hand part of the
    // constituents (ur and lr)

    auto wr = in - uli ;

    // 'wl' ('weight left') is used for the opposite constituents,
    // namely ul and ll.

    auto wl = 1.0 - wr ;

    // from the discrete upper left value, we derive it's three
    // neighbours to the right.

    v2i_v uri ( uli ) ;
    uri[0] += 1 ;
    
    v2i_v lli ( uli ) ;
    lli[1] += 1 ;
    
    v2i_v lri ( lli ) ;
    lri[0] += 1 ;
    
    // shorthand for width and height

    int w = latlon.shape[0] ;
    int h = latlon.shape[1] ;

    // first we look at the vertical axis and map excessive values
    // back into the range. Note how this isn't as straightforward
    // as handling the horizontal: The continuation is on the
    // opposite hemisphere (add w/2 to the horizontal component)
    // then mirror on the pole.

    uli[0] ( uli[1] < 0 ) += w / 2 ;
    uli[1] ( uli[1] < 0 ) = 1 - uli[1] ; // e.g. -1 -> 0, opposite

    uri[0] ( uri[1] < 0 ) += w / 2 ;
    uri[1] ( uri[1] < 0 ) = 1 - uri[1] ; // e.g. -1 -> 0, opposite

    lli[0] ( lli[1] >= h ) += w / 2 ;
    lli[1] ( lli[1] >= h ) = ( h - 1 ) - ( lli[1] - h ) ;

    lri[0] ( lri[1] >= h ) += w / 2 ;
    lri[1] ( lri[1] >= h ) = ( h - 1 ) - ( lri[1] - h ) ;

    // now we look at the horizontal axis - the longitude axis -
    // and map any coordinates which are outside the range back in,
    // exploiting the periodicity. Note that the code for the vertical
    // may have put values way outside the range (by adding w/2), but
    // not so far as that they wouldn't be mapped back into the range
    // now. We don't expect uri and lri to ever have negative values.

    uli[0] ( uli[0] < 0 ) += w ;
    lli[0] ( lli[0] < 0 ) += w ;

    uli[0] ( uli[0] >= w ) -= w ;
    uri[0] ( uri[0] >= w ) -= w ;
    lli[0] ( lli[0] >= w ) -= w ;
    lri[0] ( lri[0] >= w ) -= w ;

    // this can go later:

    assert ( all_of ( uli[0] >= 0 ) ) ;
    assert ( all_of ( uri[0] >= 0 ) ) ;
    assert ( all_of ( lli[0] >= 0 ) ) ;
    assert ( all_of ( lri[0] >= 0 ) ) ;

    assert ( all_of ( uli[0] < w ) ) ;
    assert ( all_of ( uri[0] < w ) ) ;
    assert ( all_of ( lli[0] < w ) ) ;
    assert ( all_of ( lri[0] < w ) ) ;

    assert ( all_of ( uli[1] >= 0 ) ) ;
    assert ( all_of ( uri[1] >= 0 ) ) ;
    assert ( all_of ( lli[1] >= 0 ) ) ;
    assert ( all_of ( lri[1] >= 0 ) ) ;

    assert ( all_of ( uli[1] < h ) ) ;
    assert ( all_of ( uri[1] < h ) ) ;
    assert ( all_of ( lli[1] < h ) ) ;
    assert ( all_of ( lri[1] < h ) ) ;

    // base pointer for the gather operation

    const auto * p = (float*) ( latlon.data() ) ;

    // obtain the four constituents by first truncating their
    // coordinate to int and then gathering from p.

    index_v idxul { uli[0] , uli[1] } ;
    auto ofs = ( idxul * latlon.strides ) . sum() * nchannels ;
    px_v pxul ;
    pxul.gather ( p , ofs ) ;

    index_v idxur { uri[0] , uri[1] } ;
    ofs = ( idxur * latlon.strides ) . sum() * nchannels ;
    px_v pxur ;
    pxur.gather ( p , ofs ) ;

    index_v idxll { lli[0] , lli[1] } ;
    ofs = ( idxll * latlon.strides ) . sum() * nchannels ;
    px_v pxll ;
    pxll.gather ( p , ofs ) ;

    index_v idxlr { lri[0] , lri[1] } ;
    ofs = ( idxlr * latlon.strides ) . sum() * nchannels ;
    px_v pxlr ;
    pxlr.gather ( p , ofs ) ;

    // apply the bilinear formula with the weights gleaned above

    out  = wl[1] * ( wl[0] * pxul + wr[0] * pxur ) ;
    out += wr[1] * ( wl[0] * pxll + wr[0] * pxlr ) ;
  }
} ;

template < int nchannels >
using pix_t = zimt::xel_t < float , nchannels > ;

// this function takes a lat/lon image as it's input and transforms
// it into the IR image of a sixfold_t. The source image is passed
// in as a zimt::view_t, the target as a reference to sixfold_t.

template < int nchannels >
void fill_ir ( const zimt::view_t < 2 , pix_t<nchannels> > & latlon ,
               sixfold_t < nchannels > & sf )
{
  // we set up a linspace_t object to step through the sample points.
  // in image coordinates, we'd use these start and step values:

  v2_t start { - double ( sf.store.shape[0] - 1 ) / 2.0 ,
               - double ( sf.store.shape[1] - 1 ) / 2.0 } ;

  v2_t step { 1.0 , 1.0 } ;

  // but we want to feed model space coordinates to the 'act'
  // functor, hence we scale:

  start *= sf.px_to_model ;
  step *= sf.px_to_model ;

  zimt::linspace_t < float , 2 , 2 , LANES > ls ( start , step ) ;

  // the act functor is a chain of three separate functors, which we
  // set up first:

  // this one converts coordinates pertaining to the IR image to
  // 3D ray coordinates.

  ir_to_ray_gen itr ( sf ) ;

  // this one converts 3D ray coordinates to lat/lon values

  ray_to_ll_t rtl ;

  // and this one does the pick-up from the lat/lon image

  eval_latlon < nchannels > ltp ( latlon ) ;

  // now we form the act functor by chaining these three functors

  auto act = itr + rtl + ltp ;

  // the data are to be stored to the IR image, held in sf.store

  zimt::storer < float , nchannels , 2 , LANES > st ( sf.store ) ;

  // showtime!

  zimt::process ( sf.store.shape , ls , act , st ) ;
}

// the 'worker' function does all the work. I had the code im main,
// but now I've added support for images with up to four channels,
// in order to support RGBA and monochrome images as well, and I
// need to have the number of channels as a template argument to
// 'trickle down' to code manipulating pixel data, hence the
// factoring-out of the worker code to this function template.

template < int nchannels >
void worker ( std::unique_ptr < ImageInput > & inp ,
              std::size_t height ,
              const std::string & latlon )
{
  // we expect an image with 1:6 aspect ratio. We don't check
  // for metadata specific to environments - it can be anything,
  // but the aspect ratio must be correct.

  const ImageSpec &spec = inp->spec() ;
  int xres = spec.width ;
  int yres = spec.height ;

  std::cout << "spec.width: " << spec.width << std::endl ;
  assert ( spec.width * 6 == spec.height ) ;

  // we set up the sixfold_t object, 'preparing the ground'
  // to pull in the image data

  sixfold_t<nchannels> sf ( spec.width ) ;

  // load the cube faces into slots in the array in the
  // sixfold_t object

  sf.load ( inp ) ;

  // now we build up the 'support'. The square images for the
  // cube faces are cut off at the ninety-degree point, and to
  // interpolate in the vicinity of a cube face's margin, we
  // need some support in good quality. Further out from the
  // margins, the quality of the support data isn't so critical -
  // these pixels only come to play when the cube face is
  // mip-mapped. But going over these pixels again and filling
  // them in with values from bilinear interpolation over the
  // data in the neighbouring cube faces does no harm - we might
  // reduce the target area to some thinner stripe near the
  // edge, but since this is only working over a small-ish part
  // of the data, for now, we'll just redo the lot a few times.

  // we start out mirroring out the square's 1-pixel edge if
  // there is no 'inherent support', which we have only if the
  // cube face image spans more than ninety degrees.

  if ( sf.metrics.inherent_support == 0 )
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

  // if ( save_ir )
  //   save_array ( "internal.exr" , sf.store ) ;

  TextureSystem * ts = TextureSystem::create() ;
  ustring uenvironment ( "internal.exr" ) ;
  auto th = ts->get_texture_handle ( uenvironment ) ;

  std::cout << "ts: " << ts << " th: " << th << std::endl ;

  sf.set_ts ( ts , th ) ;

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

  // The target array will receive the output pixel data

  typedef zimt::xel_t < float , nchannels > px_t ;

  zimt::array_t < 2, px_t > trg ( { 2 * height , height } ) ;

  // set up a linspace_t over the lon/lat sample points as get_t
  // (a.k.a input generator). d is the step width from one sample
  // to the next:

  double d = 2.0 * M_PI / trg.shape[0] ;

  // note that the first sample point is not at -pi, -pi/2 but
  // half a sample step in the horizonal and vertical toward the
  // center. This way, the sample points are distributed evenly
  // around the image center - the last sample point will be
  // at pi-d/2, pi/2-d/2.
  
  v2_t start { - M_PI + d / 2.0 , - M_PI_2 + d / 2.0 } ;
  v2_t step { d , d } ;

  zimt::linspace_t < float , 2 , 2 , LANES > linspace ( start , step ) ;

  // set up a zimt::storer writing to to the target array

  zimt::storer < float , nchannels , 2 , LANES > st ( trg ) ;

  // set up ll_to_px_t using bilinear interpolation (argument 1).
  // This is the functor which takes lon/lat coordinates and
  // produces pixel data from the internal representation of the
  // cubemap held in the sixfold_t object

  ll_to_px_t<nchannels> act ( sf , -1 , step ) ;

  // showtime! call zimt::process.

  zimt::bill_t bill ;
  bill.njobs = 1 ;

  zimt::process ( trg.shape , linspace , act , st , bill ) ;

  // finally we store the data to an image file - note how we have
  // float data in 'trg', and OIIO will convert these on-the-fly to
  // HALF, as specified in the write_image invocation.
  // Note that the target will receive image data in the same colour
  // space as the input. If you feed, e.g. openEXR, and store to JPEG,
  // the image will look too dark, because the linear RGB data are
  // stored as if they were sRGB.

  save_array<nchannels> ( latlon , trg ) ;

  // if 'regenerate' is true, we add a control: we transform the
  // lat/lon image we have just made back into the sixfold_t object's
  // 'store' - the IR image. The resulting image is then saved to
  // 'regen.exr'. this image can be compared to 'internal.exr' which
  // can be produced by setting save_ir true. We'll see some degradation
  // due to the to-and-fro conversion, both 'only' with bilinear
  // interpolation, but we can also see that the geometry is correct.

  if ( regenerate )
  {
    // let's regenerate the IR from the lat/lon

    fill_ir < nchannels > ( trg , sf ) ;

    save_array ( "regen.exr" , sf.store ) ;
  }
}

int main ( int argc , char * argv[] )
{
  // collect arguments
  // TODO: consider 'normal' argument syntax.

  if ( argc != 4 )
  {
    std::cerr << "latlon: skybox <cubemap> <height> <latlon>" << std::endl ;
    std::cerr << "cubemap must contain a 1:6 single-image cubemap" << std::endl ;
    std::cerr << "height is the height of the output" << std::endl ;
    std::cerr << "latlon is the filename for the output, a full" << std::endl ;
    std::cerr << "spherical in openEXR latlon environment format." << std::endl ;
    exit ( -1 ) ;
  }

  std::string skybox ( argv[1] ) ;
  std::size_t height = std::stoi ( argv[2] ) ;
  std::string latlon ( argv[3] ) ;

  std::cout << "skybox: " << skybox
            << " output height: " << height
            << " output: " << latlon << std::endl ;

  // uncomment this to see the IR and the image regenerated from
  // the output, which should be like the IR image, with some
  // degradation due to the transformation to lat/lon and back.

  save_ir = true ;

  auto inp = ImageInput::open ( skybox ) ;
  assert ( inp != nullptr ) ;

  const ImageSpec &spec = inp->spec() ;
  int nchannels = spec.nchannels ;

  if ( nchannels >= 4 )
  {
    // if there is an alpha channel, we rely on OIIO to provide
    // associated alpha values, which can be processed as such without
    // the need to handle the alpha channel in a special way (e.g. for
    // interpolation) and then can be stored again by OIIO.

    worker<4> ( inp , height , latlon ) ;
  }
  else if ( nchannels == 3 )
  {
    // this is normal RGB input without an alha channel.

    worker<3> ( inp , height , latlon ) ;
  }

  // we offer code for two- and one-channel images, but for two-channel
  // data, it's not quite clear what they might constitute.

  else if ( nchannels == 2 )
  {
    worker<2> ( inp , height , latlon ) ;
  }
  else if ( nchannels == 1 )
  {
    worker<1> ( inp , height , latlon ) ;
  }
}

