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
// 'cubemap' or 'cubemap' format. It captures the environment in
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
// specific sequence and orientations, and this is the convention I use
// in this program. openEXR offers a tool to convert between the two
// types of environment, but I found that it doesn't seem to work
// correctly - see this issue:
// https://github.com/AcademySoftwareFoundation/openexr/issues/1675
// I now think that this may be due to a difference in conception:
// Some documentation I've looked at seems to suggest that the cube
// face images openEXR expects are what I would consider slightly
// wider than ninety degrees: Their outermost pixels coincide with
// the cube's edges, whereas the cube faces I use have their outermost
// pixels half a sample step away from the cube's edges. The openEXR
// way has some merits - e.g. bilinear interpolation is immediately
// possible over the entire cube face - but I won't cater for that
// type of cube face in this program.
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
// like to find a way to feed them directly, but for now I use
// an intermediate image on disk for the task.
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
// because it's a derivative of the openEXR cubemap layout.
// You can store this intermediate in a file called internal.exr
// by setting the global boolean 'save_ir' to true.
// The program has grown since it's inception to provide more code
// on the topic, which I use for now to verify that the conversion
// is correct and to be able to look at intermediate images. The
// central object, the 'sixfold_t', which holds the internal
// representation of the data, might be a good candidate to factor
// out into a separate TU.
// I use some unconventional terminology: 'model space units' are
// coordinates pertaining to 'archetypal' 2D manifolds 'draped'
// in space. The image plane is draped at unit distance forward
// from the origin, and the image points are distributed on it
// so that their other two coordinates coincide with intersections
// of rays pointing at them. Using this scheme makes conversions
// easier and provides a common frame of reference. For spherical
// data, I use the surface of a sphere with unit radius as the
// 'archetypal' 2D manifold, and the image points are located
// where the rays pointing towards them intersect with this sphere.
// This results in the 'draped' image points residing at unit
// distance from the origin. A full spherical image is draped so
// that it's center coincides with the center of the image plane,
// and a rectilinear image is draped in the same way.
// I use lux coordinate system convention: x axis points right,
// y axis points down, z axis points forward.
// 'simdized' values are in SoA format, so a simdized pixel is
// made up from three vectors with LANES elements each.

#include <array>
#include <filesystem>
#include <zimt/zimt.h>
#include <OpenImageIO/texture.h>
#include <OpenImageIO/filesystem.h>

using namespace OIIO ;

// zimt types for 2D and 3D coordinates and pixels

typedef zimt::xel_t < int , 2 > v2i_t ;
typedef zimt::xel_t < int , 3 > v3i_t ;
typedef zimt::xel_t < long , 2 > index_type ;
typedef zimt::xel_t < std::size_t , 2 > shape_type ;

typedef zimt::xel_t < float , 2 > v2_t ;
typedef zimt::xel_t < float , 3 > v3_t ;

// some SIMDized types we'll use. I use 16 SIMD lanes for now,
// which is also the lane count currently supported by OIIO.

#define LANES 16

typedef zimt::simdized_type < float , LANES > f_v ;
typedef zimt::simdized_type < int , LANES > i_v ;
typedef zimt::simdized_type < v2_t ,  LANES > crd2_v ;
typedef zimt::simdized_type < v2i_t , LANES > v2i_v ;
typedef zimt::simdized_type < v3i_t , LANES > v3i_v ;
typedef zimt::simdized_type < v3_t ,  LANES > crd3_v ;
typedef zimt::simdized_type < index_type , LANES > index_v ;

// enum encoding the sequence of cube face images in the cubemap
// This is the sequence used for openEXR cubmap layout. The top
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
// components agree with normal image indexing conventions, namely
// x is to the right and y down. Note that I put the fastest-moving
// index first, which is 'fortran' style, whereas C/C++ use the
// opposite order for nD arrays.

enum { RIGHT , DOWN , FORWARD } ;

// openEXR uses different 3D axis semantics, and if we want to use
// OIIO's environment lookup function, we need openEXR 3D coordinates.

// Here's what the openEXR documentation sys about their axis
// order (next to a drawing which says differently, see this issue:
// https://github.com/AcademySoftwareFoundation/openexr/issues/1687)

// quote:
// We assume that a camera is located at the origin, O, of a 3D
// camera coordinate system. The camera looks along the positive z
// axis. The positive x and y axes correspond to the camera’s left
// and up directions.
// end quote

// so we'd get this axis order, assuming they store x,y,z:

enum { EXR_LEFT , EXR_UP , EXR_FORWARD } ;

// the cubemap comes out right this way, so I assume that their text
// is correct and the drawing is wrong.

// some globals, which will be set via argparse in main

#include <regex>
#include <OpenImageIO/argparse.h>

static bool verbose = false;
static bool help    = false;
static std::string metamatch;
static std::regex field_re;
std::string input , output , save_ir ;
int extent ;
int itp ;

// helper function to save a zimt array to an image file. I am
// working with openEXR images here, hence the HALF data type.
// If the output format can't produce HALF, it will use the
// 'next-best' thing. Note that input and output should agree
// on the colour space. If you stay within one format, that's
// not an issue.

template < std::size_t nchannels >
void save_array ( const std::string & filename ,
                  const zimt::view_t
                    < 2 ,
                      zimt::xel_t < float , nchannels >
                    > & pixels ,
                  bool is_latlon = false )
{
  auto out = ImageOutput::create ( filename );
  assert ( out != nullptr ) ;
  ImageSpec ospec ( pixels.shape[0] , pixels.shape[1] ,
                    nchannels , TypeDesc::HALF ) ;
  out->open ( filename , ospec ) ;

  if ( is_latlon )
    ospec.attribute ( "textureformat" , "LatLong Environment" ) ;

  auto success = out->write_image ( TypeDesc::FLOAT , pixels.data() ) ;
  assert ( success ) ;
  out->close();
}

// coordinate transformations, coded as templates in zimt 'act'
// functor style, returning the result via a reference argument

// code to convert lat/lon to 3D ray. Note that I am using lux
// coordinate convention ('book order': x is right, y down and
// z forward). The lat/lon values coming in are
// angles in radians, and the resulting 'ray' coordinates are
// in 'model space' units. We're using an implicit radius of 1.0
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
  bool discrete90 ;

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
    discrete90 = true ;

    // calculate the diameter in model space units. The diameter
    // for a ninety degree face image would be precisely 2, and
    // if the partial image has larger field of view, it will be
    // larger.

    if ( face_hfov > M_PI_2 )
    {
      diameter = 2.0 * tan ( face_hfov / 2.0 ) ;
      discrete90 = false ;
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
    // We have a boolean 'discrete90' which indicates whether
    // the central 'ninety degrees proper' section coincides
    // with a discrete size.

    if ( face_hfov > M_PI_2 )
    {      
      overscan = tan ( face_hfov / 2.0 ) - 1.0 ;

      // how wide is the overscan, expressed in pixel units?

      double px_overscan = model_to_px * overscan ;

      // truncate to integer to receive the inherent support in
      // pixel units. If this value is as large or larger than the
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

  // variant to get the pick-up coordinate in model space units

  template < typename face_index_t , typename crd_t >
  void get_pickup_coordinate_md ( const face_index_t & face_index ,
                                  const crd_t & in_face_coordinate ,
                                  crd_t & target ) const
  {
    target =    in_face_coordinate // in (-1,1)
              + 1.0                // now in (0,2)
              + ref90 ;            // distance from section UL

    // we add the per-face offset and we're done.

    target[1] += face_index * section_size ;
  }

  // variant of get_pickup_coordinate which yields the pickup
  // coordinate in texture units with extent in [0,1].

  template < typename face_index_t , typename crd_t >
  void get_pickup_coordinate_tx ( const face_index_t & face_index ,
                                  const crd_t & in_face ,
                                  crd_t & target ) const
  {
    target =    in_face // in (-1,1)
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

    target[0] /= float ( outer_width ) ;
    target[1] /= float ( 6 * outer_width ) ;
  }
} ;

// sixfold_t contains data for a six-square sky box with support
// around the cube faces to allow for proper filtering and
// interpolation. The content is set up following the layout
// of openEXR environment maps. cubemaps in openEXR contain six
// square images concatenated vertically. The sequence of the
// images, from top to bottom, is: left, right, top, bottom,
// front, back. The top and bottom images are oriented so that
// they align vertically with the 'back' image (lux aligns with
// the front image) - but note that labels like 'front' are
// somewhat arbitrary - I associate 'front' with the center of
// a full spherical image and 'left' with the first image in the
// cubemap. Note, again, that openEXR's 'own' cubemap format
// uses slightly larger cube faces than what I use here.
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
// texture lookup. As a fall-back, there's an implementation of
// simple bilinear interpolation directly from the IR image.
// The template argument 'nchannels' is the number of channels
// in the image. We accept up to four in main - RGBA should
// come in with associated alpha.

template < int nchannels >
struct sixfold_t
{
  // shorthand for pixels and SIMDized pixels

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

  // pointers to an OIIO texture system and an OIIO texture handle.
  // These are only used if the pick-up is done using OIIO's
  // 'texture' function.

  TextureSystem * ts ;
  TextureSystem::TextureHandle * th ;

  // This array holds all the image data. We mark the array as const,
  // but the data it holds are not const. Later on, in comments, I'll
  // refer to this array as the 'IR image': the internal representation.

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

  // Note how, for now, we don't exploit all the possibilities in
  // the metrics_t object - like cube face images with more than
  // ninety degrees fov - but limit the scope to incoming 'proper'
  // cubemaps with cube face images spanning precisely ninety degrees.

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

  // We can use the fall-back bilinear interpolation to pick up
  // pixel values from the IR image, but to use OIIO's 'texture'
  // function, we need access to the texture system and - for
  // speed - the texture handle. But we don't have these data
  // when the sixfold_t is created - the texture has to be
  // generated first, then stored to disk and then fed to the
  // texture system. So we can only call this function later:

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
    // appropriate slots in the sixfold_t object's 'store' array.

    if ( inp->supports ( "scanlines" ) )
    {
      // if the input is scanline-based, we copy batches of
      // scanlines into the cube face slots in the store

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
      
        // TODO: read_scanlines doesn't work with tile-based files
      
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
    else
    {
      // if the input is not scanline-based, we read the entire image
      // into a buffer, then copy the cube faces to the store. We can
      // use a 3D array as a target, with the six cube faces 'on
      // top of each other' populating the third spatial dimension

      zimt::array_t < 3 , px_t >
        buffer ( { face_width , face_width , 6UL } ) ;
      
      // and a view to the store with the same shape, but strides to
      // match the metrics of the target memory area in the store
        
      zimt::view_t < 3 , px_t >
        target ( p_ul ,
                { 1L , long(outer_width) , long(offset) } ,
                { face_width , face_width , 6UL } ) ;
      
      inp->read_image ( 0 , 0 , 0 , nchannels ,
                        TypeDesc::FLOAT ,
                        buffer.data() ) ;

      // zimt handles the data transfer from the buffer to the view

      target.copy_data ( buffer ) ;
    }
    
    // for ( int face = 0 ; face < 6 ; face++ )
    // {
    //   auto * p_trg = p_ul + face * offset ;
    // 
    //   // for reference: OIIO's read_scanlines' signature
    // 
    //   // virtual bool read_scanlines ( int subimage, int miplevel,
    //   //                               int ybegin, int yend,
    //   //                               int z, int chbegin, int chend,
    //   //                               TypeDesc format, void *data,
    //   //                               stride_t xstride = AutoStride,
    //   //                               stride_t ystride = AutoStride)
    // 
    //   // note how we read face_width scanlines in one go, using
    //   // appropriate strides to place the image data inside the
    //   // larger 'store' array, converting to float as we go along.
    //   // The channels are capped at nchannels. We ask OIIO to
    //   // provide float data.
    // 
    //   // TODO: read_scanlines doesn't work with tile-based files
    // 
    //   auto success =
    //   inp->read_scanlines ( 0 , 0 ,
    //                         face * face_width , (face+1) * face_width ,
    //                         0 , 0 , nchannels ,
    //                         TypeDesc::FLOAT , p_trg ,
    //                         nchannels * 4 ,
    //                         nchannels * 4 * store.strides[1] ) ;
    //   assert ( success ) ;
    // }
  }

  // store_cubemap stores a standard cubemap with cube faces with
  // a field of view of precisely ninety degrees. The store is
  // expected to hold the data ready to export, so discrete90
  // is mandatory - the data aren't gleaned by interpolation,
  // but simply copied out from the central parts of each
  // section.

  void store_cubemap ( const std::string & filename )
  {
    assert ( metrics.discrete90 ) ;

    auto inner_width = outer_width - 2 * frame_width ;
    const int xres = inner_width ;
    const int yres = 6 * inner_width ;

    std::unique_ptr<ImageOutput> out
      = ImageOutput::create ( filename.c_str() ) ;
    assert ( out != nullptr ) ;

    ImageSpec spec ( xres , yres , nchannels , TypeDesc::HALF ) ;
    spec.attribute ( "textureformat" , "CubeFace Environment" ) ;
    out->open ( filename.c_str() , spec ) ;

    auto p_base = store.data() ;
    p_base += ( frame_width * store.strides ) . sum() ;
    auto p_offset = outer_width * outer_width ;

    for ( int face = 0 ; face < 6 ; face++ )
    {
      // virtual bool write_scanlines ( int ybegin , int yend , int z ,
      //                                TypeDesc format ,
      //                                const void * data ,
      //                                stride_t xstride = AutoStride ,
      //                                stride_t ystride = AutoStride )

      auto success =
      out->write_scanlines (   face * inner_width ,
                             ( face + 1 ) * inner_width ,
                               0 ,
                               TypeDesc::FLOAT ,
                               p_base + face * p_offset ,
                               store.strides[0] * nchannels * 4 ,
                               store.strides[1] * nchannels * 4 ) ;

      assert ( success ) ;
    }

    out->close() ;
  }

  // this is coded further down because it needs additional
  // functors. It generates a cubemap with the given width from
  // the data in the internal representation. If the central
  // 90 degree section is contained in a discrete number of
  // pixels and it's extent matches the desired width, the
  // data are simply copied out, otherwise they are generated
  // by interpolation from the IR image - per default by
  // bilinear interpolation.

  void gen_cubemap ( const std::string & filename ,
                     const std::size_t & width ,
                     const int degree = 1 ) ;

  // given a 3D 'ray' coordinate, find the corresponding cube face
  // and the in-face coordinate - note the two references which take
  // the result values. The incoming 'ray' coordinate does not have
  // to be normalized. The resulting in-face coordinates are in the
  // range of [-1,1] - in 'model space units' pertaining to planes
  // 'draped' at unit distance from the origin and perpendicular
  // to one of the axes. This function is coded as pure SIMD code,
  // we don't need it for scalars.

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
  // variants here, the first one using nearest neighbour
  // interpolation, the second bilinear. The first one is currently
  // unused. The pick-up is not guaranteed to look up pixel data
  // strictly inside the 90-degree cube face 'proper' but may
  // glean some information from the support frame, so this has
  // to be present. Initially we provide a one-pixel-wide
  // support frame of mirrored pixels (if necessary - if the
  // incoming partial images have 'inherent support' because
  // they span more than ninety degrees, this is not necessary)
  // which is enough for bilinear interpolation. Once we have
  // filled in the support frame, we can use interpolators with
  // wider support.

  void cubemap_to_pixel ( const i_v & face ,
                          const crd2_v & in_face ,
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
      px_v px2, help ;

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

  // this is the equivalent function to 'cubemap_to_pixel', above,
  // using OIIO's 'texture' function to perform the gleaning of
  // pixel data from the IR image. The IR image is not accessed
  // directly, but provided via OIIO's texture system.
  // On top of the pick-up coordinate (coming in in texture
  // units in [0,1], rather than pixel units which are used in
  // 'cubemap_to_pixel') we have the four derivatives and the
  // OIIO texture batch options.

  void get_filtered_px ( const crd2_v & pickup ,
                         px_v & px ,
                         const f_v & dsdx ,
                         const f_v & dtdx ,
                         const f_v & dsdy ,
                         const f_v & dtdy ,
                         TextureOptBatch & batch_options ) const
  {
    // code to truncate the pickup coordinate to int and gather,
    // after restoring the pickup to pixel units

    // pickup[0] *= float ( store.shape[0] ) ;
    // pickup[1] *= float ( store.shape[1] ) ;
    // index_v idx { pickup[0] , pickup[1] } ;
    // const auto ofs = ( idx * store.strides ) . sum() * nchannels ;
    // const auto * p = (float*) ( store.data() ) ;
    // 
    // px.gather ( p , ofs ) ;
    // return ;

    // OIIO's 'texture' signature is quite a mouthful:

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

    #if defined USE_VC or defined USE_STDSIMD

    // to interface with zimt's Vc and std::simd backends, we need to
    // extract the data from the SIMDized objects and re-package the
    // ouptut as a SIMDized object. The compiler will likely optimize
    // this away and work the entire operation in registers, so let's
    // call this a 'semantic manoevre'.

    float scratch [ 6 * LANES + nchannels * LANES ] ;

    pickup.store ( scratch ) ; // stores 2 * LANES
    dsdx.store ( scratch + 2 * LANES ) ;
    dtdx.store ( scratch + 3 * LANES ) ;
    dsdy.store ( scratch + 4 * LANES ) ;
    dtdy.store ( scratch + 5 * LANES ) ;

    bool result =
    ts->texture ( th , nullptr , batch_options , Tex::RunMaskOn ,
                  scratch , scratch + LANES ,
                  scratch + 2 * LANES , scratch + 3 * LANES ,
                  scratch + 4 * LANES , scratch + 5 * LANES ,
                  nchannels , scratch + 6 * LANES ) ;

    assert ( result ) ;
    px.load ( scratch + 6 * LANES ) ;

    #else

    // zimt's own and the highway backend have a representation as
    // a C vector of fundamentals and provide a 'data' function
    // to yield it's address. This simplifies matters, we can pass
    // these pointers to OIIO directly.

    bool result =
    ts->texture ( th , nullptr , batch_options , Tex::RunMaskOn ,
                  pickup[0].data() , pickup[1].data() ,
                  dsdx.data() , dtdx.data() ,
                  dsdy.data() , dtdy.data() ,
                  nchannels , (float*) ( px[0].data() ) ) ;

    #endif
}

  // variant taking derivatives of the in-face coordinate, which
  // are approximated by calculating the difference to a canonical
  // (target image) coordinate one sample step to the right (x)
  // or below (y), respectively. The derivatives are in texture
  // units aready, and we also convert the pickup coordinate to
  // texture units.

  void cubemap_to_pixel ( const i_v & face ,
                          crd2_v in_face ,
                          px_v & px ,
                          const f_v & dsdx ,
                          const f_v & dtdx ,
                          const f_v & dsdy ,
                          const f_v & dtdy ,
                          TextureOptBatch & bo ) const
  {
    crd2_v pickup ;

    // obtain the pickup coordinate in texture units (in [0,1])

    metrics.get_pickup_coordinate_tx ( face , in_face , pickup ) ;

    // use OIIO to get the pixel value

    get_filtered_px ( pickup , px , dsdx , dtdx , dsdy , dtdy , bo) ;
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
  void eval ( const I & crd2 , O & crd3 ) const
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
  void eval ( const I & _crd2 , O & crd3 ) const
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

// to make the conversion efficient and transparent, I refrain from
// using ir_to_ray_gen and a subsequent coordinate transformation
// in favour of this dedicated functor, which is basically a copy of
// the one above, but with different component indexes and signs
// inverted where necessary (namely for the left and up direction,
// which are the negative of zimt's right and down).

template < int nchannels >
struct ir_to_exr_gen
: public zimt::unary_functor < v2_t , v3_t , LANES >
{
  const sixfold_t < nchannels > & sf ;

  ir_to_exr_gen ( const sixfold_t < nchannels > & _sf )
  : sf ( _sf )
  { }

  // incoming, we have 2D model space coordinates, with the origin
  // at the (total!) IR image's center.

  template < typename I , typename O >
  void eval ( const I & _crd2 , O & crd3 ) const
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
        crd3[EXR_LEFT](m) =    - 1.0f ;
        crd3[EXR_UP](m) =      - crd2[DOWN] ;
        crd3[EXR_FORWARD](m) = - crd2[RIGHT] ;
      }
      m = ( section == CM_LEFT ) ;
      if ( any_of ( m ) )
      {
        crd3[EXR_LEFT](m) =      1.0f ;
        crd3[EXR_UP](m) =      - crd2[DOWN] ;
        crd3[EXR_FORWARD](m) =   crd2[RIGHT] ;
      }
    }
    if ( any_of ( dom == 1 ) )
    {
      auto m = ( section == CM_BOTTOM ) ;
      if ( any_of ( m ) )
      {
        crd3[EXR_LEFT](m) =      crd2[RIGHT] ;
        crd3[EXR_UP](m) =      - 1.0f ;
        crd3[EXR_FORWARD](m) =   crd2[DOWN] ;
      }
      m = ( section == CM_TOP ) ;
      if ( any_of ( m ) )
      {
        crd3[EXR_LEFT](m) =      crd2[RIGHT] ;
        crd3[EXR_UP](m) =        1.0f ;
        crd3[EXR_FORWARD](m) = - crd2[DOWN] ;
      }
    }
    if ( any_of ( dom == 2 ) )
    {
      auto m = ( section == CM_FRONT ) ;
      if ( any_of ( m ) )
      {
        crd3[EXR_LEFT](m)   = - crd2[RIGHT] ;
        crd3[EXR_UP](m)    =  - crd2[DOWN] ;
        crd3[EXR_FORWARD](m) =  1.0f ;
      }
      m = ( section == CM_BACK ) ;
      if ( any_of ( m ) )
      {
        crd3[EXR_LEFT](m)   =    crd2[RIGHT] ;
        crd3[EXR_UP](m)    =   - crd2[DOWN] ;
        crd3[EXR_FORWARD](m) = - 1.0f ;
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
    
    // now we fill in the lower and upper limits. This is a good
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

// this functor takes float 2D in-face coordinates and yields pixels.

template < int nchannels >
struct fill_ir_t
: public zimt::unary_functor
    < v2_t , zimt::xel_t < float , nchannels > , LANES >
{
  const sixfold_t < nchannels > & sf ;
  const int face ;
  const int degree ;

  typedef zimt::xel_t < float , nchannels > px_t ;
  typedef zimt::simdized_type < px_t , LANES > px_v ;

  fill_ir_t ( const sixfold_t < nchannels > & _cubemap ,
              const int & _face ,
              const int & _degree )
  : sf ( _cubemap ) ,
    face ( _face ) ,
    degree ( _degree )
  {
    assert ( degree == 0 || degree == 1 ) ;
  }

  template < typename I , typename O >
  void eval ( const I & crd2 , O & px ) const
  {
    sf.cubemap_to_pixel ( face , crd2 , px , degree ) ;
  }
} ;

template < int nchannels >
void sixfold_t < nchannels > :: gen_cubemap
  ( const std::string & filename ,
    const std::size_t & width ,
    const int degree )
{
  // if the IR already has the data handy, we use store_cubemap

  if (    metrics.discrete90
       && ( width == outer_width - 2 * frame_width ) )
  {
    store_cubemap ( filename ) ;
    return ;
  }

  // otherwise we have to calculate the data by interpolation

  const int xres = width ;
  const int yres = 6 * width ;

  std::unique_ptr<ImageOutput> out
    = ImageOutput::create ( filename.c_str() ) ;

  if ( ! out )
    return ;

  ImageSpec spec ( xres , yres , nchannels , TypeDesc::HALF ) ;
  spec.attribute ( "textureformat" , "CubeFace Environment" ) ;
  out->open ( filename.c_str() , spec ) ;

  // set up an array for a single cube face

  zimt::array_t < 2 , px_t > section ( { width , width } ) ;

  for ( int face = 0 ; face < 6 ; face++ )
  {
    // set up the sample step width and the starting point

    double delta = 2.0 / double ( width ) ;
    double start = -1.0 + delta / 2.0 ;

    // set up a linspace as data source and a storer as data sink

    zimt::linspace_t < float , 2 , 2 , LANES > ls ( start , delta ) ;
    zimt::storer < float , nchannels , 2 , LANES > st ( section ) ;

    // set up the act functor. We have incoming in-face coordinates
    // and a fixed face index, and we have fill_ir_t just for that.
    // We do the filling-in with bilinear interpolation, which is
    // perfectly adequate for the purpose: the data we generate are
    // only support, they won't make it into the output.

    fill_ir_t fill_ir ( *this , face , degree ) ;

    // fill the 'section' array with pixel data

    zimt::process ( section.shape , ls , fill_ir , st ) ;

    // write the cube face to the output file

    // virtual bool write_scanlines ( int ybegin , int yend , int z ,
    //                                TypeDesc format ,
    //                                const void * data ,
    //                                stride_t xstride = AutoStride ,
    //                                stride_t ystride = AutoStride )

    auto success = out->write_scanlines ( face * width ,
                                          ( face + 1 ) * width ,
                                          0 ,
                                          TypeDesc::FLOAT ,
                                          section.data() ) ;

    assert ( success ) ;
  }

  out->close() ;
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
  // OIIO 'texture' function expects an lvalue.

  TextureOptBatch & batch_options ;

  // source of pixel data

  const sixfold_t < nchannels > & cubemap ;

  // parameter to choose the interpolator

  const int degree ;

  // sampling step

  v2_t delta ;

  // scaling factor to move from model space units to texture units
  // (separate for the s and t direction - these factors are applied
  // to coordinates pertaining to the IR image of the cubemap, which
  // has 1:6 aspect ratio)

  const float scale_s ;
  const float scale_t ;


  // ll_to_px_t's c'tor obtains a const reference to the sixfold_t
  // object holding pixel data, the degree of the interpolator and
  // the delta of the sampling step, to form derivatives.

  ll_to_px_t ( const sixfold_t < nchannels > & _cubemap ,
               const int & _degree ,
               const v2_t & _delta ,
               TextureOptBatch & _batch_options )
  : cubemap ( _cubemap ) ,
    degree ( _degree ) ,
    delta ( _delta ) ,
    batch_options ( _batch_options ) ,
    scale_s ( _cubemap.model_to_px / _cubemap.store.shape[0] ) ,
    scale_t ( _cubemap.model_to_px / -cubemap.store.shape[1] ) ,
    ll_to_ray() // g++ is picky.
  { }

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
      // degree == -1 stands for "use OIIO's 'texture' function".
      // for this interpolation, we need the derivatives, in
      // pixel coordinates pertaining to the IR image. We use
      // the same approach as in cubemap.cc: we obtain source
      // image coordinates for two points, each one step in
      // one of the canonical directions away. Then we subtract
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
      // handle separately, and we scale to texture coordinates.

      auto dsdx = ( dx1_if[0] - in_face[0] ) * scale_s ;
      auto dtdx = ( dx1_if[1] - in_face[1] ) * scale_t ;

      // we repeat the process for a coordinate one sample step away
      // along the vertical axis

      crd2_v dy1 = lat_lon ;
      dy1[1] += delta[1] ;
      crd3_v dy1_3 ;
      ll_to_ray.eval ( dy1 , dy1_3 ) ;
      crd2_v dy1_if ;
      cubemap.ray_to_cubeface_fixed ( dy1_3 , face , dy1_if ) ;

      auto dsdy = ( dy1_if[0] - in_face[0] ) * scale_s ;
      auto dtdy = ( dy1_if[1] - in_face[1] ) * scale_t ;

      // now we can call the cubemap_to_pixel variant which is based
      // on OIIO's texture lookup and takes derivatives

      cubemap.cubemap_to_pixel ( face , in_face , px ,
                                 dsdx , dtdx , dsdy , dtdy ,
                                 batch_options ) ;
    }
    else
    {
      // use 'face' and 'in_face' to obtain pixel values directly
      // from the IR image with NN (degree 0) or bilinear (degree 1)
      // interpolation.

      cubemap.cubemap_to_pixel ( face , in_face , px , degree ) ;
    }
  }

} ;

// This functor 'looks at' a full spherical image. It receives
// 2D lat/lon coordinates and yields pixel values. Pick-up is done
// with bilinear interpolation. This is not needed for this program,
// unless we 'regenerate' the cubemap from the lat/lon image we
// produce. We might extend this program and use OIIO's function
// 'environment' to do the pickup via OIIO's texture system. Then,
// we could 'swallow' cubemap.cc and provide both conversions
// in this program.

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

    index_v idsdxl { uli[0] , uli[1] } ;
    auto ofs = ( idsdxl * latlon.strides ) . sum() * nchannels ;
    px_v pxul ;
    pxul.gather ( p , ofs ) ;

    index_v idsdxr { uri[0] , uri[1] } ;
    ofs = ( idsdxr * latlon.strides ) . sum() * nchannels ;
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
// This function is only used if the cubemap is 'regenerated' from
// the lat/lon image we have just created, or if the cubemap is
// made from a lat/lon image passed in as a file.

template < int nchannels >
void latlon_to_ir ( const zimt::view_t < 2 , pix_t<nchannels> > & latlon ,
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

// next we have a functor converting discrete cubemap coordinates
// (so, pixel units, starting at (0,0) for the upper left corner)
// to pixel values gleaned from an openEXR lat/lon environment
// map, using OIIO's 'environment' function. The environment is
// introduced via it's texture handle th referring to it's internal
// rfpresentation in the texture system ts. The calling code can
// set the batch options to influence the rendition - with the
// default options, OIIO uses a quite intense antialiasing filter
// which removes high frequency content, giving the result a
// slightly blurred appearance.

template < std::size_t nchannels >
struct eval_env
: public zimt::unary_functor
   < v2i_t , zimt::xel_t < float , nchannels > , LANES >
{
  TextureSystem * ts ;
  TextureOptBatch & batch_options ;
  TextureSystem::TextureHandle * th ;
  int width ;
  double px2_to_model ;
  const sixfold_t < nchannels > & sf ;
  ir_to_exr_gen < nchannels > ir_to_exr ;

  // pull in the c'tor arguments

  eval_env ( TextureSystem * _ts ,
             TextureOptBatch & _batch_options ,
             TextureSystem::TextureHandle * _th ,
             const sixfold_t < nchannels > & _sf )
  : ts ( _ts ) ,
    batch_options ( _batch_options ) ,
    th ( _th ) ,
    width ( int ( _sf.outer_width ) ) ,
    px2_to_model ( _sf.px_to_model * 0.5 ) ,
    sf ( _sf ) ,
    ir_to_exr ( _sf )
   { }

  // set up the eval function.

  template < typename I , typename O >
  void eval ( const I & crd2 , O & px ) const
  {
    // Incoming, we have 2D discrete (!) coordinates pertaining
    // to the IR image. We convert them to 3D discrete coordinates
    // with doubled (!) value at appropriate distance. this way,
    // we avoid working in float for the coordinate calculations;
    // the floating point values we'll generate later will be
    // precise, rather than multiples of a float delta.

    // v3i_v crdi3 { - ( 2 * crd2[0] - ( width - 1 ) ) ,
    //               - ( 2 * crd2[1] - ( 6 * width - 1 ) ) ,
    //               i_v ( 2 * width ) } ;

    v3i_v crdi3 { 2 * crd2[0] - ( width - 1 ) ,
                  2 * crd2[1] - ( 6 * width - 1 ) ,
                  i_v ( 2 * width ) } ;

    // to get the right and lower neighbour, we add two (!) to the
    // appropriate component (we're working in doubled coordinates!)

    auto crdi3_x1 = crdi3 ;
    crdi3_x1[0] += 2 ;

    auto crdi3_y1 = crdi3 ;
    crdi3_y1[1] += 2 ;

    // now we move to floating point and model space units, note
    // the factor px2_to_model, which also takes care of halving
    // the doubled coordinates

    crd3_v p00 ( crdi3 ) ;
    p00 *= px2_to_model ;

    crd3_v p10 ( crdi3_x1 ) ;
    p10 *= px2_to_model ;

    crd3_v p01 ( crdi3_y1 ) ;
    p01 *= px2_to_model ;

    // now we obtain ray coordinates from model space coordinates.
    // We have a ready-made functor for the purpose already set up:
    
    crd3_v p00r , p10r , p01r ;
    
    ir_to_exr.eval ( p00 , p00r ) ;
    ir_to_exr.eval ( p10 , p10r ) ;
    ir_to_exr.eval ( p01 , p01r ) ;

    // with the ray coordinates for the current coordinate and it's
    // two neighbours, we can obtain a reasonable approximation of
    // the derivatives in canonical x and y direction by forming the
    // difference

    auto ds = p10r - p00r ;
    auto dt = p01r - p00r ;

    // now we can call 'environment', but depending on the SIMD
    // back-end, we provide the pointers which 'environemnt' needs
    // in different ways. The first form would in fact work for all
    // back-ends, but the second form is more concise:

#if defined USE_VC or defined USE_STDSIMD

    // to interface with zimt's Vc and std::simd backends, we need to
    // extract the data from the SIMDized objects and re-package the
    // ouptut as a SIMDized object. The compiler will likely optimize
    // this away and work the entire operation in registers, so let's
    // call this a 'semantic manoevre'.

    float scratch [ 4 * nchannels * LANES ] ;

    p00r.store ( scratch ) ;
    ds.store ( scratch + nchannels * LANES ) ;
    dt.store ( scratch + nchannels * LANES ) ;

    ts->environment ( th , nullptr, batch_options ,
                      Tex::RunMaskOn ,
                      scratch ,
                      scratch + nchannels * LANES ,
                      scratch + 2 * nchannels * LANES ,
                      nchannels ,
                      scratch + 3 * nchannels * LANES ) ;

    px.load ( scratch + 3 * nchannels * LANES ) ;

#else

    // the highway and zimt's own backend have an internal representation
    // as a C vector of fundamentals, so we van use data() on them, making
    // the code even simpler - though the code above would work just the
    // same.

    ts->environment ( th , nullptr, batch_options ,
                      Tex::RunMaskOn ,
                      p00r[0].data() ,
                      ds[0].data() ,
                      dt[0].data() ,
                      nchannels ,
                      px[0].data() ) ;

#endif

    // and that's us done! the 'environment' function has returned pixel
    // values, which have been passed as result to the zimt::transform
    // process invoking this functor.
  }
} ;

// cubemap_to_latlon converts a cubemap to a lat/lon environment
// map, a.k.a full spherical panorama.

template < int nchannels >
void cubemap_to_latlon ( std::unique_ptr < ImageInput > & inp ,
                         std::size_t height ,
                         const std::string & latlon ,
                         int degree )
{
  // we expect an image with 1:6 aspect ratio. We don't check
  // for metadata specific to environments - it can be anything,
  // but the aspect ratio must be correct.

  const ImageSpec &spec = inp->spec() ;
  int xres = spec.width ;
  int yres = spec.height ;

  if ( verbose )
    std::cout << "cube face width: " << spec.width << std::endl ;

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
  // cube face image spans more than ninety degrees. Note that
  // the possible contribution of the pixels provided by
  // mirroring is very small, and even quite negligible with
  // 'normal' face widths. But if the value at that position
  // weren't initialized and it would enter the interpolation,
  // there's no telling what might happen - if it were very
  // large, even a small contribution would spoil the result.
  // Hence the mirroring-around. With it, we can rest assured
  // that the result will be near-perfect, and with the
  // subsequent support-gleaning runs we're well on the safe
  // side. A sloppy approach would be to initialize the area
  // with, say, medium grey.

  if ( sf.metrics.inherent_support == 0 )
    sf.mirror_around() ;

  // to refine the result, we generate support with bilinear
  // interpolation. We assume that the face width will not be
  // 'very small' and the errors from using the mirrored pixels
  // will be negligible.

  sf.fill_support ( 1 ) ;

  // to load the texture with OIIO's texture system code, it
  // has to be in a file, so we store it to a temporary file:

  auto temp_path = std::filesystem::temp_directory_path() ;
  auto temp_filename = temp_path / "temp_texture.exr" ;

  if ( degree == -1 )
  {
    if ( verbose )
      std::cout << "saving generated texture to " << temp_filename.c_str()
                << std::endl ;

    save_array ( temp_filename.c_str() , sf.store ) ;

    // now we can introduce it to the texture system and receive
    // a texture handle for fast access

    TextureSystem * ts = TextureSystem::create() ;
    ustring uenvironment ( temp_filename.c_str() ) ;
    auto th = ts->get_texture_handle ( uenvironment ) ;

    sf.set_ts ( ts , th ) ;
  }

  // with proper support near the edges, we can now run the
  // actual payload code - the conversion to lon/lat - with
  // bilinear interpolation, which is a step up from the
  // nearest-neighbour interpolation we had until now and
  // is good enough if the resolution of input and output
  // don't differ too much. But the substrate we have
  // generated is also good for mip-mapping, so we have the
  // option to switch to interpolation methods with larger
  // support, like OIIO's texture access.
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
  // It might be a good idea to store the entire internal
  // representation - cubefaces plus support - to an image file
  // for faster access, avoiding the production of the support
  // area from outher cube faces, at the cost of saving a few
  // extra pixels. The process as it stands now seems to produce
  // support which is just as good as support which is created
  // by rendering cube face images with slightly more than ninety
  // degrees (as it can be done in lux), so we can process the
  // 'orthodox' format and yet avoid it's shortcomings. For now
  // I'll stick to using the the cubemap format with cube face
  // images covering precisely ninety degrees.

  // time to do the remaining work.

  // The target array will receive the output pixel data

  typedef zimt::xel_t < float , nchannels > px_t ;

  zimt::array_t < 2, px_t > trg ( { 2 * height , height } ) ;

  // we directly provide input in 'model space units' with a
  // zimt linspace_t object, rather than feeding discrete coordinates
  // and scaling and shifting them, which would work just the same.

  // set up a linspace_t over the lon/lat sample points as get_t
  // (a.k.a input generator). d is the step width from one sample
  // to the next:

  double d = M_PI / height ;

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

  TextureOptBatch batch_options ;

  // TextureOptBatch's c'tor does not initialize these members, hence:

  for ( int i = 0 ; i < 16 ; i++ )
    batch_options.swidth[i] = batch_options.twidth[i] = 1 ;
  
  for ( int i = 0 ; i < 16 ; i++ )
    batch_options.sblur[i] = batch_options.tblur[i] = 0 ;

  // set up ll_to_px_t using bilinear interpolation (argument 1)
  // or argument -1 to use pick-up with OIIO's 'texture' function.
  // The value is taken from the global 'degree' which is set with
  // the CL parameter of the same name.
  // This is the functor which takes lon/lat coordinates and produces
  // pixel data from the internal representation of the cubemap held
  // in the sixfold_t object.

  ll_to_px_t<nchannels> act ( sf , degree , step , batch_options ) ;

  // showtime! call zimt::process.

  zimt::process ( trg.shape , linspace , act , st ) ;

  // we don't need the temporary texture any more. If save_ir is
  // set, we save it under the given name.
  if ( save_ir != std::string() )
  {
    if ( verbose )
      std::cout << "saving internal representation to '" << save_ir
                << "'" << std::endl ;
    if ( degree == -1 )
      std::filesystem::rename ( temp_filename , save_ir ) ;
    else
      save_array ( save_ir , sf.store ) ;
  }
  else if ( degree == -1 )
  {
    if ( verbose )
      std::cout << "removing temporary texture file "
                << temp_filename.c_str() << std::endl ;
    std::filesystem::remove ( temp_filename ) ;
  }

  // finally we store the data to an image file - note how we have
  // float data in 'trg', and OIIO will convert these on-the-fly to
  // HALF, as specified in the write_image invocation.
  // Note that the target will receive image data in the same colour
  // space as the input. If you feed, e.g. openEXR, and store to JPEG,
  // the image will look too dark, because the linear RGB data are
  // stored as if they were sRGB.
  // the output is a lat/lon environment with openEXR image order and
  // orientation, so we set the textureformat tag, but TODO: there may
  // be slight differences in the precise format of the individual
  // cube faces - the cube faces we store here are precisely ninety
  // degrees from the left edge of the leftmost pixel to the right
  // edge of the rightmost pixel (and alike for top/bottom), whereas
  // the openEXR spec may measure the ninety degrees from the pixel
  // centers, so the output would not conform to their spec precisely

  if ( verbose )
    std::cout << "saving lat/lon environment map to '"
              << latlon << std::endl ;

  save_array<nchannels> ( latlon , trg , true ) ;
}

// do the reverse operation. The first variant below uses bilinear
// interpolation directly on the IR image. This is fast and quite
// accurate, but if there is a large differnce in scale, it will
// produce aliasing artifacts. The version using OIIO's anisotropic
// antialiasing filer is one further down.

template < int nchannels >
void _latlon_to_cubemap ( const std::string & latlon ,
                          std::size_t width ,
                          const std::string & cubemap )
{
  typedef zimt::xel_t < float , nchannels > px_t ;

  auto inp = ImageInput::open ( latlon ) ;
  assert ( inp != nullptr ) ;

  const ImageSpec &spec = inp->spec() ;
  std::size_t w = spec.width ;
  std::size_t h = spec.height ;
  assert ( w == h * 2 ) ;

  zimt::array_t < 2 , px_t > src ( { w , h } ) ;
  
  bool success = inp->read_image ( 0 , 0 , 0 , nchannels ,
                                   TypeDesc::FLOAT , src.data() ) ;

  assert ( success ) ;

  sixfold_t<nchannels> sf ( width ) ;

  latlon_to_ir < nchannels > ( src , sf ) ;

  sf.store_cubemap ( cubemap ) ;
}

// convert a lat/lon environment map into a cubemap. The 'width'
// parameter sets the width of the cubemap's cube face images, it's
// height is six times that. The 'degree' parameter selects which
// interpolator us ised: degree 1 uses direct bilinear interpolation,
// and degree -1 uses OIIO's anisotropic antialiasing filter.

template < int nchannels >
void latlon_to_cubemap ( const std::string & latlon ,
                         std::size_t width ,
                         const std::string & cubemap ,
                         int degree )
{
  sixfold_t<nchannels> sf ( width ) ;

  if ( degree == 1 )
  {
    typedef zimt::xel_t < float , nchannels > px_t ;

    auto inp = ImageInput::open ( latlon ) ;
    assert ( inp != nullptr ) ;

    const ImageSpec &spec = inp->spec() ;
    std::size_t w = spec.width ;
    std::size_t h = spec.height ;
    assert ( w == h * 2 ) ;

    zimt::array_t < 2 , px_t > src ( { w , h } ) ;
    
    bool success = inp->read_image ( 0 , 0 , 0 , nchannels ,
                                    TypeDesc::FLOAT , src.data() ) ;

    assert ( success ) ;

    latlon_to_ir < nchannels > ( src , sf ) ;
  }
  else
  {
    TextureSystem * ts = TextureSystem::create() ;
    ustring uenvironment ( latlon.c_str() ) ;
    auto th = ts->get_texture_handle ( uenvironment ) ;
    TextureOptBatch batch_options ;

    // TextureOptBatch's c'tor does not initialize these members, hence:

    for ( int i = 0 ; i < 16 ; i++ )
      batch_options.swidth[i] = batch_options.twidth[i] = 1 ;
    
    for ( int i = 0 ; i < 16 ; i++ )
      batch_options.sblur[i] = batch_options.tblur[i] = 0 ;

    // we have a dedicated functor going all the way from discrete
    // cubemap image coordinates to pixel values:

    assert ( ts != nullptr ) ;
    assert ( th != nullptr ) ;

    eval_env < nchannels > act ( ts , batch_options , th , sf ) ;

    // showtime! notice the call signature: we pass no source, because
    // we want discrete coordinates to start from. Omitting the source
    // parameter does just that: the input to the act functor will be
    // discrete coordinates of the target location for which the functor
    // is supposed to calculate content.

    zimt::transform ( act , sf.store ) ;
  }

  // the result of the zimt::transform is slightly more than we need,
  // namely the entire sixfold_t object's IR image. We only store the
  // 'cubemap proper':

  if ( verbose )
    std::cout << "saving cubemap to '" << cubemap << "'" << std::endl ;

  sf.store_cubemap ( cubemap ) ;

  if ( save_ir != std::string() )
  {
    if ( verbose )
      std::cout << "saving internal representation to '" << save_ir
                << "'" << std::endl ;

    save_array ( save_ir , sf.store ) ;
  }
}

int main ( int argc , const char ** argv )
{
  // we're using OIIO's argparse, since we're using OIIO anyway.
  // This is a convenient way to glean arguments on all supported
  // platforms - getopt isn't available everywhere.

  Filesystem::convert_native_arguments(argc, (const char**)argv);
  ArgParse ap;
  ap.intro("envutil -- convert between lat/lon and cubemap format\n")
    .usage("envutil [options]");
  ap.arg("-v", &verbose)
    .help("Verbose output");
  ap.arg("--input INPUT")
    .help("input file name (mandatory)")
    .metavar("INPUT");
  ap.arg("--output OUTPUT")
    .help("output file name (mandatory)")
    .metavar("OUTPUT");
  ap.arg("--save_ir INTERNAL")
    .help("save IR image to this file")
    .metavar("INTERNAL");
  ap.arg("--extent EXTENT")
    .help("width of the cubemap / height of the envmap")
    .metavar("EXTENT");
  ap.arg("--itp ITP")
    .help("interpolator: 1 for direct bilinear, -1 for OIIO's anisotropic")
    .metavar("ITP");
    
  if (ap.parse(argc, argv) < 0 ) {
      std::cerr << ap.geterror() << std::endl;
      ap.print_help();
      return help ? EXIT_SUCCESS : EXIT_FAILURE ;
  }

  if (!metamatch.empty()) {
      field_re.assign(metamatch, std::regex_constants::extended
                                      | std::regex_constants::icase);
  }
  
  // extract the CL arguments from the argument parser

  input = ap["input"].as_string("");
  output = ap["output"].as_string("");
  save_ir = ap["save_ir"].as_string("");
  extent = ap["extent"].get<int>(0);
  itp = ap["itp"].get<int>(-1);

  assert ( input != std::string() ) ;
  assert ( output != std::string() ) ;

  if ( verbose )
  {    
    std::cout << "input: " << input << std::endl ;
    std::cout << "output: " << output << std::endl ;
    std::cout << "interpolation: "
              << ( itp == 1 ? "direct bilinear" : "OIIO anisotropic" )
              << std::endl ;
  }

  auto inp = ImageInput::open ( input ) ;

  assert ( inp ) ;

  const ImageSpec &spec = inp->spec() ;
  std::size_t w = spec.width ;
  std::size_t h = spec.height ;
  int nchannels = spec.nchannels ;

  if ( verbose )
    std::cout << "input has " << nchannels << " channels" << std::endl ;

  if ( w == 2 * h )
  {
    inp->close() ;

    if ( verbose )
      std::cout << "input has 2:1 aspect ratio, assuming latlon"
                << std::endl ;

    if ( extent == 0 )
    {
      double e = h * 2.0 / M_PI ;
      extent = e ;
      if ( extent % 64 )
        extent = ( ( extent / 64 ) + 1 ) * 64 ;
      if ( verbose )
        std::cout << "no extent given, using " << extent << std::endl ;
    }

    if ( nchannels >= 4 )
    {
      latlon_to_cubemap<4> ( input , extent , output , itp ) ;
    }
    else if ( nchannels == 3 )
    {
      latlon_to_cubemap<3> ( input , extent , output , itp ) ;
    }
    else if ( nchannels == 1 )
    {
      latlon_to_cubemap<1> ( input , extent , output , itp ) ;
    }
    else
    {
      std::cerr << "input format error: need 1,3 or >=4 channels"
                << std::endl ;
      exit ( EXIT_FAILURE ) ;
    }
  }
  else if ( h == 6 * w )
  {
    if ( verbose )
      std::cout << "input has 1:6 aspect ratio, assuming cubemap"
                << std::endl ;

    if ( extent == 0 )
    {
      extent = 2 * w ;
      if ( extent % 64 )
        extent = ( ( extent / 64 ) + 1 ) * 64 ;
      if ( verbose )
        std::cout << "no extent given, using " << extent << std::endl ;
    }

    if ( nchannels >= 4 )
    {
      cubemap_to_latlon<4> ( inp , extent , output , itp ) ;
    }
    else if ( nchannels == 3 )
    {
      cubemap_to_latlon<3> ( inp , extent , output , itp ) ;
    }
    else if ( nchannels == 1 )
    {
      cubemap_to_latlon<1> ( inp , extent , output , itp ) ;
    }
    else
    {
      std::cerr << "input format error: need 1,3 or >=4 channels"
                << std::endl ;
      exit ( EXIT_FAILURE ) ;
    }
  }
  else
  {
    std::cerr << "input format error: need lat/lon or cubemap input" << std::endl ;
    exit ( EXIT_FAILURE ) ;
  }
  if ( verbose )
    std::cout << "conversion complete. exiting." << std::endl ;
  exit ( EXIT_SUCCESS ) ;
}

