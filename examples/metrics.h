#include <zimt/xel.h>

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
  // the metrics_t object describes the geometry of the internal
  // representation (the 'IR image') of a cubemap as we use it
  // in the zimt examples dealing with cubemaps, and this structure
  // is also intended for use in lux.
  // we state a few preliminary rules: the square shapes which
  // constitute the geometrical object we describe are symmetric
  // both in the horizontal and the vertical, and each of the
  // six square 'sections' which are 'stacked' vertically in
  // the whole IR image are also symmetrical. The sections have
  // three distinct features: their total extent, the extent of
  // the image found in the input, and the extent of the central
  // section representing a 'cube face proper', which we refer to
  // as the 'core'. The first two are discrete values, the third
  // one may be discrete, so it's not guaranteed that the core has
  // discrete extent, but if it does, we set a flag (discrete90).
  // We add one more requirement: the cube face images must have
  // even width. This is to enforce perfect symmetry: the extent
  // of the entire IR image is a multiple of the tile size which,
  // in turn, must be a power of two. If we were to allow odd
  // sizes for the cube face images, their center would not be at
  // the same distance from the edges of the IR image they are
  // embedded in. We may be able to relax this requirement later
  // on, but as it stands, we rely on it.

  // we use variable names ending in _width to indicate discrete
  // values in pixel units.

  // the face width is the width (and height, both must agree)
  // of all the six square images in the input. This is a given
  // value which is received by the c'tor

  const std::size_t face_width ;

  // the face field of view is the field of view which the square
  // images correspond to. This is also a given, passed to the c'tor.
  // the field of view is understood as the angle from the left edge
  // of the leftmost pixel to the right edge of the rightmost pixel,
  // understanding pixels as small square areas of uniform colour
  // which have a size of one pixel unit. This indicates that if
  // the incoming images have their outermost pixels coinciding
  // precisely with the virtual cube's edges, they are slightly
  // larger than ninety degrees if we stick to the notion of the
  // field of view given above.

  const double face_fov ;

  // outer_fov is the field of view corresponding to an entire section,
  // with the same pixel-edge to pixel-edge semantics as explained
  // in the initial comment. It is not a given, but calculated during
  // set-up.
  
  double outer_fov ;

  // the width of a section must be a multiple of the tile width,
  // so we have section_width == n_tiles * tile_width. The tile
  // width is also a given, passed to the c'tor, but n_tiles and
  // section_width are calculated during set-up.

  const std::size_t tile_width ;

  std::size_t n_tiles ;
  std::size_t section_width ;
  std::size_t frame_width ;

  // the minimal support, in pixel units, is the amount of pixels
  // we require as addional 'headroom' around the central part which
  // covers the 'cube face proper' covering precisely ninety degrees.
  // To make a clear definition: It's the number of pixels from the
  // margin inwards which are wholly outside the 'cube face proper'.
  // The support is the difference between the section width and the
  // core size truncated to integer. support_min is also a given,
  // passed to the c'tor

  const std::size_t support_min ;

  // this value gives the number of pixels in the cube face image,
  // going inwards from the edge, which are entirely outside the
  // core.

  std::size_t inherent_support ;

  // in the metrics_t object, we hold two scaling factors, which
  // can be used to translate between pixel units and 'model space
  // units'. 'model space units' refer to the IR image 'draped'
  // on a plane at unit distance, where the sample points
  // are laid out so that the 3D viewing rays they concide with
  // intersect with the plane. The center of the entire IR image,
  // draped in this way, is between the third and fourth section
  // vertically, and conincides with the line through the centers
  // of all sections horizontally. A point on a cube face located
  // on a corner of a 'core' has unit coordinate values (tan (pi/4))
  // the factors are multiplicative factors, and the pair given
  // is reciprocal, so, model_to_px == 1.0 / px_to_model, within
  // floating point precision. This is to facilitate conversions,
  // where we wish to avoid divisions which are harder to compute.

  double model_to_px ;
  double px_to_model ;

  // for conversion between model space units and texture units, we
  // need separate factors for x and y axis due to the 1:6 aspect ratio

  zimt::xel_t < double , 2 > model_to_tx ;
  zimt::xel_t < double , 2 > tx_to_model ;

  // section_size is the extent of a section in model space units,
  // and we have: section_size == section_width * px_to_model.
  // Note that the 'size' of the core in model space units is
  // precisely 2.0: it's 2 * tan ( pi/4 ) - so there is no variable
  // 'core_size' because it's implicit. We don't give a size
  // corresponding to the cube face images in the input.

  double section_size ;
  const double core_size = 2.0 ;

  // ref90 is the distance, in model space units, from the edge of
  // a section to the core, the 'cube face proper'.

  double ref90 ;

  // discrete90 is a flag which indicates whether the core has discrete
  // extent (in pixel units), so that ref90, converted to pixel units,
  // is a whole number - within floating point precision.

  bool discrete90 ;

  // the c'tor only needs one argument: the size of an individual
  // cube face image. This is the size covering the entire image,
  // whose field of view may be ninety degrees or more. The default
  // is ninety degrees, as we would expect with 'standard'
  // cubemaps, but we want to cater for cube face images which
  // already contain some support themselves. A minimal support of
  // four is ample for most direct interpolators, and the tile width
  // of 64 is a common choice, allowing for a good many levels of
  // mip-mapping with simple 4:1 pixel binning.

  metrics_t ( std::size_t _face_width ,
              double _face_fov = M_PI_2 ,
              std::size_t _support_min = 4UL ,
              std::size_t _tile_width = 64UL
            )
  : face_width ( _face_width ) ,
    face_fov ( _face_fov ) ,
    support_min ( _support_min ) ,
    tile_width ( _tile_width )
  {
    // first make sure that certain minimal requirements are met

    // we want even face size for now TODO can we use odd sizes?

    assert ( ( face_width & 1 ) == 0 ) ;

    // the cube face images must have at least 90 degrees fov

    assert ( face_fov >= M_PI_2 ) ;

    // the tile width must be at least one

    assert ( tile_width > 0 ) ;

    // the tile width must be a power of two

    assert ( ( tile_width & ( tile_width - 1 ) ) == 0 ) ;

    // given the face image's field of view, how much support does
    // the face image already contain? We start out by calculating
    // The cube face image's diameter (in model space units) and the
    // 'overscan' - by how much the diameter exceeds the 2.0 diameter
    // which occurs with a cube face image of precisely ninety degrees
    // field of view. If the cube face images cover precisely ninety
    // degrees, we leave the values as they are initialized here:

    double overscan = 0.0 ;
    double diameter = 2.0 ;
    inherent_support = 0 ;

    // discrete90 is true if the core is represented by a discrete
    // number of pixels

    discrete90 = true ;

    // calculate the diameter in model space units. The diameter
    // for a ninety degree face image would be precisely 2, and
    // if the partial image has larger field of view, it will be
    // larger.

    if ( face_fov > M_PI_2 )
    {
      diameter = 2.0 * tan ( face_fov / 2.0 ) ;
      discrete90 = false ;
      // this is very unlikely:
      if ( fabs ( diameter - std::nearbyint ( diameter ) ) < .0000001 )
        discrete90 = true ;
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

    if ( face_fov > M_PI_2 )
    {      
      overscan = tan ( face_fov / 2.0 ) - 1.0 ;

      // how wide is the overscan, expressed in pixel units?

      double px_overscan = model_to_px * overscan ;

      // truncate to integer to receive the inherent support in
      // pixel units. If this value is as large or larger than the
      // required support, we needn't add any additional space.

      inherent_support = int ( px_overscan ) ;
    }

    // if there is more inherent support than the minimal support
    // required, we don't need to provide additional support. If
    // the inherent support is too small, we do need additional
    // support.
  
    std::size_t additional_support ;

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

    section_width = n_tiles * tile_width ;
    frame_width = ( section_width - face_width ) / 2UL ;

    // paranoid

    assert ( ( 2UL * frame_width + face_width ) == section_width ) ;

    // the central part of each partial image, which covers
    // precisely ninety degrees - the cube face proper - is at
    // a specific distance from the edge of the total section.
    // we know that it's precisely 1.0 from the cube face image's
    // center in model space units. So first we convert the
    // 'outer width' to model space units

    section_size = px_to_model * section_width ;

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

    model_to_tx[0] = 1.0 / section_size ;
    model_to_tx[1] = 1.0 / ( 6.0 * section_size ) ;

    tx_to_model[0] = section_size ;
    tx_to_model[1] = 6.0 * section_size ;
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

    target[1] += f_v ( face_index * int(section_width) ) ;

    // Subtract 0.5 - we look at pixels as small squares with an
    // extent of 1 pixel unit, and an incoming coordinate which
    // is precisely on the margin of the cube face (a value of
    // +/- 1) has to be mapped to the outermost pixel's margin as
    // well. The output of this function produces coordinate
    // values which coincide with the discrete pixel indices,
    // So a value of (0,0) refers to the upper left pixel's
    // center, and direct interpolation would yield this pixel's
    // value precisely.
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

    target[1] += face_index * section_size ;

    // move from model space units to pixel units. This yields us
    // a coordinate in pixel units pertaining to the section.

    target *= model_to_tx ;
  }
} ;
