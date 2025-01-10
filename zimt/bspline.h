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

/*! \file bspline.h

    \brief defines class bspline

  This code is a port from my library vspline, using zimt data types
  instead of vspline and vigra types.

  class bspline is an object to contain a b-spline's coefficients and some
  metadata in one handy package. It also provides easy access to b-spline
  prefiltering. The idea is that user code establishes a bspline object
  representing the data at hand and then proceeds to create 'evaluators'
  to evaluate the spline. You may be reminded of SciPy's bisplrep object,
  and I admit that SciPy's bspline code has been one of my inspirations.

  It attempts to do 'the right thing' by automatically creating suitable helper
  objects and parametrization so that the spline does what it's supposed to do.
  Most users will not need anything else, and using class bspline is quite
  straightforward. It's quite possible to have a b-spline up and running with
  a few lines of code without even having to make choices concerning it's
  parametrization, since there are sensible defaults for everything. At the same
  time, pretty much everything *can* be parametrized.
  
  Note that class bspline does not provide evaluation of the spline. To evaluate,
  objects of class evaluator (see eval.h) are used, which construct from a bspline
  object and additional parameters, like, whether to calculate the spline's
  value or it's derivative(s) or whether to use optimizations for special cases.
  
  While using 'raw' coefficient arrays with an evaluation scheme which applies
  boundary conditions is feasible and most memory-efficient, it's not so well
  suited for very fast evaluation, since the boundary treatment needs conditionals,
  and 'breaks' uniform access, which is especially detrimental when using
  vectorization. So zimt uses coefficient arrays with a few extra coefficients
  'framing' or 'bracing' the 'core' coefficients. Since evaluation of the spline
  looks at a certain small section of coefficients (the evaluator's 'support'),
  the bracing is chosen so that this lookup will always succeed without having to
  consider boundary conditions: the brace is set up to make the boundary conditions
  explicit, and the evaluation can proceed blindly without bounds checking. With
  large coefficient arrays, the little extra space needed for the brace becomes
  negligible, but the code for evaluation becomes faster and simpler.
  In effect, 'bracing' is taken a little bit further than merely providing
  enough extra coefficients to cover the support: additional coefficients are
  produced to allow for the spline to be evaluated without bounds checking
  
  - at the lower and upper limit of the spline's defined range
  
  - and even slightly beyond those limits: a safeguard against quantization errors
  
  This makes the code more robust: being very strict about the ends of the
  defined range can easily result in quantization errors producing out-of-bounds
  access to the coefficient array, so the slightly wider brace acts as a safeguard.
  While the 'brace' has a specific size which depends on the parameters (see
  get_left_brace_size() and get_right_brace_size()) - there may be even more
  additional coefficients if this is needed (see parameter 'headroom'). All
  additional coefficients around the core form the spline's 'frame'. So the
  frame is always at least as large as the brace.
  Adding the frame around the coefficient array uses extra memory.
  For low spline dimensions, this does not 'cost much', but with rising
  dimension, the extra coefficients of the frame take up more and more
  space, and the method becomes impractical for splines with very many
  dimensions, which you should keep at the back of your mind if you intend
  to use zimt for higher-dimensional splines. This is due to the geometrical
  fact that with rising dimensionality the 'surface' of an object becomes
  proportionally larger, and the additional coefficients envelope the
  surface of the coefficient array.

  class bspline handles two views to the coefficients it operates on, these
  are realized as view_ts, and they share the same storage:

  - the 'container' is a view to all coefficients held by the spline, including
    the extra coefficients in it's 'frame'.

  - the 'core', is a view to a subarray of the container with precisely
    the same shape as the knot point data over which the spline is calculated.
    The coefficients in the core correspond 1:1 with the knot point data.

  Probably the most common scenario is that the source data for the spline are
  available from someplace like a file. Instead of reading the file's contents
  into memory first and passing the memory to class bspline, there is a more
  efficient way: a bspline object is set up first, with the specification of
  the size of the incoming data and the intended mode of operation. The bspline
  object allocates the memory it will need for the purpose, but doesn't do
  anything else. The 'empty' bspline object is then 'filled' by the user by
  putting data into it's 'core' area. Subsequently, prefilter() is called,
  which converts the data to b-spline coefficients. This way, only one block
  of memory is used throughout, the initial data are overwritten by the
  coefficients, operation is in-place and most efficient. This method works
  well with image import functions from vigraimpex: vigraimpex accepts
  view_ts as target for image loading, so the 'core' of a
  bspline object can be directly passed to vigraimpex to be filled with
  image data.

  If this pattern can't be followed, there are alternatives:

  - if data are passed to prefilter(), they will be taken as containing the knot
    point data, rather than expecting the knot point data to be in the bspline
    oject's memory already. This can also be used to reuse a bspline object with
    new data. The source data will not be modified.

  - if a view to an array at least the size of the container array is passed into
    bspline's constructor, this view is 'adopted' and all operations will use the
    data it refers to. The caller is responsible for keeping these data alive while
    the bspline object exists, and relinquishes control over the data, which may be
    changed by the bspline object.

*/

#include <limits>

#include "prefilter.h"
#include "brace.h"

#if defined(ZIMT_BSPLINE_H) == defined(HWY_TARGET_TOGGLE)
  #ifdef ZIMT_BSPLINE_H
    #undef ZIMT_BSPLINE_H
  #else
    #define ZIMT_BSPLINE_H
  #endif

HWY_BEFORE_NAMESPACE() ;
BEGIN_ZIMT_SIMD_NAMESPACE(zimt)

/// struct bspline is the object in zimt holding b-spline coefficients.
/// In a way, the b-spline 'is' it's coefficients, since it is totally
/// determined by them - while, of course, the 'actual' spline is an
/// n-dimensional curve. So, even if this is a bit sloppy, I often refer
/// to the coefficients as 'the spline', and have named struct bspline
/// so even if it just holds the coefficients.
///
/// The coefficients held in a bspline object are 'braced', providing a few
/// extra extrapolated coefficients around a 'core' area corresponding with
/// the knot point, or original data. This way, they can be used by zimt's
/// evaluation code which relies on such a brace being present.
///
/// struct bspline is a convenience class which bundles a coefficient array
/// (and it's creation) with a set of metadata describing the parameters used
/// to create the coefficients and the resulting data. I have chosen to implement
/// class bspline so that there is only a minimal set of template arguments,
/// namely the spline's data type (like pixels etc.) and it's dimension. All
/// other parameters relevant to the spline's creation are passed in at
/// construction time. This way, if explicit specialization becomes necessary
/// (like, to interface to code which can't use templates) the number of
/// specializations remains manageable. This design decision pertains specifically
/// to the spline's degree, which could also be implemented as a template argument,
/// allowing for some optimization by making some members static. Yet going down
/// this path requires explicit specialization for every spline degree used and
/// the performance gain I found doing so was hardly measurable, while automatic
/// testing became difficult and compilation times grew.
///
/// class bspline may or may not 'own' the coefficient data it refers to - this
/// depends on the specific initialization used, but is handled privately by
/// class b-spline, using a shared_ptr to the data if they are owned, which
/// makes bspline objects trivially copyable.
///
/// The 'metadata part' of the bspline object is coded as a base class, which
/// is convenient to derive other 'spline-like' classes with the same metadata
/// structure. class bspline itself adds the coeffcients of the spline and
/// the storage holding the coefficients.

template < std::size_t _dimension >
struct bspline_base
{
  /// 'suck in' the template argument

  static const std::size_t dimension = _dimension ;
  
  /// type of a multidimensional index type
  
  typedef xel_t < std::size_t , dimension > shape_type ;
  
  /// nD type for one boundary condition per axis
  
  typedef xel_t < bc_code , dimension > bcv_type ;

  /// nD type for one limit condition per axis
  
  typedef xel_t < xlf_type , dimension > limit_type ;

  // the metadata of the spline are fixed in the constructor and can't be
  // modified later on:

  const bcv_type bcv ;       // boundary conditions, see common.h
  const xlf_type tolerance ; // acceptable error
  
  const shape_type left_frame ;   // total width(s) of the left frame
  const shape_type core_shape ;   // shape of core array
  const shape_type right_frame ;  // total width(s) of the right frame
  const shape_type container_shape ; // shape of core array

  /// lower_limit returns the lower bound of the spline's defined range.
  /// This is usually 0.0, but with REFLECT boundary condition it's -0.5,
  /// the lower point of reflection. The lowest coordinate at which the
  /// spline can be accessed may be lower: even splines have wider support,
  /// and splines with extra headroom add even more room to manoevre.
  
  static xlf_type lower_limit ( const bc_code & bc )
  {
    xlf_type limit = 0.0L ;
    
    if ( bc == REFLECT || bc == PERIODIC )
      limit = -0.5L ;
    
    return limit ;
  }
  
  xlf_type lower_limit ( const int & axis ) const
  {
    return lower_limit ( bcv [ axis ] ) ;
  }
  
  limit_type lower_limit() const
  {
    limit_type limit ;
    for ( int d = 0 ; d < dimension ; d++ )
      limit[d] = lower_limit ( d ) ;
    return limit ;
  }
  
  /// upper_limit returns the upper bound of the spline's defined range.
  /// This is normally M - 1 if the shape for this axis is M. Splines with
  /// REFLECT boundary condition use M - 0.5, the upper point of reflection,
  /// and periodic splines use M. The highest coordinate at which the spline
  /// may be accessed safely may be higher.
  
  static xlf_type upper_limit ( const std::size_t & extent ,
                                const bc_code & bc )
  {
    xlf_type limit = extent - 1 ;
    
    if ( bc == REFLECT || bc == PERIODIC )
      limit += 0.5L ;
    // else if ( bc == PERIODIC )
    //   limit += 1.0L ;
    
    return limit ;
  }
  
  xlf_type upper_limit ( const int & axis ) const
  {
    xlf_type limit = core_shape [ axis ] - 1 ;
    
    if (    bcv [ axis ] == REFLECT
         || bcv [ axis ] == PERIODIC )
      limit += 0.5L ;
    // else if ( bcv [ axis ] == PERIODIC )
    //   limit += 1.0L ;
    
    return limit ;
  }
  
  limit_type upper_limit() const
  {
    limit_type limit ;
    for ( int d = 0 ; d < dimension ; d++ )
      limit[d] = upper_limit ( d ) ;
    return limit ;
  }
  
  /// get_left_brace_size and get_right_brace_size calculate the size of
  /// the brace zimt puts around the 'core' coefficients to allow evaluation
  /// inside the defined range (and even slightly beyond) without bounds
  /// checking. These routines are static to allow user code to establish
  /// zimt's bracing requirements without creating a bspline object.
  /// user code might use this information to generate coefficient arrays
  /// suitable for use with zimt evaluation code, sidestepping use of
  /// a bspline object.

  static shape_type get_left_brace_size ( int spline_degree , bcv_type bcv )
  {
    // we start out with left_brace as large as the support
    // of the reconstruction kernel

    int support = spline_degree / 2 ;
    shape_type left_brace ( support ) ;

    // for some situations, we extend the array further along a specific axis

    for ( int d = 0 ; d < dimension ; d++ )
    {
      // If the spline is done with REFLECT boundary conditions,
      // the lower and upper limits are between bounds.
      // the lower limit in this case is -0.5. When using
      // floor() or round() on this value, we receive -1,
      // so we need to extend the left brace.
      // if rounding could be done so that -0.5 is rounded
      // towards zero, this brace increase could be omitted
      // for even splines, but this would also bring operation
      // close to dangerous terrain: an infinitesimal undershoot
      // would already produce an out-of-bounds access.
      
      if ( bcv[d] == REFLECT )
      {
        left_brace[d] ++ ;
      }
      
      // for other boundary conditions, the lower limit is 0. for odd
      // splines, as long as evaluation is at positions >= 0 this is
      // okay, but as soon as evaluation is tried with a value
      // even just below 0, we'd have an out-of-bounds access,
      // with potential memory fault. Rather than requiring
      // evaluation to never undershoot, we stay on the side
      // of caution and extend the left brace, so that
      // quantization errors won't result in a crash.
      // This is debatable and could be omitted, if it can
      // be made certain that evaluation will never be tried
      // at values even infinitesimally below zero.
      // for even splines, this problem does not exist, since
      // coordinate splitting is done with std::round.
      
      else if ( spline_degree & 1 )
      {
        left_brace[d]++ ;
      }

      // for even periodic splines, we also use a wider brace.
      // without that, we'de get a technical lower limit of 0.5,
      // which is cutting it too fine if we're putting the lower
      // limit at 0.5 as well.

      if ( bcv[d] == PERIODIC && ( ! ( spline_degree & 1 ) ) )
      {
        left_brace[d]++ ;
      }
    }
    return left_brace ;
  }
  
  static shape_type get_right_brace_size ( int spline_degree , bcv_type bcv )
  {
    // we start out with right_brace as large as the support
    // of the reconstruction kernel

    int support = spline_degree / 2 ;
    shape_type right_brace ( support ) ;

    // for some situations, we extend the array further along a specific axis

    for ( int d = 0 ; d < dimension ; d++ )
    {
      // If the spline is done with REFLECT boundary conditions,
      // the lower and upper limits are between bounds.
      // So the upper limit is Z + 0.5 where Z is integer.
      // using floor on this value lands at Z, which is fine,
      // but using round() (as is done for even splines)
      // lands at Z+1, so for this case we need to extend
      // the right brace. If we could use a rounding mode
      // rounding towards zero, we could omit this extension,
      // but we'd also be cutting it very fine. Again we stay
      // on the side of caution.
      
      if ( bcv[d] == REFLECT )
      {
        if ( ! ( spline_degree & 1 ) )
          right_brace[d] ++ ;
      }
      
      // The upper limit is M-1 for most splines, and M-1+0.5 for
      // splines with REFLECT BCs. When accessing the spline at
      // this value, we'd be out of bounds.
      // For splines done with REFLECT BCs, we have to extend the
      // right brace to allow access to coordinates in [M-1,M-1+0.5],
      // there is no other option.
      // For other splines, We could require the evaluation code
      // to check and split incoming values of M-1 to M-2, 1.0, but this
      // would require additional inner-loop code. So we add another
      // coefficient on the upper side for these as well.
      // This is debatable, but with the current implementation of the
      // evaluation it's necessary.

      // So, staying on the side of caution, we add the extra coefficient
      // for all odd splines.

      if ( spline_degree & 1 )
      {
        right_brace[d]++ ;
      }
      
      // periodic splines need an extra coefficient on the upper
      // side, to allow evaluation in [M-1,M]. This interval is
      // implicit in the original data since the value at M is
      // equal to the value at 0, but if we want to process without
      // bounds checking and index manipulations, we must provide
      // an extra coefficient.
      
      if ( bcv[d] == PERIODIC )
      {
        right_brace[d]++ ;
      }
    }
    return right_brace ;
  }
  
  /// convenience method to caculate the shape of a container array needed to hold
  /// the coefficients of a spline with the given properties. The arguments are the
  /// same as those passed to the bspline object's constructor, but this method is
  /// static, so it can be called on the spline's type and does not need an object.
  /// I'm including this to make it easier for code which creates the container
  /// array externally before constructing the bspline object, rather than relying
  /// on class bspline to allocate it's own storage.
  
  static shape_type get_container_shape ( int spline_degree ,
                                          bcv_type bcv ,
                                          shape_type core_shape ,
                                          int headroom
                                        )
  {
    auto left_frame = get_left_brace_size ( spline_degree , bcv ) ;
    left_frame += headroom ;

    auto right_frame = get_right_brace_size ( spline_degree , bcv ) ;
    right_frame += headroom ;

    shape_type container_shape = core_shape + left_frame + right_frame ;
    
    return container_shape ;
  }

  /// variant for 1D splines. This variant accepts the shape as a plain long,
  /// rather than requiring a TinyVector of one long.

  template < typename = std::enable_if < _dimension == 1 > >
  static long get_container_shape ( int spline_degree ,
                                    bc_code bc ,
                                    long core_shape ,
                                    int headroom
                                  )
  {
    return get_container_shape ( spline_degree ,
                                 bcv_type ( bc ) ,
                                 shape_type ( core_shape ) ,
                                 headroom ) [ 0 ] ;
  }

  /// The constructor sets up all metrics of the spline from the basic
  /// properties given for the spline. Here, no defaults are offered,
  /// because this class is meant as a base class for splines only, and
  /// the expectation is that no objects of this class will be created.

  bspline_base ( shape_type _core_shape , // shape of knot point data
                 int _spline_degree ,     // spline degree
                 bcv_type _bcv ,          // boundary condition
                 xlf_type _tolerance ,    // acceptable error
                 int headroom
               )
  : core_shape ( _core_shape ) ,
    left_frame ( get_left_brace_size ( _spline_degree , _bcv )
                 + headroom ) ,
    right_frame ( get_right_brace_size ( _spline_degree , _bcv )
                  + headroom ) ,
    container_shape ( get_container_shape ( _spline_degree ,
                                            _bcv ,
                                            _core_shape ,
                                            headroom ) ) ,
    bcv ( _bcv ) ,
    tolerance ( _tolerance <= 0.0
                ? std::numeric_limits < xlf_type > :: epsilon()
                : _tolerance
              )
    { }

} ;

/// class bspline now builds on class bspline_base, adding coefficient storage,
/// while bspline_base provides metadata handling. This separation makes it
/// easier to generate classes which use the same metadata scheme. One concrete
/// example is a class to type-erase a spline (not yet part of the library) which
/// abstracts a spline, hiding the type of it's coefficients. Such a class can
/// inherit from bspline_base and initialize the base class in the c'tor from
/// a given bspline object, resulting in a uniform interface to the metadata.
/// class bspline takes an additional template argument: the value type. This
/// is the type of the coefficients stored in the spline, and it will typically
/// be a single fundamental type or a small aggregate - like a xel_t.
/// zimt uses vigra's ExpandElementResult mechanism to inquire for a value
/// type's elementary type and size, which makes it easy to adapt to new value
/// types because the mechanism is traits-based.

template < class _value_type , std::size_t _dimension >
struct bspline
: public bspline_base < _dimension >
{
  typedef _value_type value_type ;
  
  /// number of channels in value_type

  static const std::size_t channels = get_ele_t < value_type >::size ;

  /// elementary type of value_type, like float or double
  
  typedef typename get_ele_t < value_type >::type ele_type ;
  
  typedef bspline_base < _dimension > base_type ;

  /// inheritance between templates requires explicit coding of the use
  /// of base class facilities, hence we have a large-ish section of using
  /// declarations.

  using typename base_type::shape_type ;
  using typename base_type::bcv_type ;
  using typename base_type::limit_type ;

  using base_type::get_container_shape ;
  using base_type::get_left_brace_size ;
  using base_type::get_right_brace_size ;
  using base_type::lower_limit ;
  using base_type::upper_limit ;

  using base_type::dimension ;
  using base_type::core_shape ;
  using base_type::container_shape ;
  using base_type::left_frame ;
  using base_type::right_frame ;
  using base_type::tolerance ;
  using base_type::bcv ;

  /// if the coefficients are owned, an array of this type holds the data:
  
  typedef array_t < dimension , value_type > array_type ;
  
  /// data are read and written to vigra view_ts:
  
  typedef view_t < dimension , value_type > view_type ;
  
  /// the type of 'this' bspline and of a 'channel view'
  
  typedef bspline < value_type , dimension > spline_type ;
  
  typedef bspline < ele_type , dimension > channel_view_type ;
  
private:

  // _p_coeffs points to a array_t, which is either default-initialized
  // and contains no data, or holds data which are viewed by 'container'. Using a
  // std::shared_ptr here has the pleasant side-effect that class bspline objects 
  // can use the default copy and assignment operators without need to worry
  // about array life-time: The last bspline object going out of scope will
  // 'take the array with it'.
  // Keep in mind, though, that bspline objects based on externally allocated
  // storage (by passing _space in the c'tor) will merely copy the relevant
  // pointers to the coeffcients (as member variables of the views 'core' and
  // 'container') and rely on the storage being kept alive while the bspline
  // object persists.
  
  std::shared_ptr < array_type > _p_coeffs ;
  
public:

  view_type container ; // view to coefficient container array (incl. frame)
  view_type core ;      // view to the core part of the coefficient array
  
  /// boolean flag indicating whether the coefficients were prefiltered or not.
  /// This must be taken with a grain of salt - it's up to the user to decide
  /// whether this flag can be trusted. zimt code will set it whenever
  /// 'prefilter' is called, but since both this flag and the coefficients
  /// themselves are open to manipulation, external code performing such
  /// manipulations can decide to ignore the flag.
  /// Initially, the flag is set to 'false', and zimt does not perform
  /// prefiltering automatically, so it will remain false until prefilter
  /// is called or until the user 'manually' sets it to true.

  bool prefiltered ;

  // the degree of the spline is not const, it can be modified to
  // evaluate splines 'as if' the coefficients had been calculated
  // for the given degree even if they were calculated for a
  // different degree - in zimt this is called 'shifting'
  // the spline. Use the member function shift() rather than
  // modifying the value directly to avoid a result which won't
  // compute due to insufficient frame size.

  int spline_degree ;  // degree of the spline (3 == cubic spline)

  /// calculate the spline's "technical limits". The functions
  /// fllow the same pattern as lower_limit and upper_limit, but
  /// they calculate the smallest and largest actual coordinate
  /// value which is safe to use for the given spline. This takes
  /// into account the size of the 'frame' and the width of the
  /// basis function, and it will provide a wider range of values
  /// than the ones provided by lower_limit and upper_limit, unless
  /// the spline was 'messed with' and does not have the brace
  /// size which is normally assigned to it. If the spline has
  /// been created with extra headroom (to allow for 'shifting' it),
  /// this will be reflected in wider technical limits.
  /// The braces which zimt adds to the coefficient arrays
  /// are generous, and the technical limit will usually be at
  /// least just under 0.5 and often 1.0 beyond the 'straight'
  /// limit. The 'straight' limit is also used to give start and
  /// end point of a full 'period' - this makes sense throughout,
  /// because due to the periodization, all splines actually do
  /// have a period.

  xlf_type lower_technical_limit ( const int & axis ) const
  {
    int left_margin = left_frame [ axis ] ;
    if ( spline_degree == 0 )
    {
      return - left_margin - .5 + std::numeric_limits<double>::epsilon() ;
    }
    if ( spline_degree & 1 )
    {
      int kernel_half = ( spline_degree + 1 ) / 2 ;
      return kernel_half - left_margin - 1 ;
    }
    else
    {
      int kernel_half = spline_degree / 2 ;
      return kernel_half - left_margin - .5 ;
    }
  }
  
  limit_type lower_technical_limit() const
  {
    limit_type limit ;
    for ( int d = 0 ; d < dimension ; d++ )
      limit[d] = lower_technical_limit ( d ) ;
    return limit ;
  }
  
  xlf_type upper_technical_limit ( const int & axis ) const
  {
    int right_margin = right_frame [ axis ] ;
    if ( spline_degree == 0 )
    {
      return core_shape [ axis ] - 1 + right_margin + .5
               - std::numeric_limits<double>::epsilon() ;
    }
    if ( spline_degree & 1 )
    {
      int kernel_half = ( spline_degree + 1 ) / 2 ;
      return core_shape [ axis ] - 1 + right_margin - kernel_half + 1 ;
    }
    else
    {
      int kernel_half = spline_degree / 2 ;
      return core_shape [ axis ] - 1 + right_margin - kernel_half + .5 ;
    }
  }
  
  limit_type upper_technical_limit() const
  {
    limit_type limit ;
    for ( int d = 0 ; d < dimension ; d++ )
      limit[d] = upper_technical_limit ( d ) ;
    return limit ;
  }
  
  /// construct a bspline object with appropriate storage space to contain and process an array
  /// of knot point data with shape _core_shape. Depending on the the other
  /// parameters passed, more space than _core_shape may be allocated. Once the bspline object
  /// is ready, usually it is filled with the knot point data and then the prefiltering needs
  /// to be done. This sequence assures that the knot point data are present in memory only once,
  /// the prefiltering is done in-place. So the user can create the bspline, fill in data (like,
  /// from a file), prefilter, and then evaluate.
  ///
  /// alternatively, if the knot point data are already manifest elsewhere, they can be passed
  /// to prefilter(). With this mode of operation, they are 'pulled in' during prefiltering.
  ///
  /// It's possible to pass in a view to an array providing space for the coefficients,
  /// or even the coefficients themselves. This is done via the parameter _space. This has
  /// to be an array of the same or larger shape than the container array would end up having
  /// given all the other parameters. This view is then 'adopted' and subsequent processing
  /// will operate on it's data.
  ///
  /// The additional parameter 'headroom' is used to make the 'frame' even wider. This is
  /// needed if the spline is to be 'shifted' up (evaluated as if it had been prefiltered
  /// with a higher-degree prefilter) - see shift(). The headroom goes on top of the brace.
  ///
  /// While bspline objects allow very specific parametrization, most use cases won't use
  /// parameters beyond the first few. The only mandatory parameter is, obviously, the
  /// shape of the knot point data, the original data which the spline is built over.
  /// This shape 'returns' as the bspline object's 'core' shape. If this is the only
  /// parameter passed to the constructor, the resulting bspline object will be a
  /// cubic b-spline with mirror boundary conditions, allocating it's own storage for the
  /// coefficients.
  ///
  /// Note that passing tolerance = 0.0 may increase prefiltering time significantly,
  /// especially when prefiltering 1D splines, which can't use multithreaded and vectorized
  /// code in this case. Really, tolerance 0 doesn't produce significantly better results
  /// than the default, which is a very low value already. The tolerance 0 code is there
  /// more for completeness' sake, as it actually produces the result of the prefilter
  /// using the *formula* to calculate the initial causal and anticausal coefficient
  /// precisely, whereas the small tolerance used by default is so small that it
  /// roughly mirrors the arithmetic precision which can be achieved with the given
  /// type, which leads to nearly the same initial coefficients. Oftentimes even the default
  /// is too conservative - a 'reasonable' value is in the order of magnitude of the noise
  /// in the signal you're processing. But using the default won't slow things down a great
  /// deal, since it only results in the initial coefficients being calculated a bit less
  /// quickly. With nD data, tolerance 0 is less of a problem because the operation will
  /// still be multithreaded and vectorized.

  bspline ( shape_type _core_shape ,                // shape of knot point data
            int _spline_degree = 3 ,                // spline degree with reasonable default
            bcv_type _bcv = bcv_type ( MIRROR ) ,   // boundary conditions and common default
            xlf_type _tolerance = -1.0 ,            // acceptable error (-1: automatic)
            int headroom = 0 ,                      // additional headroom, for 'shifting'
            view_type _space = view_type()          // coefficient storage to 'adopt'
          )
  : base_type ( _core_shape ,
                _spline_degree ,
                _bcv ,
                ( _tolerance < 0.0
                  ? std::numeric_limits < ele_type > :: epsilon() 
                  : ( _tolerance == 0 
                      ? std::numeric_limits < xlf_type > :: epsilon()
                      : _tolerance
                    )
                ) ,
                headroom ) ,
    spline_degree ( _spline_degree ) ,
    prefiltered ( false )
  {
    // now either adopt external memory or allocate memory for the coefficients
    
    if ( _space.origin != nullptr )
    {
      // caller has provided space for the coefficient array. This space
      // has to be at least as large as the container_shape we have
      // determined to make sure it's compatible with the other parameters.
      // With the array having been provided by the caller, it's the caller's
      // responsibility to keep the data alive as long as the bspline object
      // is used to access them.

      if ( _space.shape != container_shape )
        throw shape_mismatch ( "the intended container shape does not match the shape of the storage space passed in" ) ;
      
      // if the shape matches, we adopt the data in _space.
      // We take a view to the container_shape-sized subarray only.
      
      container = _space ;
      
      // _p_coeffs is made to point to a default-constructed MultiArray,
      // which holds no data.
      
      _p_coeffs = std::make_shared < array_type >() ;
    }
    else
    {
      // _space was default-constructed and has no data.
      // in this case we allocate a container array and hold a shared_ptr
      // to it. so we can copy bspline objects without having to worry about
      // dangling pointers, or who destroys the array.
      
      _p_coeffs = std::make_shared < array_type > ( container_shape ) ;
      
      // 'container' is made to refer to a view to this array.
      
      container = *_p_coeffs ;
    }

    // finally we set the view to the core area
    
    core = container.subarray ( left_frame , left_frame + _core_shape ) ;
  } ;

  // set up a spline with two views, one for the 'core' and one for
  // the 'container. the caller is expected to set these two views
  // up correctly.

  bspline ( view_type _core ,
            view_type _space ,
            int _spline_degree = 3 ,                // spline degree with reasonable default
            bcv_type _bcv = bcv_type ( MIRROR ) ,   // boundary conditions and common default
            xlf_type _tolerance = -1.0 ,            // acceptable error (-1: automatic)
            int headroom = 0                        // additional headroom, for 'shifting'
          )
  : base_type ( _core.shape ,
                _spline_degree ,
                _bcv ,
                ( _tolerance < 0.0
                  ? std::numeric_limits < ele_type > :: epsilon() 
                  : ( _tolerance == 0 
                      ? std::numeric_limits < xlf_type > :: epsilon()
                      : _tolerance
                    )
                ) ,
                4 ) , // << TODO: think about this
    spline_degree ( _spline_degree ) ,
    prefiltered ( false )
  {
    container = _space ;
    core = _core ;
    
    // _p_coeffs is made to point to a default-constructed MultiArray,
    // which holds no data.
    
    _p_coeffs = std::make_shared < array_type >() ;
  }

  /// overloaded constructor for 1D splines. This is useful because if we don't
  /// provide it, the caller would have to pass TinyVector < T , 1 > instead of T
  /// for the shape and the boundary condition.
  
  // KFJ 2018-07-23 Now I'm using enable_if to provide the following
  // c'tor overload only if dimension == 1. This avoids an error when
  // declaring explicit specializations: for dimension != 1, the compiler
  // would try and create this c'tor overload, which mustn't happen.
  // with the enable_if, if dimension != 1, the code is not considered.

  template < typename = std::enable_if < _dimension == 1 > >
  bspline ( long _core_shape ,                      // shape of knot point data
            int _spline_degree = 3 ,                // spline degree with reasonable default
            bc_code _bc = MIRROR ,                  // boundary conditions and common default
            xlf_type _tolerance = -1.0 ,            // acceptable error (relative to unit pulse)
            int headroom = 0 ,                      // additional headroom, for 'shifting'
            view_type _space = view_type()          // coefficient storage to 'adopt'
          )
  : bspline ( xel_t < long , 1 > ( _core_shape ) ,
              _spline_degree ,
              bcv_type ( _bc ) ,
              _tolerance ,
              headroom ,
              _space
            )
  { } ;

  /// get a bspline object for a single channel of the data. This is lightweight
  /// and requires the viewed data to remain present as long as the channel view is used.
  /// the channel view inherits all metrics from it's parent, only the view_ts
  /// to the data are different.
  
  channel_view_type get_channel_view ( const int & channel )
  {
    assert ( channel < channels ) ;
    
    ele_type * base = (ele_type*) ( container.data() ) ;
    base += channel ;
    auto stride = container.strides ;
    stride *= channels ;
    
    view_t < dimension , ele_type >
      channel_container ( container.shape() , stride , base ) ;

    // KFJ 2022-01-14 the channel view was created with headroom = 0, which
    // is only correct if the 'mother' spline has headroom == 0. Now the
    // 'mother' spline's headroom is calculated (because it's not stored as
    // a member variable) as the difference between the actual left frame's
    // size and the size as it would have been without additional headroom:

    auto std_left_frame_size = get_left_brace_size ( spline_degree , bcv ) ;
    auto headroom = left_frame[0] - std_left_frame_size[0] ;

    return channel_view_type ( core.shape() , 
                               spline_degree ,
                               bcv ,
                               tolerance ,
                               headroom ,
                               channel_container // coefficient storage to 'adopt'
                             ) ;
  } ;

  /// if the spline coefficients are already known, they obviously don't need to be
  /// prefiltered. But in order to be used by zimt's evaluation code, they need to
  /// be 'braced' - the 'core' coefficients have to be surrounded by more coeffcients
  /// covering the support the evaluator needs to operate without bounds checking
  /// inside the spline's defined range. brace() performs this operation. brace()
  /// assumes the bspline object has been set up with the desired initial parameters,
  /// so that the boundary conditions and metrics are already known and storage is
  /// available. brace() can be called for a specific axis, or for the whole container
  /// by passing -1.
  /// User code will only need to call 'brace' explicitly if the coefficients were
  /// modified 'from the outside' - when 'prefilter' is called, 'brace' is called
  /// automatically.

  void brace ( int axis = -1 ) ///< specific axis, -1: all
  {
    if ( axis == -1 )
    {
      bracer < dimension , value_type > :: apply
        ( container , bcv , left_frame , right_frame ) ;
    }
    else
    {
      bracer < dimension , value_type > :: apply
        ( container , bcv[axis] ,
          left_frame[axis] , right_frame[axis] , axis ) ;
    }
  }

  /// prefilter converts the knot point data in the 'core' area into b-spline
  /// coefficients. Bracing/framing will be applied. Even if the degree of the
  /// spline is zero or one, prefilter() can be called because it also
  /// performs the bracing.
  /// the arithmetic of the prefilter is performed in 'math_ele_type', which
  /// defaults to the vigra RealPromoted elementary type of the spline's
  /// value_type. This default ensures that integral knot point data are
  /// handled appropriately and prefiltering them will only suffer from
  /// quantization errors, which may be acceptable if the dynamic range
  /// is sufficiently large.
  /// 'boost' is an additional factor which will be used to amplify the
  /// incoming signal. This is intended for cases where the range of the
  /// signal has to be widened to fill the dynamic range of the signal's
  /// data type (specifically if it is an integral type). User code has
  /// to deal with the effects of boosting the signal, the bspline object
  /// holds no record of the 'boost' being applied. When evaluating a
  /// spline with boosted coefficients, user code will have to provide
  /// code to attenuate the resulting signal back into the original
  /// range; for an easy way of doing so, see amplify_type,
  /// which is a type derived from unary_functor providing
  /// multiplication with a factor. There is an example of it's application
  /// in the context of an integer-valued spline in int_spline.cc.
  
  template < class math_ele_type
               = REAL_EQUIV ( ele_type ) ,
             size_t vsize
               = vector_traits<math_ele_type>::size >
  void prefilter ( xlf_type boost = xlf_type ( 1 ) ,
                   int njobs = default_njobs )
  {
    // we assume data are already in 'core' and we operate in-place
    // prefilter first, passing in BC codes to pick out the appropriate functions to
    // calculate initial causal and anticausal coefficient, then 'brace' result.
    // note how, just as in brace(), the whole frame is filled, which may be more
    // than is strictly needed by the evaluator.
    
    zimt::prefilter < dimension ,
                      value_type ,
                      value_type ,
                      math_ele_type ,
                      vsize >
                    ( core ,
                      core ,
                      bcv ,
                      spline_degree ,
                      tolerance ,
                      boost ,
                      njobs
                    ) ;

    brace() ;
    prefiltered = true ;
  }

  /// If data are passed in, they have to have precisely the shape
  /// we have set up in core (_core_shape passed into the constructor).
  /// These data will then be used in place of any data present in the
  /// bspline object to calculate the coefficients. They won't be looked at
  /// after prefilter() terminates, so it's safe to pass in a view_t
  /// which is destroyed after the call to prefilter() returns. Any data
  /// residing in the bspline object's memory will be overwritten.
  /// here, the default math_ele_type ensures that math_ele_type is
  /// appropriate for both T and ele_type.

  template < typename T ,
             typename math_ele_type
               = REAL_EQUIV ( ele_type ) ,
             size_t vsize = vector_traits<math_ele_type>::size >
  void prefilter ( const view_t < dimension , T > & data ,
                   xlf_type boost = xlf_type ( 1 ) ,
                   int njobs = default_njobs
                 )
  {
    // if the user has passed in data, they have to have precisely the shape
    // we have set up in core (_core_shape passed into the constructor).
    // This can have surprising effects if the container array isn't owned by the
    // spline but constitutes a view to data kept elsewhere (by passing _space the
    // to constructor): the data held by whatever constructed the bspline object
    // will be overwritten with the (prefiltered) data passed in via 'data'.
    // Whatever data have been in the core will be overwritten.
    
    if ( data.shape != core.shape )
      throw shape_mismatch
        ( "when passing data to prefilter, they have to have precisely the core's shape" ) ;

    // prefilter first, passing in BC codes to pick out the appropriate functions to
    // calculate initial causal and anticausal coefficient, then 'brace' result.
    // note how, just as in brace(), the whole frame is filled, which may be more
    // than is strictly needed by the evaluator.
    
    zimt::prefilter < dimension ,
                      T ,
                      value_type ,
                      math_ele_type ,
                      vsize >
                    ( data ,
                      core ,
                      bcv ,
                      spline_degree ,
                      tolerance ,
                      boost ,
                      njobs
                    ) ;

    brace() ;
    prefiltered = true ;
  }

  /// 'shift' will change the interpretation of the data in a bspline object.
  /// d is taken as a difference to add to the current spline degree. Coefficients
  /// remain the same, but creating an evaluator from the shifted spline will make
  /// the evaluator produce data *as if* the coefficients were those of a spline
  /// of the changed order. Shifting with positive d will efectively blur the
  /// interpolated signal, shifting with negative d will sharpen it.
  /// For shifting to work, the spline has to have enough 'headroom', meaning that
  /// spline_degree + d, the new spline degree, has to be greater or equal to 0
  /// and smaller than the largest supported spline degree (lower fourties) -
  /// and, additionally, there has to be a wide-enough brace to allow evaluation
  /// with the wider kernel of the higher-degree spline's reconstruction filter.
  /// So if a spline is set up with degree 0 and shifted to degree 5, it has to be
  /// constructed with an additional headroom of 3 (see the constructor).
  ///
  /// shiftable() is called with a desired change of spline_degree. If it
  /// returns true, interpreting the data in the container array as coefficients
  /// of a spline with the changed degree is safe. If not, the frame size is
  /// not sufficient or the resulting degree is invalid and shiftable()
  /// returns false. Note how the decision is merely technical: if the new
  /// degree is okay and the *frame* is large enough, the shift will be
  /// considered permissible.

  // KFJ 2022-07-07 removed test for excession of max_degree: with alternative
  // basis functions, the maximal spline degree is not necessarily an issue,
  // so the 'strictly technical' criterion of sufficient frame size is now
  // used exclusively.
  
  bool shiftable ( int d ) const
  {
    int new_degree = spline_degree + d ;
    if ( new_degree < 0 ) // || new_degree > zimt_constants::max_degree )
      return false ;

    shape_type new_left_brace = get_left_brace_size ( new_degree , bcv ) ;
    shape_type new_right_brace = get_right_brace_size ( new_degree , bcv ) ;
    for ( std::size_t i = 0 ; i < dimension ; i++ )
    {
      if (    new_left_brace[i] > left_frame[i]
           || new_right_brace[i] > right_frame[i] )
        return false ;
    }
    // if (    allLessEqual ( new_left_brace , left_frame )
    //      && allLessEqual ( new_right_brace , right_frame ) )
    // {
    //   return true ;
    // }

    return true ;
  }
  
  /// shift() actually changes the interpretation of the data. The data
  /// will be taken to be coefficients of a spline with degree
  /// spline_degree + d, and the original degree is lost. This operation
  /// is only performed if it is technically safe (see shiftable()).
  /// If the shift was performed successfully, this function returns true,
  /// false otherwise.
  /// Note that, rather than 'shifting' the b-spline object, it's also
  /// possible to use a 'shifted' evaluator to produce the same result.
  /// See class evaluator's constructor.

  bool shift ( int d )
  {
    if ( shiftable ( d ) )
    {
      spline_degree += d ;
      return true ;
    }
    return false ;
  }

  /// helper function to pretty-print a bspline object to an ostream

  friend std::ostream & operator<< ( std::ostream & osr , const bspline & bsp )
  {
    osr << "dimension:................... " << bsp.dimension << std::endl ;
    osr << "degree:...................... " << bsp.spline_degree << std::endl ;
    osr << "boundary conditions:......... " ;
    for ( std::size_t i = 0 ; i < bsp.dimension ; i++ )
      osr << " " << bc_name [ bsp.bcv[i] ] ;
    osr << std::endl ;
    osr << "shape of container array:.... " << bsp.container.shape << std::endl ;
    osr << "shape of core:............... " << bsp.core.shape << std::endl ;
    osr << "left frame:.................. " << bsp.left_frame << std::endl ;
    osr << "right frame:................. " << bsp.right_frame << std::endl ;
    osr << "ownership of data:........... "  ;
    osr << ( bsp._p_coeffs->origin
             ? "bspline object owns data"
             : "data are owned externally" ) << std::endl ;
    osr << "container base adress:....... " << bsp.container.data() << std::endl ;
    osr << "core base adress:............ " << bsp.core.data() << std::endl ;
    osr << "prefiltered:................. "
        << ( bsp.prefiltered ? "yes" : "no" ) << std::endl ;
    return osr ;
  }

} ;

/// using declaration for a coordinate suitable for bspline, given
/// elementary type rc_type. This produces the elementary type itself
/// if the spline is 1D, a TinyVector of rc_type otherwise.

template < class spline_type , typename rc_type >
using bspl_coordinate_type
= canonical_type < rc_type , spline_type::dimension > ;
      
/// using declaration for a bspline's value type

template < class spline_type >
using bspl_value_type
= typename spline_type::value_type ;

END_ZIMT_SIMD_NAMESPACE
HWY_AFTER_NAMESPACE() ;

#endif // sentinel
