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

/*! \file map.h

    \brief code to handle out-of-bounds coordinates.

    This is a port from the vspline library

    Incoming coordinates may not be inside the range which can be evaluated
    by a functor. There is no one correct way of dealing with out-of-bounds
    coordinates, so I provide a few common ways of doing it.
    
    If the 'standard' gate types don't suffice, the classes provided here
    can serve as templates.
    
    The basic type handling the operation is a 'gate type', which 'treats'
    a single value or single simdized value. For nD coordinates, we use a
    set of these gate_type objects, one for each component; each one may be
    of a distinct type specific to the axis the component belongs to.
    
    Application of the gates is via a 'mapper' object, which contains
    the gate_type objects and applies them to the components in turn.
    
    The mapper object is a functor which converts an arbitrary incoming
    coordinate into a 'treated' coordinate (or, with REJECT mode, may
    throw an out_of_bounds exception).
    
    mapper objects are derived from unary_functor, so they fit in
    well with other code in zimt and can easily be combined with other
    unary_functor objects, or used stand-alone. They are used inside zimt
    to implement the factory function make_safe_evaluator, which
    chains a suitable mapper and an evaluator to create an object allowing
    safe evaluation of a b-spline with arbitrary coordinates where out-of-range
    coordinates are mapped to the defined range in a way fitting the b-spline's
    boundary conditions.
*/

#if defined(ZIMT_MAP_H) == defined(HWY_TARGET_TOGGLE)
  #ifdef ZIMT_MAP_H
    #undef ZIMT_MAP_H
  #else
    #define ZIMT_MAP_H
  #endif

#include <assert.h>

#include "unary_functor.h"

HWY_BEFORE_NAMESPACE() ;
BEGIN_ZIMT_SIMD_NAMESPACE(zimt)

/// class pass_gate passes it's input to it's output unmodified.

template < typename rc_type ,
           size_t _vsize = vector_traits < rc_type > :: size
         >
struct pass_gate
: public unary_functor < rc_type , rc_type , _vsize >
{
  template < class T >
  void eval ( const T & c ,
                    T & result ) const
  {
    result = c ;
  }
} ;

/// factory function to create a pass_gate type functor

template < typename rc_type ,
           size_t _vsize = vector_traits < rc_type > :: size
         >
pass_gate < rc_type , _vsize >
pass()
{
  return pass_gate < rc_type , _vsize >() ;
}

/// reject_gate throws out_of_bounds for invalid coordinates

template < typename rc_type ,
           size_t _vsize = vector_traits < rc_type > :: size
         >
struct reject_gate
: public unary_functor < rc_type , rc_type , _vsize >
{
  const rc_type lower ;
  const rc_type upper ;
  
  reject_gate ( rc_type _lower ,
                rc_type _upper )
  : lower ( _lower ) ,
    upper ( _upper )
  { } ;

  void eval ( const rc_type & c ,
                    rc_type & result ) const
  {
    if ( c < lower || c > upper )
      throw out_of_bounds() ;
    result = c ;
  }
  
  /// vectorized evaluation function. This is enabled only if vsize > 1
  /// to guard against cases where vectorization is used but vsize is 1.
  /// Without the enable_if, we'd end up with two overloads with the
  /// same signature, since in_v and out_v collapse to in_type and out_type
  /// with vsize 1.

  typedef unary_functor < rc_type , rc_type , _vsize > base_type ;
  
  using typename base_type::in_v ;
  using typename base_type::out_v ;
  
  template < typename = std::enable_if < ( _vsize > 1 ) > >
  void eval ( const in_v & c ,
                    out_v & result ) const
  {
    if ( any_of (   ( c < lower )
                           | ( c > upper ) ) )
      throw out_of_bounds() ;
    result = c ;
  }

} ;

/// factory function to create a reject_gate type functor given
/// a lower and upper limit for the allowed range.

template < typename rc_type ,
           size_t _vsize = vector_traits < rc_type > :: size
         >
reject_gate < rc_type , _vsize >
reject ( rc_type lower , rc_type upper )
{
  return reject_gate < rc_type , _vsize > ( lower , upper ) ;
}

/// clamp gate clamps out-of-bounds values. clamp_gate takes
/// four arguments: the lower and upper limit of the gate, and
/// the values which are returned if the input is outside the
/// range: 'lfix' if it is below 'lower' and 'ufix' if it is
/// above 'upper'

template < typename rc_type ,
           size_t _vsize = vector_traits < rc_type > :: size
         >
struct clamp_gate
: public unary_functor < rc_type , rc_type , _vsize >
{
  
  const rc_type lower ;
  const rc_type upper ;
  const rc_type lfix ;
  const rc_type ufix ;
  
  clamp_gate ( rc_type _lower ,
               rc_type _upper ,
               rc_type _lfix ,
               rc_type _ufix )
  : lower ( _lower <= _upper ? _lower : _upper ) ,
    upper ( _upper >= _lower ? _upper : _lower ) ,
    lfix ( _lower <= _upper ? _lfix : _ufix ) ,
    ufix ( _upper >= _lower ? _ufix : _lfix )
  { 
    assert ( lower <= upper ) ;
  } ;

  /// simplified constructor, gate clamps to _lower and _upper
  
  clamp_gate ( rc_type _lower ,
               rc_type _upper )
  : clamp_gate ( _lower , _upper , _lower , _upper )
  { } ;

  void eval ( const rc_type & c ,
                    rc_type & result ) const
  {
    if ( c < lower )
      result = lfix ;
    else if ( c > upper )
      result = ufix ;
    else
      result = c ;
  }

  typedef unary_functor < rc_type , rc_type , _vsize > base_type ;
  
  using typename base_type::in_v ;
  using typename base_type::out_v ;
  
  template < typename = std::enable_if < ( _vsize > 1 ) > >
  void eval ( const in_v & c ,
                    out_v & result ) const
  {
    result = c ;
    result ( c < lower ) = lfix ;
    result ( c > upper ) = ufix ;
  }

} ;

/// factory function to create a clamp_gate type functor given
/// a lower and upper limit for the allowed range, and, optionally,
/// the values to use if incoming coordinates are out-of-range

template < typename rc_type ,
           size_t _vsize = vector_traits < rc_type > :: size
         >
clamp_gate < rc_type , _vsize >
clamp ( rc_type lower , rc_type upper ,
        rc_type lfix , rc_type rfix )
{
  return clamp_gate < rc_type , _vsize >
    ( lower , upper , lfix , rfix ) ;
}

template < typename rc_type ,
           size_t _vsize = vector_traits < rc_type > :: size
         >
clamp_gate < rc_type , _vsize >
clamp ( rc_type lower , rc_type upper )
{
  return clamp_gate < rc_type , _vsize >
    ( lower , upper ) ;
}

/// vectorized fmod function using std::trunc, which is fast, but
/// checking the result to make sure it's always <= rhs.

template <typename rc_v>
rc_v v_fmod ( rc_v lhs ,
              const typename rc_v::value_type & rhs )
{
  rc_v help ( lhs ) ;
  help /= rhs ;
  help = trunc ( help ) ;
  help *= rhs ;
  lhs -= help ;
  // due to arithmetic imprecision, result may come out >= rhs
  // so we doublecheck and set result to 0 when this occurs
  lhs ( abs(lhs) >= abs(rhs) ) = 0 ;
  return lhs ;
}

/// mirror gate 'folds' coordinates into the range. From the infinite
/// number of mirror images resulting from mirroring the input on the
/// bounds, the only one inside the range is picked as the result.
/// When using this gate type with splines with MIRROR boundary conditions,
/// if the shape of the core for the axis in question is M, _lower would be
/// passed 0 and _upper M-1.
/// For splines with REFLECT boundary conditions, we'd pass -0.5 to
/// _lower and M-0.5 to upper, since here we mirror 'between bounds'
/// and the defined range is wider.
///
/// Note how this mode of 'mirroring' allows use of arbitrary coordinates,
/// rather than limiting the range of acceptable input to the first reflection,
/// as some implementations do.

template < typename rc_type ,
           size_t _vsize = vector_traits < rc_type > :: size
         >
struct mirror_gate
: public unary_functor < rc_type , rc_type , _vsize >
{
  const rc_type lower ;
  const rc_type upper ;
  
  mirror_gate ( rc_type _lower ,
                rc_type _upper )
  : lower ( _lower <= _upper ? _lower : _upper ) ,
    upper ( _upper >= _lower ? _upper : _lower )
  { 
    assert ( lower < upper ) ;
  } ;

  void eval ( const rc_type & c ,
                    rc_type & result ) const
  {
    rc_type cc ( c - lower ) ;
    auto w = upper - lower ;

    cc = std::abs ( cc ) ;        // left mirror, v is now >= 0

    if ( cc >= w )
    {
      cc = fmod ( cc , 2 * w ) ;  // map to one full period
      cc -= w ;                   // center
      cc = std::abs ( cc ) ;      // map to half period
      cc = w - cc ;               // flip
    }
    
    result = cc + lower ;
  }

  typedef unary_functor < rc_type , rc_type , _vsize > base_type ;
  
  using typename base_type::in_v ;
  using typename base_type::out_v ;
  
  template < typename = std::enable_if < ( _vsize > 1 ) > >
  void eval ( const in_v & c ,
                    out_v & result ) const
  {
    in_v cc ( c - lower ) ;
    auto w = upper - lower ;

    cc = abs ( cc ) ;               // left mirror, v is now >= 0

    auto mask = ( cc >= w ) ;
    if ( any_of ( mask ) )
    {
      auto cm = v_fmod ( cc , 2 * w ) ;  // map to one full period
      cm -= w ;                          // center
      cm = abs ( cm ) ;                  // map to half period
      cm = in_v(w) - cm ;                // flip
      cc ( mask ) = cm ;
    }
    
    result = cc + lower ;
  }
  
} ;

/// factory function to create a mirror_gate type functor given
/// a lower and upper limit for the allowed range.

template < typename rc_type ,
           size_t _vsize = vector_traits < rc_type > :: size
         >
mirror_gate < rc_type , _vsize >
mirror ( rc_type lower , rc_type upper )
{
  return mirror_gate < rc_type , _vsize > ( lower , upper ) ;
}

/// the periodic mapping also folds the incoming value into the allowed range.
/// The resulting value will be ( N * period ) from the input value and inside
/// the range, period being upper - lower.
/// For splines done with PERIODIC boundary conditions, if the shape of
/// the core for this axis is M, we'd pass 0 to _lower and M to _upper.

template < typename rc_type ,
           size_t _vsize = vector_traits < rc_type > :: size
         >
struct periodic_gate
: public unary_functor < rc_type , rc_type , _vsize >
{
  const rc_type lower ;
  const rc_type upper ;
  
  periodic_gate ( rc_type _lower ,
               rc_type _upper )
  : lower ( _lower <= _upper ? _lower : _upper ) ,
    upper ( _upper >= _lower ? _upper : _lower )
  { 
    assert ( lower < upper ) ;
  } ;

  void eval ( const rc_type & c ,
                    rc_type & result ) const
  {
    rc_type cc = c - lower ;
    auto w = upper - lower ;
    
    if ( ( cc < 0 ) || ( cc >= w ) )
    {
      cc = fmod ( cc , w ) ;
      if ( cc < 0 )
        cc += w ;
      // due to arithmetic imprecision, even though cc < 0
      // cc+w may come out == w, so we need to test again:
      if ( cc >= w )
        cc = 0 ;
    }
    
    result = cc + lower ;
  }

  typedef unary_functor < rc_type , rc_type , _vsize > base_type ;
  
  using typename base_type::in_v ;
  using typename base_type::out_v ;
  
  template < typename = std::enable_if < ( _vsize > 1 ) > >
  void eval ( const in_v & c ,
                    out_v & result ) const
  {
    in_v cc ;
    
    cc = c - lower ;
    auto w = upper - lower ;

    auto mask_below = ( cc < 0 ) ;
    auto mask_above = ( cc >= w ) ;
    auto mask_any = mask_above | mask_below ;

    if ( any_of ( mask_any ) )
    {
      auto cm = v_fmod ( cc , w ) ;
      cm ( mask_below ) = ( cm + w ) ;
      // due to arithmetic imprecision, even though cc < 0
      // cc+w may come out == w, so we need to test again:
      cm ( cm >= w ) = 0 ;
      cc ( mask_any ) = cm ;
    }
    
    result = cc + lower ;
  }
} ;

/// factory function to create a periodic_gate type functor given
/// a lower and upper limit for the allowed range.

template < typename rc_type ,
           size_t _vsize = vector_traits < rc_type > :: size
         >
periodic_gate < rc_type , _vsize >
periodic ( rc_type lower , rc_type upper )
{
  return periodic_gate < rc_type , _vsize > ( lower , upper ) ;
}

/// finally we define class mapper which is initialized with a set of
/// gate objects (of arbitrary type) which are applied to each component
/// of an incoming nD coordinate in turn.
/// The trickery with the variadic template argument list is necessary,
/// because we want to be able to combine arbitrary gate types (which
/// have distinct types) to make the mapper as efficient as possible.
/// the only requirement for a gate type is that it has to provide the
/// necessary eval() functions.

template < typename nd_rc_type ,
           size_t _vsize ,
           class ... gate_types >
struct map_functor
: public unary_functor < nd_rc_type , nd_rc_type , _vsize >
{
  typedef unary_functor
            < nd_rc_type , nd_rc_type , _vsize > base_type ;
  
  typedef typename base_type::in_type in_type ;
  typedef typename base_type::out_type out_type ;
  
  static const std::size_t vsize = _vsize ;

  static const std::size_t dimension = get_ele_t<nd_rc_type>::size ;
  
  // we hold the 1D mappers in a tuple
  
  typedef std::tuple < gate_types... > mvec_type ;
  
  // mvec holds the 1D gate objects passed to the constructor
  
  const mvec_type mvec ;
  
  // the constructor receives gate objects

  map_functor ( gate_types ... args )
  : mvec ( args... )
  { } ;
  
  // constructor variant taking a tuple of gates
  
  map_functor ( const mvec_type & _mvec )
  : mvec ( _mvec )
  { } ;
  
  // to handle the application of the 1D gates, we use a recursive
  // helper type which applies the 1D gate for a specific axis and
  // then recurses to the next axis until axis 0 is reached.
  // We also pass 'dimension' as template argument, so we can specialize
  // for 1D operation (see below)

  template < int level , int dimension , typename nd_coordinate_type >
  struct _map
  { 
    void operator() ( const mvec_type & mvec ,
                      const nd_coordinate_type & in ,
                      nd_coordinate_type & out ) const
    {
      std::get<level>(mvec).eval ( in[level] , out[level] ) ;
      _map < level - 1 , dimension , nd_coordinate_type >() ( mvec , in , out ) ;
    }
  } ;
  
  // at level 0 the recursion ends
  
  template < int dimension , typename nd_coordinate_type >
  struct _map < 0 , dimension , nd_coordinate_type >
  { 
    void operator() ( const mvec_type & mvec ,
                      const nd_coordinate_type & in ,
                      nd_coordinate_type & out ) const
    {
      std::get<0>(mvec).eval ( in[0] , out[0] ) ;
    }
  } ;
  
  // here's the specialization for 1D operation

  template < typename coordinate_type >
  struct _map < 0 , 1 , coordinate_type >
  { 
    void operator() ( const mvec_type & mvec ,
                      const coordinate_type & in ,
                      coordinate_type & out ) const
    {
      std::get<0>(mvec).eval ( in , out ) ;
    }
  } ;

  // now we define eval for unvectorized and vectorized operation
  // by simply delegating to struct _map at the top level.

  template < class in_type , class out_type >
  void eval ( const in_type & in ,
                    out_type & out ) const
  {
    _map < dimension - 1 , dimension , in_type >() ( mvec , in , out ) ;
  }

} ;

/// factory function to create a mapper type functor given
/// a set of gate_type objects. Please see make_safe_evaluator
/// for code to automatically create a mapper object suitable for a
/// specific bspline.

template < typename nd_rc_type ,
           size_t _vsize = vector_traits < nd_rc_type > :: size ,
           class ... gate_types >
map_functor < nd_rc_type , _vsize , gate_types... >
mapper ( gate_types ... args )
{
  return map_functor < nd_rc_type , _vsize , gate_types... >
    ( args... ) ;
}

END_ZIMT_SIMD_NAMESPACE
HWY_AFTER_NAMESPACE() ;

#endif // sentinel
