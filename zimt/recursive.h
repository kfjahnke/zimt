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

/*! \file recursive.h

    \brief Code to create the coefficient array for a b-spline.

    This is a port from vspline/prefilter.h, I've factored out
    the code which is not specific for b-splines, to avoid pulling
    in 'poles.h', which is only needed for b-spline processing.

    The code in this file implements specific recursive filters:
    n-pole forward-backward recursive filters. This is the specific
    type of filter wjich is used for b-spline prefiltering, but in
    a more general form where the filter poles can be chosen
    arbitrarily, whereas a b-spline prefilter picks precomputed poles
    for a given spline degree.

    The code is quite old; it's been in use in vspline for a long
    time and I see little need to renovate it - it does the trick.
    At the end of the file is a function template meant to be called
    by user code - for b-spline prefiltering, use the more specific
    function template 'prefilter' in prefilter.h

*/

#include <limits>
#include "common.h"
#include "filter.h"

#if defined(ZIMT_CONVOLVE_H) == defined(HWY_TARGET_TOGGLE)
  #ifdef ZIMT_CONVOLVE_H
    #undef ZIMT_CONVOLVE_H
  #else
    #define ZIMT_CONVOLVE_H
  #endif

HWY_BEFORE_NAMESPACE() ;
BEGIN_ZIMT_SIMD_NAMESPACE(zimt)

using namespace std ;

/// overall_gain is a helper routine:
/// Simply executing the filtering code by itself will attenuate the signal. Here
/// we calculate the gain which, pre-applied to the signal, will cancel this effect.
/// While this code was initially part of the filter's constructor, I took it out
/// to gain some flexibility by passing in the gain as a parameter.
///
/// Note that higher-degree splines need filtering with some poles which are *very*
/// small numerically. This is a problem: The data get 'squashed', since there are
/// mathematical operations between attenuated and unattenuated values. So for high
/// spline degrees, float data aren't suitable, and even doubles and long doubles
/// suffer from squashing and lose precision.
///
/// Note also how we perform the arithmetics in this routine in the highest precision
/// available. Calling code will cast the product down to the type it uses for maths.

static xlf_type overall_gain ( const int & nbpoles ,
                               const xlf_type * const pole )
{
  xlf_type lambda = 1 ;

  for ( int k = 0 ; k < nbpoles ; k++ )

    lambda *= ( 1 - pole[k] ) * ( 1 - 1 / pole[k] ) ;
  
  return lambda ;
}

/// structure to hold specifications for an iir_filter object.
/// This set of parameters has to be passed through from
/// the calling code through the multithreading code to the worker threads
/// where the filter objects are finally constructed. Rather than passing
/// the parameters via some variadic mechanism, it's more concise and
/// expressive to contain them in a structure and pass that around.
/// The filter itself inherits its specification type, and if the code
/// knows the handler's type, it can derive the spec type. This way the
/// argument passing can be formalized, allowing for uniform handling of
/// several different filter types with the same code. Here we have the
/// concrete parameter set needed for b-spline prefiltering. We'll pass
/// one set of 'specs' per axis; it contains:
/// - the boundary condition for this axis
/// - the number of filter poles (see poles.h)
/// - a pointer to npoles poles
/// - the acceptable tolerance

// TODO: KFJ 2018-03-21 added another member 'boost' to the filter specs.
// This value is used as a factor on 'gain', resulting in the signal
// being amplified by this factor at no additional computational cost,
// which might be desirable when pulling integral signals up to the
// maximal dynamic range. But beware: there are some corner cases with
// splines holding integral data which may cause wrong results
// if 'boost' is too large. Have a look at int_spline.cc and also the
// comments above _process_1d in filter.h

struct iir_filter_specs
{
  bc_code bc ;
  int npoles ;
  const xlf_type * pole ;
  xlf_type tolerance ;
  xlf_type boost ;
  
  iir_filter_specs ( bc_code _bc ,
                     int _npoles ,
                     const xlf_type * _pole ,
                     xlf_type _tolerance ,
                     xlf_type _boost = xlf_type ( 1 )
                   )
  : bc ( _bc ) ,
    npoles ( _npoles ) ,
    pole ( _pole ) ,
    tolerance ( _tolerance ) ,
    boost ( _boost )
  { } ;
} ;
        
/// class iir_filter implements an n-pole forward/backward recursive filter
/// to be used for b-spline prefiltering. It inherits from the 'specs'
/// class for easy initialization.
  
template < typename in_type ,
           typename out_type = in_type ,
           typename _math_type = out_type >
class iir_filter
: public iir_filter_specs
{
  typedef _math_type math_type ;
  
  typedef view_t < 1 , in_type > in_buffer_type ;
  typedef view_t < 1 , out_type > out_buffer_type ;
  
  /// typedef the fully qualified type for brevity, to make the typedefs below
  /// more legible

  typedef iir_filter < in_type , out_type , math_type > filter_type ;

  xlf_type gain ;
  std::vector < int > horizon ;

  // we handle the polymorphism internally, working with method pointers.
  // this saves us having to set up a base class with virtual member functions
  // and inheriting from it.
  
  typedef void  ( filter_type::*p_solve ) ( const in_buffer_type & input ,
                                                 out_buffer_type & output ) const ;
  typedef math_type ( filter_type::*p_icc )   ( const in_buffer_type & buffer , int k ) const ;
  typedef math_type ( filter_type::*p_iccx )  ( const out_buffer_type & buffer , int k ) const ;
  typedef math_type ( filter_type::*p_iacc )  ( const out_buffer_type & buffer , int k ) const ;
  
  // these are the method pointers used:
  
  p_solve _p_solve ; ///< pointer to the solve method
  p_icc   _p_icc ;   ///< pointer to calculation of initial causal coefficient (from in_)
  p_iccx  _p_iccx ;  ///< pointer to calculation of initial causal coefficient (from out_)
  p_iacc  _p_iacc ;  ///< pointer to calculation of initial anticausal coefficient
    
public:

  // this filter runs over the data several times and stores the result
  // of each run back to be picked up by the next run. This has certain
  // implications: if out_type is an integral type, using it to store
  // intermediates will produce quantization errors with every run.
  // this flag signals to the wielding code in filter.h that intermediates
  // need to be stored, so it can avoid the problem by providing a buffer
  // in a 'better' type as output ('output' is used to store intermediates)
  // and converting the data back to the 'real' output afterwards.
  
  static const bool is_single_pass { false } ;
  
  /// calling code may have to set up buffers with additional
  /// space around the actual data to allow filtering code to
  /// 'run up' to the data, shedding margin effects in the
  /// process. For an IIR filter, this is theoretically
  /// infinite , but since we usually work to a specified precision,
  /// we can pass 'horizon' - horizon[0] containing the largest
  /// of the horizon values.
  
  int get_support_width ( ) const
  {
    if ( npoles )
      return horizon [ 0 ] ;
    
    // TODO quick fix. I think this case never occurs, since the filtering
    // code is avoided for npoles < 1
    
    return 64 ;
  }
  
 /// solve() takes two buffers, one to the input data and one to the output space.
 /// The containers must have the same size. It's safe to use solve() in-place.

 void solve ( const in_buffer_type & input , out_buffer_type & output )
 {
   assert ( input.size ( ) == output.size ( ) ) ;
   ( this->*_p_solve ) ( input , output ) ;
 }
 
 /// for in-place operation we use the same filter routine.
 
 void solve ( out_buffer_type & data )
 {
   ( this->*_p_solve ) ( data , data ) ;
 }
 
// I use adapted versions of P. Thevenaz' code to calculate the initial causal and
// anticausal coefficients for the filter. The code is changed just a little to work
// with an iterator instead of a C vector.

private:

/// The code for mirrored BCs is adapted from P. Thevenaz' code, the other routines are my
/// own doing, with aid from a digest of spline formulae I received from P. Thevenaz and which
/// were helpful to verify the code against a trusted source.
///
/// note how, in the routines to find the initial causal coefficient, there are two different
/// cases: first the 'accelerated loop', which is used when the theoretically infinite sum of
/// terms has reached sufficient precision , and the 'full loop', which implements the mathematically
/// precise representation of the limes of the infinite sum towards an infinite number of terms,
/// which happens to be calculable due to the fact that the absolute value of all poles is < 1 and
///
///  lim     n                a
///         sum a * q ^ k =  ---
/// n->inf  k=0              1-q
///
/// first are mirror BCs. This is mirroring 'on bounds',
/// f ( -x ) == f ( x ) and f ( n-1 - x ) == f (n-1 + x)
///
/// note how mirror BCs are equivalent to requiring the first derivative to be zero in the
/// linear algebra approach. Obviously with mirrored data this has to be the case, the location
/// where mirroring occurs is always an extremum. So this case covers 'FLAT' BCs as well
///
/// the initial causal coefficient routines are templated by buffer type, because depending
/// on the circumstances, they may be used either on the input or the output.
  
// TODO format to zimt standard

/// we use accessor classes to access the input and output buffers.
/// To access an input buffer (which remains constant), we use
/// 'as_math_type' which simply provides the ith element cast to
/// math_type. This makes for legible, concise code. We return
/// const math_type from operator[] to make sure X[..] won't be
/// accidentally assigned to.

template < typename buffer_type >
struct as_math_type
{
  const buffer_type & c ;
  
  as_math_type ( const buffer_type & _c )
  : c ( _c )
  { } ;
  
  const math_type operator[] ( int i ) const
  {
    return math_type ( c [ i ] ) ;
  }
} ;

/// the second helper class, as_target, is meant for output
/// buffers. Here we need to read as well as write. Writing is
/// rare, so I use a method 'store' in preference to doing artistry
/// with a proxy. We return const math_type from operator[] to make
/// sure X[..] won't be accidentally assigned to.

template < typename buffer_type >
struct as_target
{
  buffer_type & x ;
  
  as_target ( buffer_type & _x )
  : x ( _x )
  { } ;
  
  const math_type operator[] ( int i ) const
  {
    return math_type ( x [ i ] ) ;
  }
  
  void store ( const math_type & v , const int & i )
  {
    x [ i ] = typename buffer_type::value_type ( v ) ;
  }
} ;

template < class buffer_type >
math_type icc_mirror ( const buffer_type & _c , int k ) const
{
  int M = _c.size ( ) ;
  as_math_type < buffer_type > c ( _c ) ;
  
  math_type z = math_type ( pole[k] ) ;
  math_type zn , z2n , iz ;
  math_type Sum ;
  int  n ;

  if ( horizon[k] < M )
  {
    /* accelerated loop */
    zn = z ;
    Sum = c[0] ;
    for ( n = 1 ; n < horizon[k] ; n++ )
    {
      Sum += zn * c[n] ;
      zn *= z ;
    }
  }
  else
  {
    /* full loop */
    zn = z ;
    iz = math_type ( 1.0 ) / z ;
    z2n = math_type ( pow ( xlf_type ( pole[k] ) , xlf_type ( M - 1 ) ) ) ;
    Sum = c[0] + z2n * c[M - 1] ;
    z2n *= z2n * iz ;
    for ( n = 1 ; n <= M - 2 ; n++ )
    {
      Sum += ( zn + z2n ) * c[n] ;
      zn *= z ;
      z2n *= iz ;
    }
    Sum /= ( math_type ( 1.0 ) - zn * zn ) ;
  } 
 return ( Sum ) ;
}

/// the initial anticausal coefficient routines are always called with the output buffer,
/// so they needn't be templated like the icc routines.
///
/// I still haven't understood the 'magic' which allows to calculate the initial anticausal
/// coefficient from just two results of the causal filter, but I assume it's some exploitation
/// of the symmetry of the data. This code is adapted from P. Thevenaz'.

math_type iacc_mirror ( const out_buffer_type & _c , int k ) const
{
  int M = _c.size ( ) ;
  as_math_type < out_buffer_type > c ( _c ) ;
  
  math_type z = math_type ( pole[k] ) ;

  return ( math_type ( z / ( z * z - math_type ( 1.0 ) ) ) * ( c [ M - 1 ] + z * c [ M - 2 ] ) ) ;
}

/// next are 'antimirrored' BCs. This is the same as 'natural' BCs: the signal is
/// extrapolated via point mirroring at the ends, resulting in point-symmetry at the ends,
/// which is equivalent to the second derivative being zero, the constraint used in
/// the linear algebra approach to calculate 'natural' BCs:
///
/// f ( x ) - f ( 0 ) == f ( 0 ) - f ( -x ) ;
/// f ( x+n-1 ) - f ( n-1 ) == f ( n-1 ) - f (n-1-x)

template < class buffer_type >
math_type icc_natural ( const buffer_type & _c , int k ) const
{
  int M = _c.size ( ) ;
  as_math_type < buffer_type > c ( _c ) ;
  
  math_type z = math_type ( pole[k] ) ;
  math_type zn , z2n , iz ;
  math_type Sum , c02 ;
  int  n ;

  // f ( x ) - f ( 0 ) == f ( 0 ) - f (-x)
  // f ( -x ) == 2 * f ( 0 ) - f (x)
  
  if ( horizon[k] < M )
  {
    c02 = c[0] + c[0] ;
    zn = z ;
    Sum = c[0] ;
    for ( n = 1 ; n < horizon[k] ; n++ )
    {
      Sum += zn * ( c02 - c[n] ) ;
      zn *= z ;
    }
    return ( Sum ) ;
  }
  else {
    zn = z ;
    iz = math_type ( 1.0 ) / z ;
    z2n = math_type ( pow ( xlf_type ( pole[k] ) , xlf_type ( M - 1 )) ) ;
    Sum = math_type ( ( math_type ( 1.0 ) + z ) / ( math_type ( 1.0 ) - z ) )
          * ( c[0] - z2n * c[M - 1] ) ;
    z2n *= z2n * iz ;                                                   // z2n == z^2M-3
    for ( n = 1 ; n <= M - 2 ; n++ )
    {
      Sum -= ( zn - z2n ) * c[n] ;
      zn *= z ;
      z2n *= iz ;
    }
    return ( Sum / ( math_type ( 1.0 ) - zn * zn )) ;
  } 
}

/// I still haven't understood the 'magic' which allows to calculate the initial anticausal
/// coefficient from just two results of the causal filter, but I assume it's some exploitation
/// of the symmetry of the data. This code is adapted from P. Thevenaz' formula.

math_type iacc_natural ( const out_buffer_type & _c , int k ) const
{
  int M = _c.size ( ) ;
  as_math_type < out_buffer_type > c ( _c ) ;
  
  math_type z = math_type ( pole[k] ) ;

  return - math_type ( z / ( ( math_type ( 1.0 ) - z ) * ( math_type ( 1.0 ) - z ) ) ) * ( c [ M - 1 ] - z * c [ M - 2 ] ) ;
}

/// next are reflective BCs. This is mirroring 'between bounds':
///
/// f ( -1 - x ) == f ( x ) and f ( n + x ) == f (n-1 - x)
///
/// I took Thevenaz' routine for mirrored data as a template and adapted it.
/// 'reflective' BCs have some nice properties which make them more suited than mirror BCs in
/// some situations:
/// - the artificial discontinuity is 'pushed out' half a unit spacing
/// - the extrapolated data are just as long as the source data
/// - they play well with even splines

template < class buffer_type >
math_type icc_reflect ( const buffer_type & _c , int k ) const
{
  int M = _c.size ( ) ;
  as_math_type < buffer_type > c ( _c ) ;
  
  math_type z = math_type ( pole[k] ) ;
  math_type zn , z2n , iz ;
  math_type Sum ;
  int  n ;

  if ( horizon[k] < M )
  {
    zn = z ;
    Sum = c[0] ;
    for ( n = 0 ; n < horizon[k] ; n++ )
    {
      Sum += zn * c[n] ;
      zn *= z ;
    }
    return ( Sum ) ;
  }
  else
  {
    zn = z ;
    iz = math_type ( 1.0 ) / z ;
    z2n = math_type ( pow ( xlf_type ( pole[k] ) , xlf_type ( 2 * M )) ) ;
    Sum = 0 ;
    for ( n = 0 ; n < M - 1 ; n++ )
    {
      Sum += ( zn + z2n ) * c[n] ;
      zn *= z ;
      z2n *= iz ;
    }
    Sum += ( zn + z2n ) * c[n] ;
    return c[0] + Sum / ( math_type ( 1.0 ) - zn * zn ) ;
  } 
}

/// I still haven't understood the 'magic' which allows to calculate the initial anticausal
/// coefficient from just one result of the causal filter, but I assume it's some exploitation
/// of the symmetry of the data. I have to thank P. Thevenaz for his formula which let me code:

math_type iacc_reflect ( const out_buffer_type & _c , int k ) const
{
  int M = _c.size ( ) ;
  as_math_type < out_buffer_type > c ( _c ) ;
  
  math_type z = math_type ( pole[k] ) ;

  return c[M - 1] / ( math_type ( 1.0 ) - math_type ( 1.0 ) / z ) ;
}

/// next is periodic BCs. so, f ( x ) = f (x+N)
///
/// Implementing this is more straightforward than implementing the various mirrored types.
/// The mirrored types are, in fact, also periodic, but with a period twice as large, since they
/// repeat only after the first reflection. So especially the code for the full loop is more complex
/// for mirrored types. The down side here is the lack of symmetry to exploit, which made me code
/// a loop for the initial anticausal coefficient as well.

template < class buffer_type >
math_type icc_periodic ( const buffer_type & _c , int k ) const
{
  int M = _c.size ( ) ;
  as_math_type < buffer_type > c ( _c ) ;
  
  math_type z = math_type ( pole[k] ) ;
  math_type zn ;
  math_type Sum ;
  int  n ;

  if ( horizon[k] < M )
  {
    zn = z ;
    Sum = c[0] ;
    for ( n = M - 1 ; n > ( M - horizon[k] ) ; n-- )
    {
      Sum += zn * c[n] ;
      zn *= z ;
    }
   }
  else
  {
    zn = z ;
    Sum = c[0] ;
    for ( n = M - 1 ; n > 0 ; n-- )
    {
      Sum += zn * c[n] ;
      zn *= z ;
    }
    Sum /= ( math_type ( 1.0 ) - zn ) ;
  }
 return Sum ;
}

math_type iacc_periodic ( const out_buffer_type & _c , int k ) const
{
  int M = _c.size ( ) ;
  as_math_type < out_buffer_type > c ( _c ) ;
  
  math_type z = math_type ( pole[k] ) ;
  math_type zn ;
  math_type Sum ;

  if ( horizon[k] < M )
  {
    zn = z ;
    Sum = c[M-1] * z ;
    for ( int n = 0 ; n < horizon[k] ; n++ )
    {
      zn *= z ;
      Sum += zn * c[n] ;
    }
    Sum = -Sum ;
  }
  else
  {
    zn = z ;
    Sum = c[M-1] ;
    for ( int n = 0 ; n < M - 1 ; n++ )
    {
      Sum += zn * c[n] ;
      zn *= z ;
    }
    Sum = z * Sum / ( zn - math_type ( 1.0 ) ) ;
  }
  return Sum ;
}

/// guess the initial coefficient. This tries to minimize the effect
/// of starting out with a hard discontinuity as it occurs with zero-padding,
/// while at the same time requiring little arithmetic effort
///
/// for the forward filter, we guess an extrapolation of the signal to the left
/// repeating c[0] indefinitely, which is cheap to compute:

template < class buffer_type >
math_type icc_guess ( const buffer_type & _c , int k ) const
{
  as_math_type < buffer_type > c ( _c ) ;
  
  return c[0] * math_type ( 1.0 / ( 1.0 - pole[k] ) ) ;
}

// for the backward filter , we assume mirror BC, which is also cheap to compute:

math_type iacc_guess ( const out_buffer_type & c , int k ) const
{
  return iacc_mirror ( c , k ) ;
}

template < class buffer_type >
math_type icc_identity ( const buffer_type & _c , int k ) const
{
  as_math_type < buffer_type > c ( _c ) ;
  
  return c[0] ;
}

math_type iacc_identity ( const out_buffer_type & _c , int k ) const
{
  int M = _c.size ( ) ;
  as_math_type < out_buffer_type > c ( _c ) ;
  
  return c[M-1] ;
}

/// now we come to the solving, or prefiltering code itself.
/// The code is adapted from P. Thevenaz' code.
///
/// I use a 'carry' element, 'X', to carry the result of the recursion
/// from one iteration to the next instead of using the direct implementation 
/// of the recursion formula, which would read the previous value of the 
/// recursion from memory by accessing x[n-1], or x[n+1], respectively.

void solve_gain_inlined ( const in_buffer_type & _c ,
                          out_buffer_type & _x ) const
{
  int M = _c.size ( ) ;
  assert ( _x.size ( ) == M ) ;
  as_math_type < in_buffer_type > c ( _c ) ;
  as_target < out_buffer_type > x ( _x ) ;
  
  if ( M == 1 )
  {
    x.store ( c[0] , 0 ) ;
    return ;
  }
  
  assert ( M > 1 ) ;
  
  // use a buffer of one math_type for the recursion (see below)

  math_type X ;
  math_type p = math_type ( pole[0] ) ;
  
  // use first filter pole, applying overall gain in the process
  // of consuming the input.
  // Note that the application of the gain is performed during the processing
  // of the first (maybe the only) pole of the filter, instead of running a separate
  // loop over the input to apply it before processing starts.
  
  // note how the gain is applied to the initial causal coefficient. This is
  // equivalent to first applying the gain to the input and then calculating
  // the initial causal coefficient from the processed input.
  
  X = math_type ( gain ) * ( this->*_p_icc ) ( _c , 0 ) ;
  x.store ( X , 0 ) ;

  /* causal recursion */
  // the gain is applied to each input value as it is consumed
  
  for ( int n = 1 ; n < M ; n++ )
  {
    // KFJ 2019-02-12 tentative use of fma
#ifdef USE_FMA
    math_type cc = math_type ( gain ) * c[n] ;
    X = fma ( X , p , cc ) ;
#else
    X = math_type ( gain ) * c[n] + p * X ;
#endif
    x.store ( X , n ) ;
  }
  
  // now the input is used up and won't be looked at any more; all subsequent
  // processing operates on the output.
  
  /* anticausal initialization */
  
  X = ( this->*_p_iacc ) ( _x , 0 ) ;
  x.store ( X , M - 1 ) ;
  
  /* anticausal recursion */
  for ( int n = M - 2 ; 0 <= n ; n-- )
  {
    X = p * ( X - x[n] ) ;
    x.store ( X , n ) ;
  }
  
  // for the remaining poles, if any, don't apply the gain
  // and process the result from applying the first pole
  
  for ( int k = 1 ; k < npoles ; k++ )
  {
    p = math_type ( pole[k] ) ;
    /* causal initialization */
    X = ( this->*_p_iccx ) ( _x , k ) ;
    x.store ( X , 0 ) ;
    
    /* causal recursion */
    for ( int n = 1 ; n < M ; n++ )
    {
    // KFJ 2019-02-12 tentative use of fma
#ifdef USE_FMA
      math_type xx = x[n] ;
      X = fma ( X , p , xx ) ;
#else
      X = x[n] + p * X ;
#endif
      x.store ( X , n ) ;
    }
    
    /* anticausal initialization */
    X = ( this->*_p_iacc ) ( _x , k ) ;
    x.store ( X , M - 1 ) ;
    
    /* anticausal recursion */
    for ( int n = M - 2 ; 0 <= n ; n-- )
    {
      X = p * ( X - x[n] ) ;
      x.store ( X , n ) ;
    }
  }
}

/// solve_identity is used for spline degrees 0 and 1. In this case
/// there are no poles to apply, but if the operation is not in-place
/// and/or there is a 'boost' factor which is different from 1, the
/// data are copied and/or amplified with 'boost'.

void solve_identity ( const in_buffer_type & _c ,
                           out_buffer_type & _x ) const
{
  int M = _c.size ( ) ;
  assert ( _x.size ( ) == M ) ;
  as_math_type < in_buffer_type > c ( _c ) ;
  as_target < out_buffer_type > x ( _x ) ;
  
  if ( boost == xlf_type ( 1 ) )
  {
    // boost is 1, check if operation is not in-place
    if ( ( void* ) ( _c.data ( ) ) != ( void* ) ( _x.data ( ) ) )
    {
      // operation is not in-place, copy input to output
      for ( int n = 0 ; n < M ; n++ )
      {
        x.store ( c[n] , n ) ;
      }
    }
  }
  else
  {
    // we have a boost factor, so we apply it.
    math_type factor = math_type ( boost ) ;
    
    for ( int n = 0 ; n < M ; n++ )
    {
      x.store ( factor * c[n] , n ) ;
    }
  }
}

/// The last bit of work left is the constructor. This simply passes
/// the specs to the base class constructor, as iir_filter inherits
/// from the specs type.

public:
  
  iir_filter ( const iir_filter_specs & specs )
  : iir_filter_specs ( specs )
{
  // TODO we have a problem if the gain is getting very large, as it happens
  // for high spline degrees. The iir_filter attenuates the signal to next-to-nothing,
  // then it's amplified back to the previous amplitude. This degrades the signal,
  // most noticeably when the numeric type is lo-fi, since there are operations involving
  // both the attenuated and unattenuated data ('squashing').
  
  if ( npoles < 1 )
  {
    // zero poles means there's nothing to do but possibly
    // copying the input to the output, which solve_identity
    // will do if the operation isn't in-place.
    _p_solve = & filter_type::solve_identity ;
    return ;
  }
  
  // calculate the horizon for each pole, this is the number of iterations
  // the filter must perform on a unit pulse to decay below 'tolerance'
  
  // If tolerance is 0 (or negative) we set 'horizon' to MAX_INT. This
  // will have the effect of making it larger than M, or at least so
  // large that there won't be a difference between the accelerated and
  // the full loop. We might use a smaller value which still guarantees
  // the complete decay.

  for ( int i = 0 ; i < npoles ; i++ )
  {
    if ( tolerance > 0 )
      horizon.push_back (   ceil ( log ( tolerance )
                          / log ( std::abs ( pole[i] ) ) ) ) ;
    else
      horizon.push_back ( INT_MAX ) ; // TODO quick fix, think about it
  }

  // contrary to my initial implementation I use per-axis gain instead of
  // cumulating the gain for all axes. This may perform slightly worse, but
  // is more stable numerically and simplifies the code.
  
  gain = boost * overall_gain ( npoles , pole ) ;
  _p_solve = & filter_type::solve_gain_inlined ;

  // while the forward/backward IIR iir_filter in the solve_... routines is the same for all
  // boundary conditions, the calculation of the initial causal and anticausal coefficients
  // depends on the boundary conditions and is handled by a call through a method pointer
  // in the solve_... routines. Here we fix these method pointers:

  if ( bc == MIRROR )
  {     
    _p_icc = & filter_type::icc_mirror<in_buffer_type> ;
    _p_iccx = & filter_type::icc_mirror<out_buffer_type> ;
    _p_iacc = & filter_type::iacc_mirror ;
  }
  else if ( bc == NATURAL )
  {     
    _p_icc = & filter_type::icc_natural<in_buffer_type> ;
    _p_iccx = & filter_type::icc_natural<out_buffer_type> ;
    _p_iacc = & filter_type::iacc_natural ;
  }
  else if ( bc == PERIODIC )
  {
    _p_icc = & filter_type::icc_periodic<in_buffer_type> ;
    _p_iccx = & filter_type::icc_periodic<out_buffer_type> ;
    _p_iacc = & filter_type::iacc_periodic ;
  }
  else if ( bc == REFLECT )
  {
    _p_icc = & filter_type::icc_reflect<in_buffer_type> ;
    _p_iccx = & filter_type::icc_reflect<out_buffer_type> ;
    _p_iacc = & filter_type::iacc_reflect ;
  }
  else if ( bc == ZEROPAD )
  {
    _p_icc = & filter_type::icc_identity<in_buffer_type> ;
    _p_iccx = & filter_type::icc_identity<out_buffer_type> ;
    _p_iacc = & filter_type::iacc_identity ;
  }
  else if ( bc == GUESS )
  {
    _p_icc = & filter_type::icc_guess<in_buffer_type> ;
    _p_iccx = & filter_type::icc_guess<out_buffer_type> ;
    _p_iacc = & filter_type::iacc_guess ;
  }
  else
  {
    throw not_supported ( "boundary condition not supported by filter" ) ;
  }
}

} ; // end of class iir_filter

/// class to provide recursive filtering, using 'iir_filter' above.
/// both prefilter.h and general_filter.h use this base object.
/// The actual filter object has to interface with the data handling
/// routine ('present', see filter.h). So this class functions as an
/// adapter, combining the code needed to set up adequate buffers
/// and creation of the actual IIR filter itself.
/// The interface to the data handling routine is provided by
/// inheriting from class buffer_handling

// KFJ 2019-04-16 added default for _vsize template argument

template < template < typename , size_t > class _vtype ,
           typename _math_ele_type ,
           size_t _vsize =
             vector_traits<_math_ele_type>::size >
struct recursive_filter
: public buffer_handling < _vtype , _math_ele_type , _vsize > ,
  public iir_filter < _vtype < _math_ele_type , _vsize > >
{
  // provide this type for queries
  
  typedef _math_ele_type math_ele_type ;

  // we'll use a few types from the buffer_handling type

  typedef buffer_handling < _vtype , _math_ele_type , _vsize > buffer_handling_type ;
  
  using typename buffer_handling_type::vtype ;
  using buffer_handling_type::vsize ;
  using buffer_handling_type::init ;

  // instances of class recursive_filter hold the buffer:
  
  array_t < 1 , vtype > buffer ;

  // the filter's 'solve' routine has the workhorse code to filter
  // the data inside the buffer:
  
  typedef _vtype < _math_ele_type , _vsize > simdized_math_type ;
  typedef iir_filter < simdized_math_type > filter_type ;
  using filter_type::solve ;
  
  // by defining arg_type, we allow code to infer what type of
  // argument initializer the filter takes
  
  typedef iir_filter_specs arg_type ;
  
  // the constructor invokes the filter's constructor,
  // sets up the buffer and initializes the buffer_handling
  // component to use the whole buffer to accept incoming and
  // provide outgoing data.

  recursive_filter ( const iir_filter_specs & specs , size_t size )
  : filter_type ( specs ) ,
    buffer ( size )
  {
    // operate in-place and use the whole buffer to receive and
    // deliver data
    
    init ( buffer , buffer ) ;
  } ;

  // operator() simply delegates to the filter's 'solve' routine,
  // which filters the data in the buffer.
  
  void operator() ( )
  {
    solve ( buffer , buffer ) ;
  }
  
  // factory function to provide a filter with the same set of
  // parameters, but possibly different data types. this is used
  // for processing of 1D data, where the normal buffering mechanism
  // may be sidestepped

  template < typename in_type ,
             typename out_type = in_type ,
             typename math_type = out_type >
  static iir_filter < in_type , out_type , math_type >
         get_raw_filter ( const iir_filter_specs & specs )
  {
    return iir_filter < in_type , out_type , math_type >
           ( specs ) ;
  }
  
} ;

/// forward_backward_recursive_filter applies one or more pairs of simple
/// recursive filters to the input. Each pair consists of:
/// 1. the forward filter: x[n] = g * x[n] + p * x[n-1]
/// 2. the backward filter: x[n] = -p * x[n] + p * x[n+1]
/// where 'p' is the 'pole' of the filter (e.g from poles.h) and g is
/// the 'gain', which is computed as g = ( 1 - p ) * ( 1 - 1 / p );
/// multiplication of the input with the gain in the first stage
/// makes sure that subsequent convolution with the corresponding
/// reconstruction kernel will restore the original signal.
/// The pair of filters used here, in sum, has no phase shift
/// (the phase shift of the forward filter is precisely matched by
/// an inverse phase shift of the backward filter due to the reversal
/// of the processing direction).
/// input and output must be equally-sized arrays of compatible types
/// (meaning, in_value_type and out_value_type must have the same number
/// of channels). bcv holds a set of boundary conditions, one per
/// dimension. pv is a std::vector holding the filter poles poles,
/// 'tolerance' is the acceptable error, defaulting to
/// a very conservative value. Try and resist from passing zero here,
/// which results in great efforts to produce a 'mathematically correct'
/// result - the result from using the conservative default will differ
/// very little from the one obtained with 'zero tolerance', but is usually
/// faster to compute.
/// Next, a 'boost' value can be passed which will be applied once
/// as a multiplicative factor. If any value apart from -1 is passed for
/// 'axis', filtering will be limited to the indicated axis. Per default,
/// the filters will be applied to all axes in turn.
/// Finally, njobs defines how many jobs will be used to multithread
/// the operation.
/// While b-spline prefiltering uses 'poles' from poles.h, this generalized
/// routine will take any poles you pass. But note that the magnitude of
/// each pole should be below 1. If you use negative poles, the effect will
/// be a high-pass filter, as it is used for b-spline prefiltering. Using
/// positive poles will blur the signal very effectively. Also note that
/// use of small negative poles will amplify the signal strongly. This
/// effect can be seen when prefiltering high-degree b-splines: the resulting
/// signal will have much higher energy. Make sure your data types can deal
/// with the increased dynamic.

template < std::size_t dimension ,
           typename in_value_type ,
           typename out_value_type ,
           typename math_ele_type =
                    ET < PROMOTE ( in_value_type , out_value_type ) > ,
           size_t vsize =
                  vector_traits < math_ele_type > :: size
         >
void forward_backward_recursive_filter (
                 const
                 view_t
                   < dimension ,
                     in_value_type > & input ,
                 view_t
                   < dimension ,
                     out_value_type > & output ,
                 xel_t < bc_code , dimension > bcv ,
                 std::vector < xlf_type > poles ,
                 xlf_type tolerance = -1 ,
                 xlf_type boost = xlf_type ( 1 ) ,
                 int axis = -1 , // -1: apply along all axes
                 int njobs = default_njobs )
{
  if ( output.shape != input.shape )
    throw shape_mismatch
     ( "forward_backward_recursive_filter: input and output shape must match" ) ;

  if ( tolerance < 0 )
    tolerance = std::numeric_limits < math_ele_type > :: epsilon() ;

  int npoles = poles.size() ;
  
  if ( npoles < 1 )
  {
    // if npoles < 1, there is no filter to apply, but we may need
    // to apply 'boost' and/or copy input to output. We use 'amplify'
    // for the purpose, which multithreads the operation (if it is at
    // all necessary). I found this is (slightly) faster than doing the
    // job in a single thread - the process is mainly memory-bound, so
    // the gain is moderate.

    amplify < dimension , in_value_type , out_value_type , math_ele_type >
      ( input , output , math_ele_type ( boost ) ) ;
      
    return ;
  }
  

  typedef recursive_filter < simdized_type ,
                             math_ele_type ,
                             vsize
                           > filter_type ;

  // now call the 'wielding' code in filter.h

  if ( axis == -1 )
  {
    // user has passed -1 for 'axis', apply the same filter along all axes

    std::vector < iir_filter_specs > vspecs ;

    // package the arguments to the filter; one set of arguments
    // per axis of the data

    for ( int axis = 0 ; axis < dimension ; axis++ )
    {
      vspecs.push_back
        ( iir_filter_specs
          ( bcv [ axis ] , npoles , poles.data() , tolerance , 1 ) ) ;
    }

    // 'boost' is only applied to dimension 0, since it is meant to
    // affect the whole data set just once, not once per axis.

    vspecs [ 0 ] . boost = boost ;
    
    filter
    < in_value_type , out_value_type , dimension , filter_type > 
    ( input , output , vspecs , njobs ) ;
  }
  else
  {
    // user has passed a specific axis, apply filter only to this axis

    assert ( axis >=0 && axis < dimension ) ;

    filter
    < in_value_type , out_value_type , dimension , filter_type > 
    ( input , output , axis ,
      iir_filter_specs (
        bcv [ axis ] , npoles , poles.data() , tolerance , boost ) ,
      njobs ) ;
  }
}

END_ZIMT_SIMD_NAMESPACE
HWY_AFTER_NAMESPACE() ;

#endif // sentinel
