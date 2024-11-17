/************************************************************************/
/*                                                                      */
/*    zimt - a set of generic tools for creation and evaluation      */
/*              of uniform b-splines                                    */
/*                                                                      */
/*            Copyright 2015 - 2023 by Kay F. Jahnke                    */
/*                                                                      */
/*    The git repository for this software is at                        */
/*                                                                      */
/*    https://bitbucket.org/kfj/zimt                                 */
/*                                                                      */
/*    Please direct questions, bug reports, and contributions to        */
/*                                                                      */
/*    kfjahnke+zimt@gmail.com                                        */
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

/*! \file basis.h

    \brief Code to calculate the value of the B-spline basis function
    and it's derivatives.

    This file begins with some collateral code used to 'split' coordinates
    into an integral part and a small real remainder. This split is used
    in b-spline evaluation and fits thematically with the remainder of the
    file, which deals with the basis function.

    zimt only uses the B-spline basis function values at multiples
    of 0.5. With these values it can construct it's evaluators which in turn
    are capable of evaluating the spline at real coordinates. Since these
    values aren't 'too many' but take some effort to calculate precisely,
    they are provided as precalculated constants in poles.h, which also holds
    the prefilter poles.
    
    The basis function values at half unit steps are used for evaluation
    via a 'weight matrix'. This is a matrix of numbers which can yield the
    value of a b-spline by employing a simple fixed-time matrix/vector
    multiplication (which also vectorizes well). In a similar way, a
    'basis_functor' can be set up from these values which can provide
    the value of the basis function for arbitrary real arguments.

    for a discussion of the b-spline basis function, have a look at
    http://www.cs.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/B-spline/bspline-basis.html
*/

// #ifndef VSPLINE_BASIS_H
// #define VSPLINE_BASIS_H

#if defined(ZIMT_BASIS_H) == defined(HWY_TARGET_TOGGLE)
  #ifdef ZIMT_BASIS_H
    #undef ZIMT_BASIS_H
  #else
    #define ZIMT_BASIS_H
  #endif

#include "common.h"
#include "xel.h"
#include "array.h"

// poles.h has precomputed basis function values sampled at n * 1/2.
// These values were calculated to very high precision in a separate
// program (see bootstrap.cc) using GNU GMP, GSL and BLAS.

#include "poles.h"

HWY_BEFORE_NAMESPACE() ;
BEGIN_ZIMT_SIMD_NAMESPACE(zimt)

/// coordinates are split into an integral part and a remainder. this is
/// used for weight generation, and also for calculating the basis function
/// value. The split is done differently for odd and even splines.
///
/// Note how the initial call to std::floor produces a real type, which
/// is used to subtract from 'v', yielding the remainder in 'fv'. Only after
/// having used this real representation of the integral part, it is cast
/// to an integral type by assigning it to 'iv'. This is the most efficient
/// route, better than producing an integral-typed integral part directly
/// and subtracting that from 'v', which would require another conversion.
/// Technically, one might consider this 'split' as a remainder division by 1.

template < typename ic_t , typename rc_t >
void odd_split ( rc_t v , ic_t& iv , rc_t& fv )
{
  rc_t fl_i = floor ( v ) ;
  fv = v - fl_i ;
  // assign ( iv , fl_i ) ;
  iv = fl_i ;
}

// roll out the split function for zimt::xel_ts

template < typename ic_t , typename rc_t , int N >
void odd_split ( zimt::xel_t < rc_t , N > v ,
                 zimt::xel_t < ic_t , N > & iv ,
                 zimt::xel_t < rc_t , N > & fv )
{
  for ( int d = 0 ; d < N ; d++ )
    odd_split ( v[d] , iv[d] , fv[d] ) ;
}

/// for even splines, the integral part is obtained by rounding. when the
/// result of rounding is subtracted from the original coordinate, a value
/// between -0.5 and 0.5 is obtained which is used for weight generation.

// TODO: there is an issue here: the lower limit for an even spline
// is -0.5, which should be rounded *towards* zero, but std::round rounds
// away from zero. The same applies to the upper limit, which should
// also be rounded towards zero, not away from it. Currently I am working
// around the issue by increasing the spline's headroom by 1 for even splines,
// but I'd like to be able to use rounding towards zero. It might be argued,
// though, that potentially producing out-of-range access by values which
// are only just outside the range is cutting it a bit fine and the extra
// headroom for even splines makes the code more robust, so accepting the
// extra headroom would be just as acceptable as the widened right brace for
// some splines which saves checking the incoming coordinate against
// the upper limit. The code increasing the headroom is in bspline.h,
// in bspline's constructor just befor the call to setup_metrics.

template < typename ic_t , typename rc_t >
void even_split ( rc_t v , ic_t& iv , rc_t& fv )
{
  rc_t fl_i = round ( v ) ;
  fv = v - fl_i ;
  // assign ( iv , fl_i ) ;
  iv = fl_i ;
}

template < typename ic_t , typename rc_t , int N >
void even_split ( zimt::xel_t < rc_t , N > v ,
                  zimt::xel_t < ic_t , N > & iv ,
                  zimt::xel_t < rc_t , N > & fv )
{
  for ( int d = 0 ; d < N ; d++ )
    even_split ( v[d] , iv[d] , fv[d] ) ;
}

/// bspline_basis_2 yields the value of the b-spline basis function
/// at multiples of 1/2, for which zimt has precomputed values.
/// Instead of passing real values x which are multiples of 1/2, this
/// routine takes the doubled argument, so instead of calling it with
/// x = 0.5, you call it with x2 = 1.
///
/// this is a helper routine to offer convenient access to zimt's
/// precomputed basis function values, and it also handles the special
/// case of degree 0, x = -1/2, where the b-spline basis function
/// is not symmetric and yields 1.
///
/// Inside zimt, this routine is only used by calculate_weight_matrix.
/// User code should rarely need it, but I hold on to it as a separate entity.
/// The code also handles derivatives. This is done by recursion, which is
/// potentially very slow (for high derivatives), so this routine should
/// only be used for 'moderate' derivative values. For fast access to the
/// basis function's derivatives, use of basis_functor is
/// recommended instead.

template < class real_type >
real_type bspline_basis_2 ( int x2 , int degree , int derivative = 0 )
{
  real_type result ;

  if ( derivative == 0 )
  {
    if ( degree == 0 )
    {
      if ( x2 == -1 || x2 == 0 )
        result = real_type ( 1 ) ;
      else
        result = real_type ( 0 ) ;
    }
    else if ( abs ( x2 ) > degree )
    {
      result = real_type ( 0 ) ;
    }
    else
    {
      // here we pick the precomputed value:

      const xlf_type * pk
        = zimt_constants::precomputed_basis_function_values [ degree ] ;

      result = real_type ( pk [ abs ( x2 ) ] ) ;
    }
    return result ;
  }
  else
  {
    // recurse. this will only produce recursion until 'derivative' becomes
    // zero, at which point precomputed values are picked.

    --derivative;
    return   bspline_basis_2<real_type> ( x2 + 1 , degree - 1 , derivative )
           - bspline_basis_2<real_type> ( x2 - 1 , degree - 1 , derivative ) ;
  }
}

/// a 'weight matrix' is used to calculate a set of weights for a given remainder
/// part of a real coordinate. The 'weight matrix' is multipled with a vector
/// containing the power series of the given remainder, yielding a set of weights
/// to apply to a window of b-spline coefficients.
///
/// The routine 'calculate_weight_matrix' originated from vigra, but I rewrote it
/// to avoid calculating values of derivatives of the basis function by using
/// recursion. The content of the 'weight matrix' is now calculated directly
/// with a forward iteration starting from precomputed basis function values,
/// the derivatives needed are formed by repeatedly differencing these values.
/// The recursive formulation in vigra makes sense, since the degree of the spline
/// is a template argument in vigra and the recursive formulation can be evaluated
/// at compile-time, allowing for certain optimizations. But in zimt, spline
/// degree is a run-time variable, and zimt offers calculation of splines of
/// degrees up to (currently) 45, which aren't feasibly calculable by recursion,
/// with a recursion calling itself twice for every step: this would take a very
/// long time or exceed the system's capacity. Analyzing the recursive implementation,
/// it can be seen that it produces a great many redundant calculations, which soon
/// exceed reasonable limits. With vigra's maximal spline degree the load is just
/// about manageable, but beyond that (degree 19 or so at the time of this writing)
/// it's clearly no-go.
///
/// The forward iteration is reasonably fast, even for high spline degrees,
/// while my previous implementation did slow down very noticeably from, say,
/// degree 20, making it unusable for high spline degrees. I really only noticed
/// the problem after raising the maximal degree zimt can handle, following the
/// rewrite of bootstrap.cc using arbitrary-precision maths.
///
/// A weight matrix is also used by 'basis_functor', in a similar way to how
/// it's used for weight generation.

// template < class target_type >
// void calculate_weight_matrix ( zimt::array_t < 2 , target_type > & res )
// {
//   int order = res.shape[0] ;
//   int degree = order - 1 ;
//   int derivative = order - res.shape[1] ;
//   
//   // guard against impossible parameters
//   
//   if ( derivative >= order )
//     return ;
// 
//   xlf_type faculty = 1 ; // why xlf_type? because integral types overflow
//   
//   // we do the calculations for each row of the weight matrix in the same type
//   // as the precomputed basis function values, only casting down to 'target_type'
//   // once the row is ready
//   
//   xlf_type der_line [ degree + 1 ] ;
//   
//   for ( int row = 0 ; row < order - derivative ; row++ )
//   {
//     if ( row > 1 )
//       faculty *= row ;
// 
//     // obtain pointers to beginning of row and past it's end
//     
//     xlf_type * p_first = der_line ;
//     xlf_type * p_end = der_line + degree + 1 ;
//     
//     // now we want to pick basis function values. The first row to go into
//     // the weight matrix gets basis function values for 'degree', the next
//     // one values for degree-1, but differenced once, the next one values
//     // for degree-2, differenced twice etc.
//     // Picking the values is done so that for odd degrees, basis function
//     // values for whole x are picked, for even spline degrees, values
//     // 1/2 + n, n E N are picked. Why so? When looking at the recursion
//     // used in bspline_basis_2, you can see that each step of the recursion
//     // forms the difference between a value 'to the left' and a value 'to
//     // the right' of the current position (x2-1 and x2+1). If you follow
//     // this pattern, you can see that, depending on the degree, the recursion
//     // will run down to either all even or all odd x2, so this is what we
//     // pick, since the other values will not participate in the result
//     // at all.
//     // Note how, to pick the basis function values, we use bspline_basis_2,
//     // which 'knows' how to get the right precomputed values. Contrary to my
//     // previous implementation, it is *not* used to calculate the derivatives,
//     // it is only used as a convenient way to pick the precomputed values.
// 
//     int m = degree - derivative - row ;
//     
//     if ( m == 0 )
//     {
//       der_line[0] = 1 ;
//       ++p_first ;
//     }
//     else if ( degree & 1 )
//     {
//       for ( int x2 = - m + 1 ; x2 <= m - 1 ; x2 += 2 )
//       {
//         *(p_first++) = bspline_basis_2<xlf_type> ( x2 , m ) ;
//       }
//     }
//     else
//     {
//       for ( int x2 = - m ; x2 <= m ; x2 += 2 )
//       {
//         *(p_first++) = bspline_basis_2<xlf_type> ( x2 , m ) ;
//       }
//     }
//     
//     // fill the remainder of the line with zeroes
//     
//     xlf_type * p = p_first ;
//     while ( p < p_end )
//       *(p++) = 0 ;
//     
//     // now we have the initial basis function values. We need to differentiate
//     // the sequence, possibly several times. We have the initial values
//     // flush with the line's left bound, so we perform the differentiation
//     // back to front, and after the last differentiation the line is full.
//     // Note how this process might be abbreviated further by exploiting
//     // symmetry relations (most rows are symmetric or antisymmetric).
//     // I refrain from doing so (for now) and I suspect that this may even be
//     // preferable in respect to error propagation (TODO: check).
// 
//     for ( int d = m ; d < degree ; d++ )
//     {
//       // deposit first difference after the last basis function value
//       
//       xlf_type * put = p_first ;
//       
//       // and form the difference to the value before it.
//       
//       xlf_type * pick = put - 1 ;
//       
//       // now form differences back to front
//       
//       while ( pick >= der_line )
//       {
//         *put = *pick - *put ;
//         --put ;
//         --pick ;
//       }
//       
//       // since we have nothing left 'to the left', where another zero
//       // would be, we simply invert the sign (*put = 0 - *put).
//       
//       *put = - *put ;
//       
//       // The next iteration has to start one place further to the right,
//       // since there is now one more value in the line
//       
//       p_first++ ;
//     }
//       
//     // the row is ready and can now be assigned to the corresponding row of
//     // the weight matrix after applying the final division by 'faculty' and,
//     // possibly, downcasting to 'target_type'.
// 
//     // we store to a array_t, which is row-major, so storing as we do
//     // places the results in memory in the precise order in which we want to
//     // use them later on in the weight calculation.
//     
//     for ( int k = 0 ; k < degree + 1 ; k++ )
// 
//       res [ { k , row } ] = der_line[k] / faculty ;
//   }
// }

/// basis_functor is an object producing the b-spline basis function value
/// for given arguments, or optionally a derivative of the basis function.
/// While basis_functor can produce single basis function values for single
/// arguments, it can also produce a set of basis function values for a
/// given 'delta'. This set is a unit-spaced sampling of the basis function
/// sampled at n + delta for all n E N. Such samplings are used to evaluate
/// b-splines; they constitute the set of weights which have to be applied
/// to a set of b-spline coefficients to form the weighted sum which is the
/// spline's value at a given position.

/// The calculation is done by using a 'weight matrix'. The columns of the
/// weight matrix contain the coefficients for the partial polynomials defining
/// the basis function for the corresponding interval. In 'general' evaluation,
/// all partial polynomials are evaluated. To obtain single basis function
/// values, we pick out a single column only. By evaluating the partial
/// polynomial for this slot, we obtain a single basis function value.
///
/// This functor provides the value(s) in constant time and there is no
/// recursion. Setting up the functor costs a bit of time (for calculating
/// the 'weight matrix'), evaluating it merely evaluates the partial
/// polynomial(s) which is quick by comparison. So this is the way to go if
/// basis function values are needed - especially if there is a need for
/// several values of a given basis function. I refrain from giving a 'one-shot'
/// function using a basis_functor - this is easily achieved by coding
///
/// b = basis_functor ( degree , derivative ) ( x ) ;
///
/// ... which does the trick, but 'wastes' the weight matrix.
///
/// The weight matrix and all variables use 'math_type' which defaults to
/// xlf_type, zimt's most exact type. By instantiating with a lesser type,
/// The computation can be done more quickly, but less precisely.

template < typename math_type = xlf_type >
struct basis_functor
{
  zimt::array_t < 2 , math_type > weight_matrix ;
  zimt::array_t < 1 , xlf_type > der_line ;
  const int degree ;
  const int order ;
  const int derivative ;
  
  void calculate_weight_matrix()
  {
    // guard against impossible parameters
    
    if ( derivative >= order )
      return ;

    xlf_type faculty = 1 ; // why xlf_type? because integral types overflow
    
    // we do the calculations for each row of the weight matrix in the same
    // type as the precomputed basis function values, only casting down to
    // 'target_type' once the row is ready
    
    // xlf_type der_line [ degree + 1 ] ;
    
    for ( int row = 0 ; row < order - derivative ; row++ )
    {
      if ( row > 1 )
        faculty *= row ;

      // obtain pointers to beginning of row and past it's end
      
      xlf_type * p_first = der_line.data() ;
      xlf_type * p_end = p_first + degree + 1 ;
      
      // now we want to pick basis function values. The first row to go into
      // the weight matrix gets basis function values for 'degree', the next
      // one values for degree-1, but differenced once, the next one values
      // for degree-2, differenced twice etc.
      // Picking the values is done so that for odd degrees, basis function
      // values for whole x are picked, for even spline degrees, values
      // 1/2 + n, n E N are picked. Why so? When looking at the recursion
      // used in bspline_basis_2, you can see that each step of the recursion
      // forms the difference between a value 'to the left' and a value 'to
      // the right' of the current position (x2-1 and x2+1). If you follow
      // this pattern, you can see that, depending on the degree, the recursion
      // will run down to either all even or all odd x2, so this is what we
      // pick, since the other values will not participate in the result
      // at all.
      // Note how, to pick the basis function values, we use bspline_basis_2,
      // which 'knows' how to get the right precomputed values. Contrary to my
      // previous implementation, it is *not* used to calculate the derivatives,
      // it is only used as a convenient way to pick the precomputed values.

      int m = degree - derivative - row ;
      
      if ( m == 0 )
      {
        der_line[0] = 1 ;
        ++p_first ;
      }
      else if ( degree & 1 )
      {
        for ( int x2 = - m + 1 ; x2 <= m - 1 ; x2 += 2 )
        {
          *(p_first++) = bspline_basis_2<xlf_type> ( x2 , m ) ;
        }
      }
      else
      {
        for ( int x2 = - m ; x2 <= m ; x2 += 2 )
        {
          *(p_first++) = bspline_basis_2<xlf_type> ( x2 , m ) ;
        }
      }
      
      // fill the remainder of the line with zeroes
      
      xlf_type * p = p_first ;
      while ( p < p_end )
        *(p++) = 0 ;
      
      // now we have the initial basis function values. We need to differentiate
      // the sequence, possibly several times. We have the initial values
      // flush with the line's left bound, so we perform the differentiation
      // back to front, and after the last differentiation the line is full.
      // Note how this process might be abbreviated further by exploiting
      // symmetry relations (most rows are symmetric or antisymmetric).
      // I refrain from doing so (for now) and I suspect that this may even be
      // preferable in respect to error propagation (TODO: check).

      for ( int d = m ; d < degree ; d++ )
      {
        // deposit first difference after the last basis function value
        
        xlf_type * put = p_first ;
        
        // and form the difference to the value before it.
        
        xlf_type * pick = put - 1 ;
        
        // now form differences back to front
        
        while ( pick >= der_line.data() )
        {
          *put = *pick - *put ;
          --put ;
          --pick ;
        }
        
        // since we have nothing left 'to the left', where another zero
        // would be, we simply invert the sign (*put = 0 - *put).
        
        *put = - *put ;
        
        // The next iteration has to start one place further to the right,
        // since there is now one more value in the line
        
        p_first++ ;
      }
        
      // the row is ready and can now be assigned to the corresponding row of
      // the weight matrix after applying the final division by 'faculty' and,
      // possibly, downcasting to 'target_type'.

      // we store to a array_t, which is row-major, so storing as we do
      // places the results in memory in the precise order in which we want to
      // use them later on in the weight calculation.
      
      for ( int k = 0 ; k < degree + 1 ; k++ )
      {
        weight_matrix [ { k , row } ] = der_line[k] / faculty ;
      }
    }
  }

  basis_functor ( int _degree , int _derivative = 0 )
  : weight_matrix ( { _degree + 1 , _degree + 1 - _derivative } ) ,
    der_line ( _degree + 1 ) ,
    degree ( _degree ) ,
    order ( _degree + 1 ) ,
    derivative ( _derivative )
  {
    calculate_weight_matrix() ;
  } ;

//   basis_functor & operator= ( const basis_functor & other )
//   {
//     weight_matrix = other.weight_matrix ;
//     degree = other.degree ;
//     return *this ;
//   }
// 
//   basis_functor ( const basis_functor & other )
//   {
//     weight_matrix = other.weight_matrix ;
//     degree = other.degree ;
//   }

  /// operator() taking a column index and a remainder. If these values
  /// are known already, the only thing left to do is the evaluation of
  /// the partial polynomial. Note that this overload is not safe for
  /// arbitrary x, it's assumed that calling code makes sure no invalid
  /// arguments are passed - as in the overload below.
  
  // TODO might generalize to allow vectorized operation

  math_type operator() ( int x , math_type delta ) const
  {
    math_type result = weight_matrix ( x , 0 ) ;
    math_type power = 1 ;
    
    // remaining rows, if any, refine result
    
    for ( int row = 1 ; row < weight_matrix.shape[1] ; row++ )
    {
      power *= delta ;
      result += power * weight_matrix ( x , row ) ;
    }

    return result ;
  }
  
  /// operator() taking an arbitrary argument. This is the overload which
  /// will likely be called from user code. The argument is clamped and
  /// split, the split value is fed to the previous overload.
  /// This routine provides a single result for a single argument and
  /// is used if the basis function itself needs to be evaluated, which
  /// doesn't happen much inside zimt. Access to sets of basis function
  /// values used as weights in b-spline evaluation is coded below.
  
  math_type operator() ( math_type rx ) const
  {
    int x ;
    math_type delta ;
    
    // we split the argument into an integer and a small real remainder

    if ( degree & 1 )
      odd_split ( rx , x , delta ) ;
    else
    {
      if ( degree == 0 )
      {
        if ( rx >= -.5 && rx < 0.5 )
          return 1 ;
        return 0 ;
      }
      even_split ( rx , x , delta ) ;
    }

    x = degree / 2 - x ;
    
    if ( x < 0 || x >= weight_matrix.shape[0] )
    {
      return 0 ;
    }
    
    return operator() ( x , delta ) ;
  }
  
  /// operator() overload to produce a set of weights for a given
  /// delta in [-.5,.5] or [0,1]. This set of weights is needed if
  /// a b-spline has to be evaluated at  some coordinate k + delta,
  /// where k is a whole number. For this evaluation, a set of
  /// coefficients has to be multiplied with a set of weights, and
  /// the products summed up. So this routine provides the set of
  /// weights. It deposits weights for the given delta at the location
  /// 'result' points to. target_type and delta_type may be fundamentals
  /// or simdized types.
  /// note that, if 'delta' is zero, 'power' will also be zero, and
  /// therefore everything after the initialization of the result with
  /// the first row of the weight matrix is futile. One might consider
  /// testing for this special case, but this would cost extra cycles.
  /// Instead, below, there is an overload which does not take a 'delta'.
  /// Why so? The special case of delta == zero occurs certainly when
  /// discrete coordinates are used, but rarely in 'normal' operation,
  /// when the whole point of using a spline is to evaluate at real
  /// coordinates with some delta.

  template < class target_type , class delta_type >
  void operator() ( target_type* result , const delta_type & delta ) const
  {
    target_type power ( delta ) ;
    const auto * factor_it = weight_matrix.data() ; //  weight_matrix.begin() ;
    const auto * end = factor_it + weight_matrix.shape.prod() ; // weight_matrix.end() ;

    // the result is initialized with the first row of the 'weight matrix'.
    // We save ourselves multiplying it with delta^0.
 
    for ( int c = 0 ; c <= degree ; c++ )
    {
      result[c] = *factor_it ;
      ++factor_it ;
    }
    
    if ( degree )
    {
      for ( ; ; )
      {
        for ( int c = 0 ; c <= degree ; c++ )
        {
         // KFJ 2019-02-12 tentative use of fma
#ifdef USE_FMA
          target_type factor ( *factor_it ) ;
          target_type rr = result[c] ;
          result[c] = fma ( power , factor , rr ) ;
#else
          result[c] += power * *factor_it ;
#endif
          ++factor_it ;
        }
        if ( factor_it == end )
        {
          // avoid next multiplication if exhausted, break now
          break ;
        }
         // otherwise produce next power(s) of delta(s)
        power *= target_type ( delta ) ;
      }
    }
  }

  // a different way of calling the overload of operator() above;
  // easier to handle instatiation from the python module.

  template < class target_type , class delta_type >
  void weights ( target_type* result , const delta_type & delta ) const
  {
    operator() ( result , delta ) ;
  }

  /// overload without delta, implies (all) delta(s) == 0.
  /// this is used for evaluation with discrete coordinates

  template < class target_type >
  void operator() ( target_type* result ) const
  {
    auto factor_it = weight_matrix.begin() ;

    // the result is initialized with the first row of the 'weight matrix'.
    // We save ourselves multiplying it with delta^0.
 
    for ( int c = 0 ; c <= degree ; c++ )
    {
      result[c] = *factor_it ;
      ++factor_it ;
    }
  }

} ;

/// this function deposits the reconstruction kernel in the array
/// 'kernel'. This kernel can be used to convolve a
/// set of coefficients, to obtain the original signal. This is a
/// convenience function which merely picks the right values from
/// the precomputed values in precomputed_basis_function_values.
/// if 'odd' is passed false, the result is an even kernel. This
/// kernel can't be used for reconstruction (.5 phase shift), but
/// it's handy to get values half a unit step from the knot points.

template < class target_type >
void get_kernel ( const int & degree ,
                  zimt::view_t < 1 , target_type > & kernel ,
                  const bool & odd = true )
{
  assert ( degree >= 0 && degree <= zimt_constants::max_degree ) ;
  if ( odd )
  {
    int headroom = degree / 2 ;
    int ksize = headroom * 2 + 1 ;
    assert ( kernel.size() == ksize ) ;

    // pick the precomputed basis function values for the kernel.
    // Note how the values in precomputed_basis_function_values
    // (see poles.h) are provided at half-unit steps, hence the
    // index acrobatics.

    for ( int k = - headroom ; k <= headroom ; k++ )
    {
      int pick = 2 * std::abs ( k ) ;
      kernel [ k + headroom ]
      = zimt_constants
        ::precomputed_basis_function_values [ degree ]
          [ pick ] ;
    }
  }
  else // produce an even kernel
  {
    int headroom = ( degree + 1 ) / 2 ;
    int ksize = headroom * 2 ;
    assert ( kernel.size() == ksize ) ;

    for ( int k = 0 ; k < headroom ; k++ )
    {
      int pick = 2 * k + 1 ;
      kernel [ headroom - k - 1 ]
      = kernel [ headroom + k ]
      = zimt_constants
        ::precomputed_basis_function_values [ degree ]
          [ pick ] ;
    }
  }
}

/// Implementation of the Cox-de Boor recursion formula to calculate
/// the value of the bspline basis function for arbitrary real x.
/// This code was taken from vigra but modified to take the spline degree
/// as a parameter. Since this routine uses recursion, it's usefulness
/// is limited to smaller degrees.
///
/// This routine operates in real and calculates the basis function value
/// for arbitrary real x, but it suffers from cumulating errors, especially
/// when the recursion is deep, so the results are not uniformly precise.
///
/// This code is expensive for higher spline orders because the routine
/// calls itself twice recursively, so the performance is 2^N with the
/// spline's degree. Luckily there are ways around using this routine at all
/// - whenever we need the b-spline basis function value in zimt, it is at
/// multiples of 1/2, and poles.h has precomputed values for all spline
/// degrees covered by zimt. The value of the basis function itself
/// can be obtained by using a basis_functor, which performs in
/// fixed time and is set up quickly.
///
/// I leave this code in here for reference purposes - it's good to have
/// another route to the basis function values, see self_test.cc.

template < class real_type >
real_type cdb_bspline_basis ( real_type x , int degree , int derivative = 0 )
{
  if ( degree == 0 )
  {
    if ( derivative == 0 )
        return ( x < real_type(0.5) && real_type(-0.5) <= x )
               ? real_type(1.0)
               : real_type(0.0) ;
    else
        return real_type(0.0);
  }
  if ( derivative == 0 )
  {
    real_type n12 = real_type((degree + 1.0) / 2.0);
    return (     ( n12 + x )
                * cdb_bspline_basis<real_type> ( x + real_type(0.5) , degree - 1 , 0 )
              +   ( n12 - x )
                * cdb_bspline_basis<real_type> ( x - real_type(0.5) , degree - 1 , 0 )
            )
            / degree;
  }
  else
  {
    --derivative;
    return   cdb_bspline_basis<real_type> ( x + real_type(0.5) , degree - 1 , derivative )
           - cdb_bspline_basis<real_type> ( x - real_type(0.5) , degree - 1 , derivative ) ;
  }
}

/// Gaussian approximation to B-spline basis function. This routine
/// approximates the basis function of degree spline_degree for real x.
/// I checked for all degrees up to 45. The partition of unity quality of the
/// resulting reconstruction filter is okay for larger degrees, the cumulated
/// error over the covered interval is quite low. Still, as the basis function
/// is never actually evaluated in zimt (whenever it's needed, it is needed
/// at n * 1/2 and we have precomputed values for that) there is not much point
/// in having this function around. I leave the code in for now.

template < typename real_type >
real_type gaussian_bspline_basis_approximation ( real_type x , int degree )
{
  // heuristic: for slightly better fit use use 
  // real_type sigma = 0.021310018257 + ( degree + 1 ) / 12.0 ;
  real_type sigma = ( degree + 1 ) / 12.0 ;
  return   real_type(1.0)
         / sqrt ( real_type(2.0 * M_PI) * sigma )
         * exp ( - ( x * x ) / ( real_type(2.0) * sigma ) ) ;
}

END_ZIMT_SIMD_NAMESPACE
HWY_AFTER_NAMESPACE() ;

#endif // sentinel
