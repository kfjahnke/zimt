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

/*! \file brace.h

    \brief This file provides code for 'bracing' a b-spline's coefficient
    array.

    This code originated in my library 'vspline'; this is a port
    using zimt data types

    Note that this isn't really user code, it's code used by class zimt::bspline.
    
    Inspired by libeinspline, I wrote code to 'brace' the spline coefficients. The concept is
    this: while the IIR filter used to calculate the coefficients has infinite support (though
    arithmetic precision limits this in real-world applications), the evaluation of the spline
    at a specific location only looks at a small window of coefficients (compact, finite support).
    This fact can be exploited by taking note of how large the support area is and providing
    a few more coefficients in a frame around the 'core' coefficients to allow the evaluation
    to proceed without having to check for boundary conditions. While the difference is not
    excessive (the main computational cost is the actual evaluation itself), it's still
    nice to be able to code the evaluation without boundary checking, which makes the code
    very straightforward and legible.

    There is another aspect to bracing: In my implementation of vectorized evaluation,
    the window into the coefficient array used to pick out coefficients to evaluate at
    a specific location is coded as a set of offsets from it's center. This way,
    several such windows can be processed in parallel. This mechanism can only function
    efficiently in a braced coefficient array, since it would otherwise have to give up
    if any of the windows accessed by the vector of coordinates had members outside the
    (unbraced) coefficient array and submit the coordinate vector to individual processing.
    I consider the logic to code this and the loss in performance too much of a bother
    to go down this path; all my evaluation code uses braced coefficient arrays. Of course
    the user is free to omit bracing, but then they have to use their own evaluation
    code.

    What's in the brace? Of course this depends on the boundary conditions chosen.
    In vspline, I offer code for several boundary conditions, but most have something
    in common: the original, finite sequence is extrapolated into an infinite periodic
    signal. With straight PERIODIC boundary conditions, the initial sequence is
    immediately followed and preceded by copies of itself. The other boundary conditions
    mirror the signal in some way and then repeat the mirrored signal periodically.
    Using boundary conditions like these, both the extrapolated signal and the
    coefficients share the same periodicity and mirroring. There is one exception:
    'natural' boundary conditions use point-mirroring on the bounds. With this extrapolation,
    the extrapolated value can *not* be obtained by a coordinate manipulation. The method
    of bracing the spline does still function, though. So with bracing, we can provide
    b-splines with 'natural' boundary conditions as well, but here we are limited to
    evaluation inside the spline's defined range.
    
    There are two ways of arriving at a coeffcient array: We can start from the
    extrapolated signal, pick a section large enough to make margin effects vanish
    (due to limited arithmetic precision), prefilter it and pick out a subsection containing
    the 'core' coefficients and their support. Alternatively, we can work only on the core
    coefficients, calculate suitable initial causal and anticausal coeffcients (where the
    calculation considers the extrapolated signal, which remains implicit), and apply the
    filter with these known initial coefficients. vspline uses the latter approach.
    Once the 'core' coefficients are known, the brace is filled.
  
    The bracing can be performed without any solver-related maths by simply copying
    (possibly trivially modified) slices of the core coefficients to the margin area.

    Since the bracing mainly requires copying data or trivial maths we can do the operations
    on higher-dimensional objects, like slices of a volume. To efficiently code these operations
    we make use of vigra's multi-math facility and it's slice array method, which makes
    these subarrays easily available.
*/


// TODO: while this is convenient, it's not too fast, as it's neither multithreaded nor
// vectorized. Still in most 'normal' scenarios the execution time is negligible, and since
// the code is trivial, it autovectorizes well.
// 
// TODO: there are 'pathological' cases where one brace is larger than the other brace
// and the width of the core together. These cases can't be handled for all bracing modes
// and will result in an exception.

#ifndef ZIMT_BRACE_H
#define ZIMT_BRACE_H

#include "common.h"
#include "array.h"

namespace zimt {

/// class bracer encodes the entire bracing process. Note that contrary
/// to my initial implementation, class bracer is now used exclusively
/// for populating the frame around a core area of data. It has no code
/// to determine which size a brace/frame should have. This is now
/// determined in class bspline, see especially class bspline's methods
/// get_left_brace_size(), get_right_brace_size() and setup_metrics().

template < std::size_t D , typename _value_type >
struct bracer
{
  typedef _value_type value_type ;
  static const std::size_t dimension = D ;
  typedef view_t < dimension , value_type > view_type ;
  typedef xel_t < std::size_t , dimension > shape_type ;

/// apply the bracing to the array, performing the required copy/arithmetic operations
/// to the 'frame' around the core. This routine performs the operation along axis 'dim'.
/// This is also the routine to be used for explicitly extrapolating a signal:
/// you place the data into the center of a larger array, and pass in the sizes of the
/// 'empty' space which is to be filled with the extrapolated data.
///
/// the bracing is done one-left-one-right, to avoid corner cases as best as posible.
/// This makes it possible to have signals which are shorter than the brace and still
/// produce a correct brace for them.

  static void apply ( view_type & a , // containing array
                      bc_code bc ,    // boundary condition code
                      int lsz ,       // space to the left which needs to be filled
                      int rsz ,       // ditto, to the right
                      int axis )      // axis along which to apply bracing 
  {
    // std::cout << "bracer receives lsz " << lsz
    //           << " rsz " << rsz << std::endl ;
    int w = a.shape [ axis ] ;  // width of containing array along axis 'axis'
    int m = w - ( lsz + rsz ) ; // width of 'core' array

    if ( m < 1 )                // has to be at least 1
      throw shape_mismatch ( "combined brace sizes must be at least one less than container size" ) ;

    if ( m == 1 )
    {
      // KFJ 2018-02-10
      // special case: the core has only shape 1 in this direction.
      // if this is so, we fill the brace with the single slice in the
      // middle, no matter what the boundary condition code may say.
      
      for ( int s = 0 ; s < w ; s++ )
      {
        if ( s != lsz )
          a.slice ( axis , s ) .copy_data ( a.slice ( axis , lsz ) ) ;
      }
      return ;
    }
    
    if (    ( lsz > m + rsz )
         || ( rsz > m + lsz ) )
    {
      // not enough data to fill brace
      if ( bc == PERIODIC || bc == NATURAL || bc == MIRROR || bc == REFLECT )
        throw std::out_of_range ( "each brace must be smaller than the sum of it's opposite brace and the core's width" ) ;
    }

    int l0 = lsz - 1 ; // index of innermost empty slice on the left; like begin()
    int r0 = lsz + m ; // ditto, on the right

    int lp = l0 + 1 ;  // index of leftmost occupied slice (p for pivot)
    int rp = r0 - 1 ;  // index of rightmost occupied slice

    int l1 = -1 ;     // index one before outermost empty slice to the left
    int r1 = w ;      // index one after outermost empty slice on the right; like end()

    int lt = l0 ;     // index to left target slice
    int rt = r0 ;     // index to right target slice ;

    int ls , rs ;     // indices to left and right source slice, will be set below

    int ds = 1 ;      // step for source index, +1 == forẃard, used for all mirroring modes
                      // for periodic bracing, it's set to -1.

    switch ( bc )
    {
      case PERIODIC :
      {
        ls = l0 + m ;
        rs = r0 - m ;
        ds = -1 ;      // step through source in reverse direction
        break ;
      }
      case NATURAL :
      case MIRROR :
      {
        ls = l0 + 2 ;
        rs = r0 - 2 ;
        break ;
      }
      case CONSTANT :
      case REFLECT :
      {
        ls = l0 + 1 ;
        rs = r0 - 1 ;
        break ;
      }
      case ZEROPAD :
      {
        break ;
      }
      default:
      {
        throw not_supported
          ( "boundary condition not supported by zimt::bracer" ) ;
        break ;
      }
    }

    for ( int i = std::max ( lsz , rsz ) ; i > 0 ; --i )
    {
      if ( lt > l1 )
      {
        switch ( bc )
        {
          case PERIODIC :
          case MIRROR :
          case REFLECT :
          {
            // with these three bracing modes, we simply copy from source to target
            a.slice ( axis , lt ) .copy_data ( a.slice ( axis , ls ) ) ;
            break ;
          }
          case NATURAL :
          {
            // here, we subtract the source slice from twice the 'pivot'
            auto source = a.slice ( axis , ls ) ;
            auto target = a.slice ( axis , lt ) ;
            auto pivot = a.slice ( axis , lp ) ;
            auto f = [] ( value_type a , value_type b )
            {
              return a + a - b ;
            } ;
            target.combine ( f , pivot , source ) ;
            break ;
          }
          case CONSTANT :
          {
            // here, we repeat the 'pivot' slice
            a.slice ( axis , lt ) .copy_data ( a.slice ( axis , lp ) ) ;
            break ;
          }
          case ZEROPAD :
          {
            // fill with 0
            a.slice ( axis , lt ) .set_data ( value_type() ) ;
            break ;
          }
          default :
            // default: leave untouched
            break ;
        }
        --lt ;
        ls += ds ;
      }
      if ( rt < r1 )
      {
        // essentially the same, but with rs instead of ls, etc.
        switch ( bc )
        {
          case PERIODIC :
          case MIRROR :
          case REFLECT :
          {
            // with these three bracing modes, we simply copy from source to target
            a.slice ( axis , rt ) .copy_data ( a.slice ( axis , rs ) ) ;
            break ;
          }
          case NATURAL :
          {
            // here, we subtract the source slice from twice the 'pivot'
            auto source = a.slice ( axis , rs ) ;
            auto target = a.slice ( axis , rt ) ;
            auto pivot = a.slice ( axis , rp ) ;
            auto f = [] ( value_type a , value_type b )
            {
              return a + a - b ;
            } ;
            target.combine ( f , pivot , source ) ;
            break ;
          }
          case CONSTANT :
          {
            // here, we repeat the 'pivot' slice
            a.slice ( axis , rt ) .copy_data ( a.slice ( axis , rp ) ) ;
            break ;
          }
          case ZEROPAD :
          {
            // fill with 0
            a.slice ( axis , rt ) .set_data ( value_type() ) ;
            break ;
          }
          default :
            // default: leave untouched
            break ;
        }
        ++rt ;
        rs -= ds ;
      }
    }
    // a.traverse ( [](const value_type & v)
    //                { std::cout << "** " << v << std::endl ;
    //                  return v ;
    //               } ) ;
  }
  
  /// This overload of 'apply' braces along all axes in one go.

  static void apply 
    ( view_type& a ,          // target array, containing the core and (empty) frame
      xel_t < bc_code , dimension > bcv ,  // boundary condition codes
      shape_type left_corner ,  // sizes of left braces
      shape_type right_corner ) // sizes of right braces
  {
    for ( int dim = 0 ; dim < dimension ; dim++ )
      apply ( a , bcv[dim] , left_corner[dim] , right_corner[dim] , dim ) ;
  }
} ;

} ; // end of namespace zimt

#endif // ZIMT_BRACE_H
