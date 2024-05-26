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

/*! \file extrapolate.h

    \brief extrapolation of 1D data sets with specific boundary conditions
*/

// This is a port from vspline/extrapolate.h

#ifndef ZIMT_EXTRAPOLATE_H
#define ZIMT_EXTRAPOLATE_H

#include "common.h"

namespace zimt
{

/// struct extrapolator is a helper class providing extrapolated
/// values for a 1D buffer indexed with possibly out-of-range indices.
/// The extrapolated value is returned by value. boundary conditions
/// PERIODIC , MIRROR , REFLECT, NATURAL and CONSTANT are currently
/// supported.
/// An extrapolator is set up by passing the boundary condition code
/// (see common.h) and a const reference to the 1D data set, coded
/// as a 1D vigra::MultiArrayView. The view has to refer to valid data
/// for the time the extrapolator is in use.
/// Now the extrapolator object can be indexed with arbitrary indices,
/// and it will return extrapolated values. The indexing is done with
/// operator() rather than operator[] to mark the semantic difference.
/// Note how buffers with size 1 are treated specially for some
/// boundary conditions: here we simply return the value at index 0.

template < class buffer_type >
struct extrapolator
{
  const buffer_type & buffer ;
  typedef typename buffer_type::value_type value_type ;
  
  // we handle the polymorphism by calling the specific extrapolation
  // routine via a method pointer. This enables us to provide a uniform
  // interface without having to set up a virtual base class and inherit
  // from it.
  
  typedef value_type ( extrapolator::*p_xtr ) ( int i ) const ;
  p_xtr _p_xtr ;
  
  value_type extrapolate_mirror ( int i ) const
  {
    int w = buffer.size() - 1 ;

    if ( w == 0 )
      return buffer[0] ;

    i = std::abs ( i ) ;
    if ( i >= w )
    {
      i %= 2 * w ;
      i -= w ;
      i = std::abs ( i ) ;
      i = w - i ;
    }
    return buffer [ i ] ;
  }
  
  value_type extrapolate_natural ( int i ) const
  {
    int w = buffer.size() - 1 ;

    if ( w == 0 )
      return buffer[0] ;

    if ( i >= 0 && i <= w )
      return buffer[i] ;
    
    int sign = i < 0 ? -1 : 1 ;
    i = std::abs ( i ) ;
    
    int p = 2 * w ;
    int np = i / p ;
    int r = i % p ;
    value_type help ;
    
    if ( r <= w )
    {
      help = buffer[r] - buffer[0] ;
      help += np * 2 * ( buffer[w] - buffer[0] ) ;
      help *= sign ;
      help += buffer [ 0 ] ;
      return help ;
    }
    
    r = 2 * w - r ;
    help = 2 * ( buffer [ w ] - buffer [ 0 ] );
    help -= ( buffer[r] - buffer [ 0 ] ) ;
    help += np * 2 * ( buffer[w] - buffer[0] ) ;
    help *= sign ;
      help += buffer [ 0 ] ;
    return help ;
  }
  
  value_type extrapolate_reflect ( int i ) const
  {
    int w = buffer.size() ;
    if ( i < 0 )
      i = -1 - i ;
    if ( i >= w )
    {
      i %= 2 * w ;
      if ( i >= w )
        i = 2 * w - i - 1 ;
    }
    return buffer [ i ] ;
  }
  
  value_type extrapolate_periodic ( int i ) const
  {
    int w = buffer.size() ;
    
    if ( w == 1 )
      return buffer[0] ;

    if ( i < 0 || i >= w )
    {
      i %= w ;
      if ( i < 0 )
        i += w ;
    }
    return buffer [ i ] ;
  }
  
  value_type extrapolate_clamp ( int i ) const
  {
    if ( i < 0 )
      return buffer [ 0 ] ;
    int w = buffer.size() - 1 ;
    if ( i >= w )
      return buffer [ w ] ;
    return buffer [ i ] ;
  }
  
  value_type extrapolate_zeropad ( int i ) const
  {
    return value_type ( 0 ) ;
  }
  
  /// class extrapolator's constructor takes the boundary
  /// condition code and a const reference to the buffer.
  /// the specific extrapolation routine is picked in the
  /// case switch and assigned to the method pointer which
  /// will be invoked by operator().
  
  extrapolator ( zimt::bc_code bc , const buffer_type & _buffer )
  : buffer ( _buffer )
  {
    switch ( bc )
    {
      case zimt::PERIODIC :
        _p_xtr = & extrapolator::extrapolate_periodic ;
        break ;
      case zimt::REFLECT :
        _p_xtr = & extrapolator::extrapolate_reflect ;
        break ;
      case zimt::NATURAL :
        _p_xtr = & extrapolator::extrapolate_natural ;
        break ;
      case zimt::MIRROR :
        _p_xtr = & extrapolator::extrapolate_mirror ;
        break ;
      case zimt::GUESS :
      case zimt::CONSTANT :
        _p_xtr = & extrapolator::extrapolate_clamp ;
        break ;
      case zimt::ZEROPAD :
        _p_xtr = & extrapolator::extrapolate_zeropad ;
        break ;
      default:
        throw zimt::not_implemented
              ( "extrapolator: unknown boundary condition" ) ;
        break ;
    }
  }
  
  /// operator() uses the specific extrapolation method to provide
  /// a value for position i.
  
  value_type operator() ( const int & i ) const
  {
    return (this->*_p_xtr) ( i ) ;
  }
} ;

} ; // namespace zimt

#endif // #define ZIMT_EXTRAPOLATE_H
