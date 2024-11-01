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

/*! \file convolve.h

    \brief separable convolution of nD arrays

    This is a port from vspline/convolve.h
    
    This file provides the core filtering code for convolution, which
    can be used by itself to filter 1D arrays, or is used with the
    'wielding' code in filter.h to filter nD arrays. The latter use is
    what's used throughout most of zimt, since it provides automatic
    multithreading and vectorization by buffering the data and applying
    the 1D code to the buffer.
    
    The implementation of convolution in this file can safely operate
    in-place. The actual convolution operation is done using a small
    kernel-sized circular buffer, which is multiplied with an adequately
    shifted and rotated representation of the kernel. This is done
    avoiding conditionals as best as possible. The 1D data are extrapolated
    with one of the boundary condition codes known to class extrapolator
    (see extrapolate.h). This is done transparently by putting extrapolated
    data into the small circular buffer where this is needed.
    
    The code is trivial insofar as it only uses indexed assignments, addition
    and multiplication. So it can operate on a wide variety of data types,
    prominently among them SIMD vector types.
    
    Note how I use the kernel front-to-back, in the same forward sequence as
    the data it is applied to. This is contrary to the normal convention of
    using the kernel values back-to-front. Inside zimt, where only
    symmetrical kernels are used, this makes no difference, but when vpline's
    convolution code is used for other convolutions, this has to be kept in
    mind.

    A function template meant to be called from user code is right at the
    end of the file. It sets up all the internal structures needed to
    perform a separable convolution of an array with given filter
    coefficients, performs the convolution and removes the 'scaffolding'.

*/

// #ifndef ZIMT_CONVOLVE_H
// #define ZIMT_CONVOLVE_H

#include "common.h"
#include "filter.h"
#include "extrapolate.h"

#if defined(ZIMT_CONVOLVE_H) == defined(HWY_TARGET_TOGGLE)
  #ifdef ZIMT_CONVOLVE_H
    #undef ZIMT_CONVOLVE_H
  #else
    #define ZIMT_CONVOLVE_H
  #endif

BEGIN_ZIMT_SIMD_NAMESPACE(zimt)

namespace zimt {

/// fir_filter_specs holds the parameters for a filter performing
/// a convolution along a single axis. In zimt, the place where
/// the specifications for a filter are fixed and the place where
/// it is finally created are far apart: the filter is created
/// in the separate worker threads. So this structure serves as
/// a vehicle to transport the arguments.
/// Note the specification of 'headroom': this allows for
/// non-symmetrical and even kernels. When applying the kernel
/// to obtain output[i], the kernel is applied to
/// input [ i - headroom ] , ... , input [ i - headroom + ksize - 1 ]
  
struct fir_filter_specs
{
  zimt::bc_code bc ;     // boundary conditions
  int ksize ;               // kernel size
  int headroom ;            // part of kernel 'to the left'
  const xlf_type * kernel ; // pointer to kernel values
  
  fir_filter_specs ( zimt::bc_code _bc ,
                     int _ksize ,
                     int _headroom ,
                     const xlf_type * _kernel )
  : bc ( _bc ) ,
    ksize ( _ksize ) ,
    headroom ( _headroom ) ,
    kernel ( _kernel )
  {
    assert ( headroom < ksize ) ;
  } ;
} ;

/// class fir_filter provides the 'solve' routine which convolves
/// a 1D signal with selectable extrapolation. Here, the convolution
/// kernel is applied to the incoming signal and the result is written
/// to the specified output location. Note that this operation
/// can be done in-place, but input and output may also be different.
/// While most of the time this routine will be invoked by class
/// convolve (below), it is also directly used by the specialized
/// code for 1D filtering.
/// Note how we conveniently inherit from the specs class. This also
/// enables us to use an instance of fir_filter or class convolve
/// as specs argument to create further filters with the same arguments.

// TODO: some kernels are symmetric, which might be exploited.

// TODO: special code for filters with 0-valued coefficients, like
//       sinc-derived half band filters

template < typename in_type ,
           typename out_type = in_type ,
           typename _math_type = out_type >
struct fir_filter
: public fir_filter_specs
{
  // this filter type does not need storage of intermediate results.

  static const bool is_single_pass { true } ;
  
  typedef zimt::view_t < 1 , in_type > in_buffer_type ;
  typedef zimt::view_t < 1 , out_type > out_buffer_type ;
  typedef _math_type math_type ;
  
  // we put all state data into a single area of memory called 'reactor'.
  // The separate parts holding the small circular buffer, the repeated
  // kernel and the tail buffer are implemented as views to 'reactor'.
  // This way, all data participating in the arithmetics are as close
  // together in memory as possible.
  // note how the current implementation does therefore hold the kernel
  // values in the 'reactor' as simdized types (if math_type is simdized).
  // this may be suboptimal, since the kernel values might be supplied
  // as scalars and could be kept in a smaller area of memory.
  // TODO: investigate
  
  using allocator_t
  = typename zimt::allocator_traits < math_type > :: type ;

  // zimt::array_t < 1 , math_type , allocator_t > reactor ;
  zimt::array_t < 1 , math_type > reactor ;
  zimt::view_t < 1 , math_type > circular_buffer ;
  zimt::view_t < 1 , math_type > kernel_values ;
  zimt::view_t < 1 , math_type > tail_buffer ;
    

  fir_filter ( const fir_filter_specs & specs )
  : fir_filter_specs ( specs ) ,
    reactor ( zimt::xel_t<std::size_t,1> ( specs.ksize * 4 ) )
  {
    circular_buffer = reactor.window
      ( zimt::xel_t<std::size_t,1> ( 0 ) , zimt::xel_t<std::size_t,1> ( ksize ) ) ;
                                          
    kernel_values = reactor.window
      ( zimt::xel_t<std::size_t,1> ( ksize ) , zimt::xel_t<std::size_t,1> ( ksize * 3 ) ) ;

    tail_buffer = reactor.window
      ( zimt::xel_t<std::size_t,1> ( ksize * 3 ) , zimt::xel_t<std::size_t,1> ( ksize * 4 ) ) ;

    for ( int i = 0 ; i < ksize ; i++ )
      kernel_values [ i ] = kernel_values [ i + ksize ] = kernel [ i ] ;
  } ;
  
  /// calling code may have to set up buffers with additional
  /// space around the actual data to allow filtering code to
  /// 'run up' to the data, shedding margin effects in the
  /// process. We stay on the safe side and return the width
  /// of the whole kernel, which is always sufficient to
  /// provide safe runup.
  
  int get_support_width() const
  {
    return ksize ;
  }
  
  /// public 'solve' routine. This is for calls 'from outside',
  /// like when this object is used by itself, not as a base class
  /// of class convolve below.
  /// an extrapolator for the boundary condition code 'bc'
  /// (see fir_filter_specs) is made, then the call is delegated
  /// to the protected routine below which accepts an extrapolator
  /// on top of input and output.
  
  void solve ( const in_buffer_type & input ,
               out_buffer_type & output )
  {
    int size = output.size() ;
    extrapolator < in_buffer_type > source ( bc , input ) ;
    solve ( input , output , source ) ;
  }

protected:

  /// protected solve routine taking an extrapolator on top of
  /// input and output. This way, the derived class (class convolve)
  /// can maintain an extrapolator fixed to it's buffer and reuse
  /// it for subsequent calls to this routine.
  /// we use the following strategy:
  /// - keep a small circular buffer as large as the kernel
  /// - have two kernels concatenated in another buffer
  /// - by pointing into the concatenated kernels, we can always
  ///   have ksize kernel values in sequence so that this sequence
  ///   is correct for the values in the circular buffer.
  /// this strategy avoids conditionals as best as possible and
  /// should be easy to optimize. the actual code is a bit more
  /// complex to account for the fact that at the beginning and
  /// end of the data, a few extrapolated values are used. The
  /// central loop can directly read from input without using the
  /// extrapolator, which is most efficient.
  
  void solve ( const in_buffer_type & input ,
               out_buffer_type & output ,
               const extrapolator < in_buffer_type > & source )
  {
    if ( ksize < 1 )
    { 
      // if kernel size is zero or even negative, then,
      // if operation isn't in-place, copy input to output
      
      if ( (void*) ( input.data() ) != (void*) ( output.data() ) )
      {
        for ( std::ptrdiff_t i = 0 ; i < output.size() ; i++ )
          output[i] = out_type ( input[i] ) ;
      }

      return ; // we're done prematurely
    }
    else if ( ksize == 1 )
    {
      // for kernel size 1 we perform the multiplication of the
      // single kernel value with the input in a simple loop without
      // using the circular buffering mechanism below. This is an
      // optimization, the circular buffer code can also handle
      // single-value kernels.
      
      math_type factor ( kernel[0] ) ;
      
      for ( std::ptrdiff_t i = 0 ; i < output.size() ; i++ )
        output[i] = out_type ( factor * math_type ( input[i] ) ) ;
      
      return ; // we're done prematurely
    }

    int si = - headroom ; // read position
    int ti = 0 ;          // store position

    // initialize circular buffer using the extrapolator
    // note: initially I coded to fetch only the first 'headroom'
    // values from the extrapolator, then up to ksize straight
    // from 'input'. but this is *not* correct: 'input' may by
    // very small, and with a large kernel we also need the
    // extrapolator further on after the input is already
    // consumed. So this is the correct way of doing it:

    for ( int i = 0 ; i < ksize ; i++ , si++ )
      circular_buffer[i] = source ( si ) ;
    
    // see how many full cycles we can run, directly accessing
    // 'input' without resorting to extrapolation
    
    int size = output.size() ;
    int leftover = size - si ;
    int full_cycles = 0 ;
    if ( leftover > 0 )
      full_cycles = leftover / ksize ;

    // stash the trailing extrapolated values: we want to be able
    // to operate in-place, and if we write to the buffer we can't
    // use the extrapolator over it anymore. note how we only fill
    // in ksize - headroom values. this is all we'll need, the buffer
    // may be slightly larger.
    
    int ntail = ksize - headroom ;
    int z = size ;
    for ( int i = 0 ; i < ntail ; i++ , z++ )
      tail_buffer[i] = source ( z ) ;
    
    // central loop, reading straight from input without extrapolation

    for ( int cycle = 0 ; cycle < full_cycles ; cycle++ )
    {
      auto p_kernel = kernel_values.data() + ksize ;
      auto p_data = circular_buffer.data() ;
      
      for ( int i = 0 ; i < ksize ; )
      {
        // perform the actual convolution
        // TODO: exploit symmetry
        
        math_type result = circular_buffer[0] * p_kernel[0] ;

      // KFJ 2019-02-12 tentative use of fma

#ifdef USE_FMA
        for ( int j = 1 ; j < ksize ; j++ )
          result = fma ( circular_buffer[j] , p_kernel[j] , result ) ;
#else
        for ( int j = 1 ; j < ksize ; j++ )
          result += circular_buffer[j] * p_kernel[j] ;
#endif

        // stash result
        
        output [ ti ] = out_type ( result ) ;
        
        // fetch next input value
        
        * p_data = input [ si ] ;
        
        // adjust pointers and indices
        
        ++ si ;
        ++ ti ;
        ++ i ;

        if ( i == ksize )
          break ;

        ++ p_data ;
        -- p_kernel ;
      }
    }
    
    // produce the last few values, resorting to tail_buffer
    // where it is necessary

    while ( ti < size )
    {
      auto p_kernel = kernel_values.data() + ksize ;
      auto p_data = circular_buffer.data() ;
      
      for ( int i = 0 ; i < ksize && ti < size ; i++ )
      {
        math_type result = circular_buffer[0] * p_kernel[0] ;
        for ( int j = 1 ; j < ksize ; j++ )
          result += circular_buffer[j] * p_kernel[j] ;

        output [ ti ] = out_type ( result ) ;

        if ( si < size )
          // still sweet
          * p_data = input [ si ] ;
        else
          // input used up, use stashed extrapolated values
          * p_data = tail_buffer [ si - size ] ;

        ++ si ;
        ++ ti ;
        
        ++ p_data ;
        -- p_kernel ;
      }
    }
  }
} ;

/// class convolve provides the combination of class fir_filter
/// above with a vector-friendly buffer. Calling code provides
/// information about what should be buffered, the data are sucked
/// into the buffer, filtered, and moved back from there.
/// The operation is orchestrated by the code in filter.h, which
/// is also used to 'wield' the b-spline prefilter. Both operations
/// are sufficiently similar to share the wielding code.

template < template < typename , size_t > class _vtype ,
           typename _math_ele_type ,
           size_t _vsize >
struct convolution_filter
: public buffer_handling < _vtype , _math_ele_type , _vsize > ,
  public zimt::fir_filter < _vtype < _math_ele_type , _vsize > >
{
  // provide this type for queries
  
  typedef _math_ele_type math_ele_type ;

  // we'll use a few types from the buffer_handling type

  typedef buffer_handling < _vtype , _math_ele_type , _vsize >
    buffer_handling_type ;
  
  using typename buffer_handling_type::vtype ;
  using buffer_handling_type::vsize ;
  using buffer_handling_type::init ;

  // instances of class convolve hold the buffer as state:
  
//   using allocator_t
//   = typename zimt::allocator_traits < vtype > :: type ;
//   
//   typedef zimt::array_t < 1 ,  vtype , allocator_t > buffer_type ;
  typedef zimt::array_t < 1 ,  vtype > buffer_type ;
  typedef zimt::view_t < 1 ,  vtype > buffer_view_type ;
  
  buffer_type buffer ;

  // and also an extrapolator, which is fixed to the buffer

  extrapolator < buffer_view_type > buffer_extrapolator ;
  
  // the filter's 'solve' routine has the workhorse code to filter
  // the data inside the buffer:
  
  typedef _vtype < _math_ele_type , _vsize > simdized_math_type ;
  typedef zimt::fir_filter < simdized_math_type > filter_type ;
  using filter_type::solve ;
  using filter_type::headroom ;
  
  // by defining arg_type, we allow code to infer what type of
  // initializer ('specs') the filter takes
  
  typedef fir_filter_specs arg_type ;
  
  // the constructor invokes the filter's constructor,
  // sets up the buffer and initializes the buffer_handling
  // component to use the whole buffer to accept incoming and
  // provide outgoing data.

  convolution_filter ( const fir_filter_specs & specs , size_t size )
  : filter_type ( specs ) ,
    buffer ( size ) ,
    buffer_extrapolator ( specs.bc , buffer )
  {
    init ( buffer , buffer ) ;
  } ;

  // operator() simply delegates to the filter's 'solve' routine,
  // which filters the data in the buffer. Note how the solve
  // overload accepting an extrapolator is used: the extrapolator
  // remains the same, so there's no point creating a new one
  // with every call.
  
  void operator() ()
  {
    solve ( buffer , buffer , buffer_extrapolator ) ;
  }
  
  // factory function to provide a filter with the same set of
  // parameters, but possibly different data types. this is used
  // for processing of 1D data, where the normal buffering mechanism
  // may be sidestepped.

  template < typename in_type ,
             typename out_type = in_type ,
             typename math_type = out_type >
  static zimt::fir_filter < in_type , out_type , math_type >
         get_raw_filter ( const fir_filter_specs & specs )
  {
    return zimt::fir_filter < in_type , out_type , math_type >
           ( specs ) ;
  }
  
} ;

/// convolve implements convolution of the input with a fixed-size
/// convolution kernel. Note that zimt does *not* follow the DSP convention
/// of using the kernel's coefficients in reverse order. The standard is to
/// calculate sum ( ck * u(n-k) ), zimt uses sum ( ck * u(n+k-h) ) where
/// 'h' is the 'headroom' of the kernel - the number of coefficients which
/// are applied 'to 'past' values, so that for for a kernel with three
/// coefficients and headroom 1, the sum at position n would be
/// c0 * u(n-1) + c1 * u(n) + c2 * u(n+1), and for a kernel with four
/// coefficients and headroom 2, you'd get
/// c0 * u(n-2) + c1 * u(n-1) + c2 * u(n) + c3 * u(n+1)
/// If you use a 'normal' kernel in zimt, you must reverse it (unless it's
/// symmetric, of course), and you must state the 'headroom': if your kernel
/// is odd-sized, this will normally be half the kernel size (integer division!),
/// producing the phase-correct result, and with an even kernel you can't get
/// the phase right, and you have to make up your mind which way the phase
/// should shift. The 'headroom' notation leaves you free to pick your choice.
/// The input, output, bcv, axis and njobs parameters are as in the routine
/// above. Here what's needed here is 'kv', a std::vector holding the filter's
/// coefficients, and the 'headroom', usually kv.size() / 2.
///
/// example: let's say you have data in 'image' and want to convolve in-place
/// with [ 1 , 2 , 1 ], using mirror boundary conditions. Then call:
///
///   zimt::convolution_filter
///    ( image ,
///      image ,
///      { zimt::MIRROR , zimt::MIRROR } ,
///      { 1 , 2 , 1 } ,
///      1 ) ;


template < std::size_t dimension ,
           typename in_value_type ,
           typename out_value_type ,
           typename math_ele_type =
                    ET < PROMOTE ( in_value_type , out_value_type ) > ,
           size_t vsize =
                  zimt::vector_traits < math_ele_type > :: size
         >
void convolve (
                 const
                 zimt::view_t
                   < dimension ,
                     in_value_type > & input ,
                 zimt::view_t
                   < dimension ,
                     out_value_type > & output ,
                 zimt::xel_t < bc_code , dimension > bcv ,
                 std::vector < zimt::xlf_type > kv ,
                 int headroom ,
                 int axis = -1 , // -1: apply along all axes
                 int njobs = default_njobs )
{
  if ( output.shape != input.shape )
    throw shape_mismatch
     ( "convolution_filter: input and output shape must match" ) ;

  size_t ksize = kv.size() ;
  
  if ( ksize < 1 )
  {
    // we can handle the no-kernel case very efficiently,
    // since we needn't apply a filter at all.

    if ( (void*) ( input.data() ) != (void*) ( output.data() ) )
    {
      // operation is not in-place, copy data to output
      output = input ;
    }
    return ;
  }
  
  zimt::xlf_type kernel [ ksize ] ;
  for ( int i = 0 ; i < ksize ; i++ )
    kernel[i] = kv[i] ;
  
  typedef typename zimt::convolution_filter
                            < zimt::simdized_type ,
                              math_ele_type ,
                              vsize
                            > filter_type ;

  if ( axis == -1 )
  {
    // user has passed -1 for 'axis', apply the same filter along all axes

    std::vector < zimt::fir_filter_specs > vspecs ;
  
    for ( int axis = 0 ; axis < dimension ; axis++ )
    {
      vspecs.push_back 
        ( zimt::fir_filter_specs
          ( bcv [ axis ] , ksize , headroom , kernel ) ) ;
    }
 
    zimt::filter
    < in_value_type , out_value_type , dimension , filter_type > 
    ( input , output , vspecs , njobs ) ;
  }
  else
  {
    // user has passed a specific axis, apply filter only to this axis

    assert ( axis >=0 && axis < dimension ) ;

    zimt::filter
    < in_value_type , out_value_type , dimension , filter_type > 
    ( input , output , axis ,
      zimt::fir_filter_specs ( bcv [ axis ] , ksize , headroom , kernel ) ,
      njobs ) ;
  }
}

END_ZIMT_SIMD_NAMESPACE

#endif // sentinel
