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

/*! \file filter.h

    \brief generic implementation of separable filtering for nD arrays

    This is a port from vspline/filter.h
    
    This body of code provides the application of separable filters,
    not the filters themselves. It's used for all types of filters
    provided in zimt:

    - convolution (see convolve.h)
    - n-pole recursive forward-backward filters (see recursive.h)
    - b-spline prefiltering (see prefilter.h)

    b-spline prefiltering is a special case of the second type where
    the 'poles' are determined by the b-spline's degree.

    All the filters operate separately along the axes of the arrays
    which they process - they are 'separable' filters. There is no code
    in zimt to use non-separable kernels.
    
    The code in this file is what I call 'wielding' code. It's function is
    to present the data in such a fashion that the code needed for the actual
    filter is a reasonably trivial 1D operation. 'Presenting' the data is
    a complex operation in zimt: the data are distributed to a set of
    worker threads, and they are restructured so that they can be processed
    by the processor's vector units. All of this is optional and transparent
    to the calling code. The 'wielding' code in this file is structurally
    similar to the code in transform.h, but here we use specific buffering
    operations which would make no sense there: for separable filtering,
    we have to preserve the adjacency of the data along the processing axis
    and present it to the filter, which is unnecessary for transformations,
    where each value can be processed in isolation.
    
    Most of the functionality in this file is in namespace detail, signalling
    that it is not meant to be called from outside. Class buffer_handling and
    the data types it uses to interface with nD memory are the exception,
    since they are meant to be inherited/used to implement specific filters.
    
    At the bottom of the file there's a free function template called
    'filter'. This is what other code will normally call.
*/

#include <vector>
#include <climits>
#include "zimt.h"

#if defined(ZIMT_FILTER_H) == defined(HWY_TARGET_TOGGLE)
  #ifdef ZIMT_FILTER_H
    #undef ZIMT_FILTER_H
  #else
    #define ZIMT_FILTER_H
  #endif

HWY_BEFORE_NAMESPACE() ;
BEGIN_ZIMT_SIMD_NAMESPACE(zimt)

// // enums for boundary conditions and their respective names as strings
// 
// typedef enum { 
//   MIRROR ,    // mirror on the bounds, so that f(-x) == f(x)
//   PERIODIC,   // periodic boundary conditions
//   REFLECT ,   // reflect, so  that f(-1) == f(0) (mirror between bounds)
//   NATURAL,    // natural boundary conditions, f(-x) + f(x) == 2 * f(0)
//   CONSTANT ,  // clamp. used for framing, with explicit prefilter scheme
//   ZEROPAD ,   // used for boundary condition, bracing
//   GUESS ,     // used instead of ZEROPAD to keep margin errors lower
//   INVALID
// } bc_code;
// 
// /// bc_name is for diagnostic output of bc codes
// 
// const std::string bc_name[] =
// {
//   "MIRROR   " ,
//   "PERIODIC ",
//   "REFLECT  " ,
//   "NATURAL  ",
//   "CONSTANT " ,
//   "ZEROPAD  " ,
//   "GUESS    "
// } ;

/// class 'bundle' holds all information needed to access a set of
/// vsize 1D subarrays of an nD array. This is the data structure
/// we use to tell the buffering and unbuffering code which data
/// we want it to put into the buffer or distribute back out. The
/// buffer itself holds the data in compact form, ready for vector
/// code to access them at maximum speed.

template < class dtype ,        // data type
           size_t vsize >       // vector width
struct bundle
{
  dtype * data ;                // data base address
  const std::ptrdiff_t * idx ;  // pointer to gather/scatter indexes
  std::ptrdiff_t stride ;       // stride in units of dtype
  unsigned long z ;             // number of repetitions

  bundle ( dtype * _data ,
           const std::ptrdiff_t * _idx ,
           std::ptrdiff_t _stride ,
           unsigned long _z )
  : data ( _data ) ,
    idx ( _idx ) ,
    stride ( _stride ) ,
    z (_z )
  { } ;
} ;

/// move bundle data to compact memory

template < class stype ,        // source data type
           class ttype ,        // target data type
           size_t vsize >       // vector width
void move ( const bundle < stype , vsize > & src ,
            ttype * trg ,
            std::ptrdiff_t trg_stride
          )
{
  auto z = src.z ;
  auto ps = src.data ;
  
  while ( z-- ) // repeat z times:
  {
    for ( size_t i = 0 ; i < vsize ; i++ )
    {
      // load from source, store to target, using ith index
      trg [ i ] = ttype ( ps [ src.idx [ i ] ] ) ;
    }
    ps += src.stride ;  // apply stride to source
    trg += trg_stride ; // and target
  }
}

// nearly the same, but takes ni as runtime parameter, effectively
// limiting the transfer to the first ni offsets in the bundle.
// This will only be used rarely, so performance is less of an issue.

template < class stype ,        // source data type
           class ttype ,        // target data type
           size_t vsize >       // vector width
void move ( const bundle < stype , vsize > & src ,
            ttype * trg ,
            std::ptrdiff_t trg_stride ,
            int ni )
{
  auto z = src.z ;
  auto ps = src.data ;
  
  while ( z-- ) // repeat z times:
  {
    for ( int i = 0 ; i < ni ; i++ )
    {
      // load from source, store to target, using ith index
      trg [ i ] = ttype ( ps [ src.idx [ i ] ] ) ;
    }
    trg += trg_stride ;
    ps += src.stride ; // apply stride to source
  }
}

/// move data from compact memory to bundle

template < class stype ,        // source data type
           class ttype ,        // target data type
           size_t vsize >       // vector width
void move ( const stype * src ,
            std::ptrdiff_t src_stride ,
            const bundle < ttype , vsize > & trg )
{
  auto z = trg.z ;
  auto pt = trg.data ;
  
  while ( z-- ) // repeat z times:
  {
    for ( size_t i = 0 ; i < vsize ; i++ )
    {
      // load from source, store to target, using ith index
      pt [ trg.idx [ i ] ] = ttype ( src [ i ] ) ;
    }
    src += src_stride ;
    pt += trg.stride ; // apply stride to target
  }
}

// nearly the same, but takes ni as runtime parameter, effectively
// limiting the transfer to the first ni offsets in the bundle.

template < class stype ,        // source data type
           class ttype ,        // target data type
           size_t vsize >       // vector width
void move ( const stype * src ,
            std::ptrdiff_t src_stride ,
            const bundle < ttype , vsize > & trg ,
            int ni )
{
  auto z = trg.z ;
  auto pt = trg.data ;
  
  while ( z-- ) // repeat z times:
  {
    for ( int i = 0 ; i < ni ; i++ )
    {
      // load from source, store to target, using ith index
      pt [ trg.idx [ i ] ] = ttype ( src [ i ] ) ;
    }
    src += src_stride ;
    pt += trg.stride ; // apply stride to target
  }
}

/// buffer_handling provides services needed for interfacing
/// with a buffer of simdized/goading data. The init() routine
/// receives two views: one to a buffer accepting incoming data,
/// and one to a buffer providing results. Currently, all filters
/// used in zimt operate in-place, but the two-argument form
/// leaves room to manoevre.
/// get() and put() receive 'bundle' arguments which are used
/// to transfer incoming data to the view defined in in_window,
/// and to transfer result data from the view defined in
/// out_window back to target memory.

template < template < typename , size_t > class _vtype ,
           typename _dtype ,
           size_t _vsize >
class buffer_handling
{
protected:

  enum { vsize = _vsize } ;
  typedef _dtype dtype ;
  typedef _vtype < dtype , vsize > vtype ;
  
  zimt::view_t < 1 ,  vtype > in_window ;
  zimt::view_t < 1 ,  vtype > out_window ;
  
  void init ( zimt::view_t < 1 ,  vtype > & _in_window ,
              zimt::view_t < 1 ,  vtype > & _out_window )
  {
    in_window = _in_window ;
    out_window = _out_window ;
  }
  
  // get and put receive 'bundle' objects, currently the code
  // uses a template for the arguments but we might fix it to
  // bundles only
  
  // note the use of 'offset' which is needed for situations
  // when input/output consists of several arrrays
  
  // why not use ni all the time? I suspect
  // fixed-width moves are faster. TODO test

  // KFJ 2018-02-20 added parameter for the buffer's stride from
  // one datum to the next, expressed in untits of 'dtype'. This
  // is needed if the buffer contains SimdArrays which hold
  // padding and therefore are larger than vsize dtype. This does
  // only occur for certain vsize values, not for the default as
  // per common.h, so it went unnoticed for some time.
  
  static const std::ptrdiff_t bf_stride = sizeof(vtype) / sizeof(dtype) ;

public:
  
  /// fetch data from 'source' into the buffer 'in_window'
  
  template < class tractor >
  void get ( const tractor & src ,
             std::ptrdiff_t offset = 0 ,
             int ni = vsize ) const
  {
    if ( ni == vsize ) // fixed-width move
      
      move ( src ,
             (dtype*) ( in_window.data() + offset ) ,
             bf_stride
           ) ;
             
    else               // reduced width move
      
      move ( src ,
             (dtype*) ( in_window.data() + offset ) ,
             bf_stride ,
             ni 
           ) ;
  }
  
  /// deposit result data from 'out_window' into target memory

  template < class tractor >
  void put ( const tractor & trg ,
             std::ptrdiff_t offset = 0 ,
             int ni = vsize ) const
  {
    if ( ni == vsize )
      move ( (const dtype *) ( out_window.data() + offset ) ,
             bf_stride ,
             trg 
           ) ;
    else
      move ( (const dtype *) ( out_window.data() + offset ) ,
             bf_stride ,
             trg ,
             ni
           ) ;
  }
  
} ;

namespace detail
{
/// 'present' feeds 'raw' data to a filter and returns the filtered
/// data. In order to perform this task with maximum efficiency,
/// the actual code is quite involved.
///
/// we have two variants of the routine, one for 'stacks' of several
/// arrays (vpresent) and one for single arrays (present).
///
/// The routine used in zimt is 'present', 'vpresent' is for special
/// cases. present splits the data into 'bundles' of 1D subarrays
/// collinear to the processing axis. These bundles are fed to the
/// 'handler', which copies them into a buffer, performs the actual
/// filtering, and then writes them back to target memory.
///
/// Using 'vpresent', incoming data are taken as std::vectors of
/// source_view_type. The incoming arrays have to have the same extent
/// in every dimension *except* the processing axis. While the actual
/// process of extracting parts of the data for processing is slightly
/// more involved, it is analogous to first concatenating all source
/// arrays into a single array, stacking along the processing axis.
/// The combined array is then split up into 1D subarrays collinear
/// to the processing axis, and sets of these subarrays are passed to
/// the handler by calling it's 'get' method. The set of 1D subarrays
//// is coded as a 'bundle', which describes such a set by a combination
/// of base address and a set of gather/scatter indexes.
///
/// Once the data have been accepted by the handler, the handler's
/// operator() is called, which results in the handler filtering the
/// data (or whatever else it might do). Next, the processed data are
/// taken back from the handler by calling it's 'put' routine. The put
/// routine also receives a 'bundle' parameter, resulting in the
/// processed data being distributed back into a multidimensional
/// array (or a set of them, like the input).
///
/// This mechanism sounds complicated, but buffering the data for
/// processing (which oftentimes has to look at the data several times)
/// is usually more efficient than operating on the data in their
/// in-array locations, which are often widely distributed, making
/// the memory access slow. On top of the memory efficiency gain,
/// there is another aspect: by choosing the bundle size wisely,
/// the buffered data can be processed by vector code. Even if the
/// data aren't explicit SIMD vectors (which is an option), the
/// simple fact that they 'fit' allows the optimizer to autovectorize
/// the code, a technique which I call 'goading': You present the
/// data in vector-friendly guise and thereby lure the optimizer to
/// do the right thing. Another aspect of buffering is that the
/// buffer can use a specific data type best suited to the arithmetic
/// operation at hand which may be different from the source and target
/// data. This is especially useful if incoming data are of an integral
/// type: operating directly on integers would spoil the data, but if
/// the buffer is set up to contain a real type, the data are lifted
/// to it on arrival in the buffer and processed with float maths.
/// A drawback to this method of dealing with integral data is the fact
/// that, when filtering nD data along several axes, intermediate
/// results are stored back to the integral type after processing along
/// each axis, accruing quantization errors with each pass. If this is
/// an issue - like, with high-dimensional data or insufficient dynamic
/// range, please consider moving to a real data type before filtering.
///
/// Note that this code operates on arrays of fundamentals. The code
/// calling this routine will have element-expanded data which weren't
/// fundamentals in the first place. This expansion helps automatic
/// vectorization, and for explicit vectorization with Vc it is even
/// necessary.
///
/// Also note that this routine operates in single-threaded code:
/// It's invoked via zimt::multithread, and each worker thread will
/// perform it's own call to 'present'. This is why the first argument
/// is a range (containing the range of the partitioning assigned to
/// the current worker thread) and why the other arguments come in as
/// pointers, where I'd usually pass by reference.

// we start out with 'present', which takes plain MultiArrayViews
// for input and output. This is the simpler case. 'vpresent', which
// follows, is structurally similar but has to deal with the extra
// complication of processing 'stacks' instead of single arrays.

template < typename source_view_type ,
           typename target_view_type ,
           typename stripe_handler_type >
void present ( zimt::atomic < std::ptrdiff_t > * p_tickets ,
               const source_view_type * p_source ,
               target_view_type * p_target ,
               const typename stripe_handler_type::arg_type * p_args ,
               int axis )
{
  enum { dimension = source_view_type::dimension } ;
  enum { vsize = stripe_handler_type::vsize } ;
  
  // get references to the source and target views

  const source_view_type & source ( *p_source ) ;
  target_view_type & target ( *p_target ) ;
  
  // get the total length of the axis we want to process
  
  std::ptrdiff_t count = source.shape [ axis ] ;

  // set up the 'stripe handler' which holds the buffer(s)
  // and calls the filter. It's important that this does not
  // happen until now when we've landed in the code executed
  // by the worker threads, since the stripe_handler holds
  // state which must not be shared between threads.

  stripe_handler_type handler ( *p_args , count ) ;
  
  // get the data types we're dealing with. These are fundamentals,
  // since the arrays have been element-expanded for processing.
  
  typedef typename source_view_type::value_type stype ;
  typedef typename target_view_type::value_type ttype ;
  
  // take a slice from the source array orthogonal to the processing
  // axis. That's where the starting points of the 1D subarrays are. Then
  // obtain a MultiCoordinateIterator over the indexes in the slice.
  // We use nD indexes rather than directly using offsets because these
  // indexes will be reused when the filtered data are written to the
  // destination memory, where strides may be different. So the offsets
  // for the way 'in' and the way 'out' are calculated from the same
  // indexes just before they are used, by multiplying with the
  // appropriate strides and summing up.
  
  auto sample_slice = source.slice ( axis , 0 ) ;

  zimt::mci_t < dimension - 1 > sliter ( sample_slice.shape ) ;

  // shape_type can hold an nD index into the slice, just what
  // sliter refers to.
  
  typedef zimt::xel_t < std::ptrdiff_t , dimension - 1 > shape_type ;  
  
  // set of indexes for one run. Note the initialization with
  // the first index, guarding against 'narrow stripes'.
  
  zimt::xel_t < shape_type , vsize > indexes ( sliter[0] ) ;
  
  // set of offsets into the source slice which will be used for
  // gather/scatter. These will be equivalent to the indexes above,
  // 'condensed' by applying the stride and summing up.

  zimt::xel_t < std::ptrdiff_t , vsize > offsets ;
  
  // now we keep fetching batches of vsize 1D subarrays until the
  // atomic which p_tickets points to yields no more indexes
  
  std::ptrdiff_t lo , hi ;
  std::ptrdiff_t batch_size = vsize ;
  std::ptrdiff_t total_size = sample_slice.size() ;
  
  while ( zimt::fetch_range_ascending
            ( *p_tickets , batch_size , total_size , lo , hi ) )
  {
    std::ptrdiff_t n_fetch = hi - lo ;

    for ( std::ptrdiff_t i = 0 , f = lo ; i < n_fetch ; ++i , ++f )
    {
      indexes [ i ] = sliter [ f ] ;
    }

    // process the source array
    
    auto stride = source.strides [ axis ] ;
    auto size = source.shape [ axis ] ;
    
    auto source_slice = source.slice ( axis , 0 ) ;
    auto source_base_adress = source_slice.data() ;

    // obtain a set of offsets from the set of indexes by 'condensing'
    // the nD index into an offset - by applying the strides and summing up
    
    for ( int e = 0 ; e < vsize ; e++ )   
    {
      offsets[e] = ( source_slice.strides * indexes[e] ).sum() ;
    }
    
    // form a 'bundle' to pass the data to the handler
    
    bundle < stype , vsize > bi ( source_base_adress ,
                                  offsets.data() ,
                                  stride ,
                                  size ) ;
      
    // now use the bundle to move the data to the handler's buffer

    handler.get ( bi , 0 , n_fetch ) ;
    
    // now call the handler's operator(). This performs the actual
    // filtering of the data
    
    handler() ;
    
    // now empty out the buffer to the target array, using pretty much
    // the same set of operations as was used for fetching the data
    // from source. Note how the offsets are recalculated, now using the
    // target slice's strides.
    
    stride = target.strides [ axis ] ;
    size = target.shape [ axis ] ;
    
    auto target_slice = target.slice ( axis , 0 ) ;
    auto target_base_adress = target_slice.data() ;

    for ( int e = 0 ; e < vsize ; e++ )          
      offsets[e] = ( target_slice.strides * indexes[e] ).sum() ;
    
    bundle < ttype , vsize > bo ( target_base_adress ,
                                  offsets.data() ,
                                  stride ,
                                  size ) ;
      
    handler.put ( bo , 0 , n_fetch ) ;
  }
}

/// vpresent is a variant of 'present' processing 'stacks' of arrays.
/// See 'present' for discussion. This variant of 'present' will rarely
/// be used. Having it does no harm but if you study the code, you may
/// safely ignore it unless you are actually using single-axis filtering
/// of stacks of arrays. the code is structurally similar to 'present',
/// with the extra complication of processing stacks instead of single
/// arrays.

template < typename source_view_type ,
           typename target_view_type ,
           typename stripe_handler_type >
void vpresent ( zimt::atomic < std::ptrdiff_t > * p_tickets ,
                const std::vector<source_view_type> * p_source ,
                std::vector<target_view_type> * p_target ,
                const typename stripe_handler_type::arg_type * p_args ,
                int axis )
{
  enum { dimension = source_view_type::dimension } ;
  enum { vsize = stripe_handler_type::vsize } ;
  
  // get references to the std::vectors holding source and target views

  const std::vector<source_view_type> & source ( *p_source ) ;
  std::vector<target_view_type> & target ( *p_target ) ;
  
  // get the total length of the axis we want to process
  
  std::ptrdiff_t count = 0 ;
  for ( auto & e : source )
    count += e.shape [ axis ] ;

  // set up the 'stripe handler' which holds the buffer(s)
  // and calls the filter. It's important that this does not
  // happen until now when we've landed in the code executed
  // by the worker threads, since the stripe_handler holds
  // state which must not be shared between threads.

  stripe_handler_type handler ( *p_args , count ) ;
  
  // get the data types we're dealing with. These are fundamentals,
  // since the arrays have been element-expanded for processing.
  
  typedef typename source_view_type::value_type stype ;
  typedef typename target_view_type::value_type ttype ;
  
  // take a slice from the first source array orthogonal to the processing
  // axis. That's where the starting points of the 1D subarrays are. Then
  // obtain a MultiCoordinateIterator over the indexes in the slice
  
  auto sample_slice = source[0].bindAt ( axis , 0 ) ;

  zimt::mci_t < dimension - 1 > sliter ( sample_slice.shape ) ;

  // shape_type can hold an nD index into the slice, just what
  // sliter refers to.
  
  typedef zimt::xel_t < std::ptrdiff_t , dimension - 1 > shape_type ;  
  
  // set of indexes for one run. Note the initialization with
  // the first index, guarding against 'narrow stripes'.
  
  zimt::xel_t < shape_type , vsize > indexes { *sliter } ;
  
  // set of offsets into the source slice which will be used for
  // gather/scatter. These will be equivalent to the indexes above,
  // 'condensed' by applying the stride and summing up.

  zimt::xel_t < std::ptrdiff_t , vsize > offsets ;
  
  // now we keep fetching batches of vsize 1D subarrays until the
  // atomic which p_tickets points to yields no more indexes
  
  std::ptrdiff_t lo , hi ;
  std::ptrdiff_t batch_size = vsize ;
  std::ptrdiff_t total_size = sample_slice.size() ;
  
  while ( zimt::fetch_range_ascending
            ( *p_tickets , batch_size , total_size , lo , hi ) )
  {
    std::ptrdiff_t n_fetch = hi - lo ;

    for ( std::ptrdiff_t i = 0 , f = lo ; i < n_fetch ; ++i , ++f )
    {
      indexes [ i ] = sliter [ f ] ;
    }

    // iterate over the input arrays, loading data into the buffer
    // from all arrays in turn, using the same set of indexes.
    // 'progress' holds the part of 'count' that has been transferred
    // already.
  
    std::ptrdiff_t progress = 0 ;
  
    // now iterate over the source arrays. While the set of nD indexes
    // used is the same for each stack member, the offsets may be different,
    // as they are calculated using specific strides for each stack member.
    
    for ( auto & input : source )
    {
      auto source_stride = input.strides [ axis ] ;
      auto part_size = input.shape [ axis ] ;
      auto slice = input.bindAt ( axis , 0 ) ;
      auto source_base_adress = slice.data() ;

      // obtain a set of offsets from the set of indexes by 'condensing'
      // the nD index into an offset - by applying the strides and summing up
      
      for ( int e = 0 ; e < vsize ; e++ )   
      {
        offsets[e] = sum ( slice.strides * indexes[e] ) ;
      }
      
      // form a 'bundle' to pass the data to the handler
      
      bundle < stype , vsize > bi ( source_base_adress ,
                                    offsets.data() ,
                                    source_stride ,
                                    part_size ) ;
        
      // now use the bundle to fill part_size entries in the handler's
      // buffer, starting at 'progress'. 'progress' records how many
      // sets of values have already been pushed into the buffer
      // then carry on with the next input array, if any

      handler.get ( bi , progress , n_fetch ) ;
      progress += part_size ;
    }
    
    // data from all stacks have been transferred to the buffer.
    // now call the handler's operator(). This performs the actual
    // filtering of the data
    
    handler() ;
    
    // now empty out the buffer to the std::vector of target arrays,
    // using pretty much the same set of operations as was used for
    // fetching the data from source.
    
    progress = 0 ;
    
    for ( auto & output : target )
    {
      auto target_stride = output.strides [ axis ] ;
      auto part_size = output.shape [ axis ] ;
      auto slice = output.bindAt ( axis , 0 ) ;
      auto target_base_adress = slice.data() ;

      for ( int e = 0 ; e < vsize ; e++ )          
        offsets[e] = sum ( slice.strides * indexes[e] ) ;
      
      bundle < ttype , vsize > bo ( target_base_adress ,
                                    offsets.data() ,
                                    target_stride ,
                                    part_size ) ;
        
      handler.put ( bo , progress , n_fetch ) ;
      progress += part_size ;
    }
  }
}

/// struct separable_filter is the central object used for 'wielding'
/// filters. The filters themselves are defined as 1D operations, which
/// is sufficient for a separable filter: the 1D operation is applied
/// to each axis in turn. If the *data* themselves are 1D, this is
/// inefficient if the run of data is very long: we'd end up with a
/// single thread processing the data without vectorization. So for this
/// special case, we use a bit of trickery: long runs of 1D data are
/// folded up, processed as 2D (with multithreading and vectorization)
/// and the result of this operation, which isn't correct everywhere,
/// is 'mended' where it is wrong. If the data are nD, we process them
/// by buffering chunks collinear to the processing axis and applying
/// the 1D filter to these chunks. 'Chunks' isn't quite the right word
/// to use here - what we're buffering are 'bundles' of 1D subarrays,
/// where a bundle holds as many 1D subarrays as a SIMD vector is wide.
/// this makes it possible to process the buffered data with vectorized
/// code. While most of the time the buffering will simply copy data into
/// and out of the buffer, we use a distinct data type for the buffer
/// which makes sure that arithmetic can be performed in floating point
/// and with sufficient precision to do the data justice. With this
/// provision we can safely process arrays of integral type. Such data
/// are 'promoted' to this type when they are buffered and converted to
/// the result type afterwards. Of course there will be quantization
/// errors if the data are converted to an integral result type; it's
/// best to use a real result type.
/// The type for arithmetic operations inside the filter is fixed via
/// stripe_handler_type, which takes a template argument '_math_ele_type'.
/// This way, the arithmetic type is distributed consistently.
/// Also note that an integral target type will receive the data via a
/// simple type conversion and not with saturation arithmetics. If this
/// is an issue, filter to a real-typed target and process separately. 
/// A good way of using integral data is to have integral input
/// and real-typed output. Promoting the integral data to a real type
/// preserves them precisely, and the 'exact' result is then stored in
/// floating point. With such a scheme, raw data (like image data,
/// which are often 8 or 16 bit integers) can be 'sucked in' without
/// need for previous conversion, producing filtered data in, say, float
/// for further processing.

template < typename input_array_type ,
           typename output_array_type ,
           typename stripe_handler_type >
struct separable_filter
{
  enum { dimension = input_array_type::dimension } ;
  static_assert ( dimension == output_array_type::dimension ,
                  "separable_filter: input and output array type must have the same dimension" ) ;
                  
  typedef typename input_array_type::value_type in_value_type ;
  typedef typename output_array_type::value_type out_value_type ;
  
  enum { channels = zimt::get_ele_t < in_value_type > :: size } ;
  static_assert ( channels
                  == zimt::get_ele_t < out_value_type > :: size ,
          "separable_filter: input and output data type must have the same number of channels" ) ;
          
  typedef typename zimt::get_ele_t < in_value_type >
                   :: type in_ele_type ;
                   
  typedef typename zimt::get_ele_t < out_value_type >
                   :: type out_ele_type ;

  typedef std::integral_constant < bool , dimension == 1 > is_1d_type ;
  typedef std::integral_constant < bool , channels == 1 > is_1_channel_type ;

  /// this is the standard entry point to the separable filter code
  /// for processing *all* axes of an array. first we use a dispatch
  /// to separate processing of 1D data from processing of nD data.

  template < class filter_args > // may be single argument or a std::vector
  void operator() ( const input_array_type & input ,
                    output_array_type & output ,
                    const filter_args & handler_args ,
                    int njobs = zimt::default_njobs ) const
  {
    // we use a dispatch depending on whether data are 1D or nD arrays

    _on_dimension ( is_1d_type() ,
                    input , output , handler_args , njobs ) ;
  }
  
  // _on_dimension differentiates between 1D and nD data. We don't
  // look at the arguments - they are simply forwarded to either
  // _process_1d or _process_nd.

  template < typename ... types >
  void _on_dimension ( std::true_type ,  // 1D data
                       types ... args ) const
  {
    // data are 1D. unpack the variadic content and call
    // the specialized method
    
    _process_1d ( args ... ) ;
  }
  
  template < typename ... types >
  void _on_dimension ( std::false_type ,  // nD data
                       types ... args ) const
  {
    // data are nD. unpack the variadic content and call
    // the code for nD processing.

    _process_nd ( args ... ) ;
  }
  
  /// specialized processing of 1D input/output.
  /// We have established that the data are 1D.
  /// we have received a std::vector of handler arguments.
  /// It has to contain precisely one element which we unpack
  /// and use to call the overload below.
  
  template < typename in_vt , typename out_vt >
  void _process_1d ( const in_vt & input ,
                     out_vt & output ,
                     const std::vector
                           < typename stripe_handler_type::arg_type >
                           & handler_args ,
                     int njobs ) const
  {
    assert ( handler_args.size() == 1 ) ;
    _process_1d ( input , output , handler_args[0] , njobs ) ;
  }
  
  /// specialized processing of 1D input/output.
  /// We have established that the data are 1D and we have
  /// a single handler argument.
  /// This routine may come as a surprise and it's quite long
  /// and complex. The strategy is this:
  /// - if the data are 'quite short', simply run a 1D filter
  ///   directly on the data, without any multithreading or
  ///   vectorization. If the user has specified 'zero tolerance',
  ///   do the same.
  /// - otherwise employ 'fake 2D processing': pretend the
  ///   data are 2D, filter them with 2D code (which offers
  ///   multithreading and vectorization) and then 'mend'
  ///   the result, which is wrong in parts due to the
  ///   inappropriate processing.
  /// expect 'fake 2D processing' to kick in for buffer sizes
  /// somewhere in the low thousands, to give you a rough idea.
  /// All data paths in this routine make sure that the maths
  /// are done in math_type, there won't be storing of
  /// intermediate values to a lesser type. If the user
  /// has specified 'zero tolerance' and the output type is not
  /// the same as math_type, we have a worst-case scenario where
  /// the entire length of data is buffered in math_type and the
  /// operation is single-threaded and unvectorized, but this
  /// should rarely happen and requires the user to explicitly
  /// override the defaults. If the data are too short for fake
  /// 2D processing, the operation will also fail to multithread
  /// or vectorize.
  
  // TODO: establish the cost of powering up the multithreaded data
  // processing to set a lower limit for data sizes which should be
  // processed with several threads: the overhead for small data sets
  // might make multithreading futile.
  
  // call receiving an axis is routed to overload below - this
  // overload here is needed for symmetry with _process_nd
  // TODO: needing this seems slightly dodgy...
  
  template < typename in_vt , typename out_vt >
  void _process_1d ( const in_vt & input ,
                     out_vt & output ,
                     int axis ,
                     const typename stripe_handler_type::arg_type
                       & handler_args ,
                     int njobs ) const
  {
    _process_1d ( input , output , handler_args , njobs ) ;
  }
  
  template < typename in_vt , typename out_vt >
  void _process_1d ( const in_vt & input ,
                     out_vt & output ,
                     const typename stripe_handler_type::arg_type
                       & handler_args ,
                     int njobs ) const
  {
    typedef typename in_vt::value_type in_value_type ;
    typedef typename out_vt::value_type out_value_type ;

    // we'll need to access the 'raw' filter. To specify it's type in
    // agreement with 'stripe_handler_type', we glean math_ele_type
    // from there and construct math_type from it.

    typedef typename stripe_handler_type::math_ele_type math_ele_type ;
    typedef canonical_type < math_ele_type , channels > math_type ;
      
    // obtain a raw filter capable of processing math_type
    
    auto raw_filter = stripe_handler_type::template
                      get_raw_filter < math_type > ( handler_args ) ;

    // right now, we only need the filter's support width, but we
    // may use the filter further down.
                      
    const int bands = channels ;
    int runup = raw_filter.get_support_width() ;

    // if we can multithread, start out with as many lanes
    // as the desired number of threads

    int lanes = njobs ;
    enum { vsize = stripe_handler_type::vsize } ;

    // the number of lanes is multiplied by the
    // number of elements a vector-friendly type can handle

    lanes *= vsize ;

    // the absolute minimum to successfully run the fake 2D filter is this:
    // TODO we might rise the threshold, min_length, here
    
    int min_length = 4 * runup * lanes + 2 * runup ;
    
    // runup == INT_MAX signals that fake 2D processing is inappropriate.
    // if input is too short to bother with fake 2D, just single-lane it
    
    if ( runup == INT_MAX || input.shape[0] < min_length )
    {
      lanes = 1 ;
    }
    else
    {
      // input is larger than the absolute minimum, maybe we can even increase
      // the number of lanes some more? we'd like to do this if the input is
      // very large, since we use buffering and don't want the buffers to become
      // overly large. But the smaller the run along the split x axis, the more
      // incorrect margin values we have to mend, so we need a compromise.
      // assume a 'good' length for input: some length where further splitting
      // would not be wanted anymore. TODO: do some testing, find a good value
      
      int good_length = 64 * runup * lanes + 2 * runup ;
      
      int split = 1 ;
      
      // suppose we split input.shape[0] in ( 2 * split ) parts, is it still larger
      // than this 'good' length? If not, leave split factor as it is.
      
      while ( input.shape[0] / ( 2 * split ) >= good_length )
      {  
        // if yes, double split factor, try again
        split *= 2 ;
      }
      
      lanes *= split ; // increase number of lanes by additional split
    }
    
    // if there's only one lane we fall back to single-threaded
    // operation, using a 'raw' filter directly processing the
    // input - either producing the output straight away or,
    // intermediately, it's representation in math_type.

    if ( lanes == 1 )
    {
      // we look at the data first: if out_value_type is the same type
      // as math_type, we can use the raw filter directly on input and
      // output. This is also possible if the filter is single-pass,
      // because a single-pass filter does not need to store intermediate
      // results - so convolution is okay, but b-spline prefiltering
      // is not.
      if (    std::is_same < out_value_type , math_type > :: value
           || stripe_handler_type::is_single_pass )
      {
        auto raw_filter = stripe_handler_type::template
          get_raw_filter < in_value_type ,
                           out_value_type ,
                           math_type > ( handler_args ) ;

        raw_filter.solve ( input , output ) ;
      }
      else
      {
        // we can't use the easy option above. So we'll have to create
        // a buffer of math_type, use that as target, run the filter
        // and copy the result to output. This is potentially expensive:
        // the worst case is that we have to create a buffer which is
        // larger than the whole input signal (if math_type's size is
        // larger than in-value_type's) - and on top, the operation is
        // single-threaded and unvectorized. This should rarely happen
        // for long signals. Mathematically, we're definitely on the
        // safe side, provided the user hasn't chosen an unsuitable
        // math_type.
        
        zimt::array_t < 1 , math_type > buffer ( input.shape ) ;
          
        auto raw_filter = stripe_handler_type::template
          get_raw_filter < in_value_type ,
                           math_type ,
                           math_type > ( handler_args ) ;

        raw_filter.solve ( input , buffer ) ;
        
        // auto trg = output.begin() ;
        auto * src = &(buffer[0]) ; // output.begin() ;
        auto * trg = &(output[0]) ; // output.begin() ;
        // for ( auto const & src : buffer )
        for ( std::size_t i = 0 ; i < input.shape[0] ; i++ )
        {
          *trg = out_value_type ( *src ) ;
          ++src ; ++trg ;
        }
      }      
      return ; // return directly. we're done
    }
    
    // the input qualifies for fake 2D processing.
    // we want as many chunks as we have lanes. There may be some data left
    // beyond the chunks (tail_size of value_type)
    
    int core_size = input.shape[0] ;
    int chunk_size = core_size / lanes ;
    core_size = lanes * chunk_size ;
    int tail_size = input.shape[0] - core_size ;
    
    // just doublecheck

    assert ( core_size + tail_size == input.shape[0] ) ;
    
    // now here's the strategy: we treat the data as if they were 2D. This will
    // introduce errors along the 'vertical' margins, since there the 2D treatment
    // will start with some boundary condition along the x axis instead of looking
    // at the neighbouring line where the actual continuation is.
    
    // first we deal with the very beginning and end of the signal. This requires
    // special treatment, because here we want the boundary conditions to take
    // effect. So we copy the beginning and end of the signal to a buffer, being
    // generous with how many data we pick. The resulting buffer will have an
    // unusable part in the middle, where tail follows head, but since we've made
    // sure that this location is surrounded by enough 'runup' data, the effect
    // will only be detectable at +/- runup from the point where tail follows head.
    // The beginning of head and the end of tail are at the beginning and end
    // of the buffer, though, so that applying the boundary condition will
    // have the desired effect. What we'll actually use of the buffer is not
    // the central bit with the effects of the clash of head and tail, but
    // only the bits at the ends which aren't affected because they are far enough
    // away. Another way of looking at this operation is that we 'cut out' a large
    // central section of the data and process the remainder, ignoring the cut-out
    // part. Then we only use that part of the result which is 'far enough' away
    // from the cut to be unaffected by it.
    
    // note how this code fixes a bug in my initial implementation, which produced
    // erroneous results with periodic splines, because the boundary condition
    // was not properly honoured.
    
    // calculate the sizes of the parts of the signal we'll put into the buffer
    int front = 2 * runup ;
    int back = tail_size + 2 * runup ;
    int total = front + back ;
    
    // create the buffer and copy the beginning and end of the signal into it.
    // Note how the data are converted to math_type to do the filtering
    
    zimt::array_t < 1 , math_type > head_and_tail ( total ) ;
    
    // auto target_it = head_and_tail.begin() ;
    // auto source_it = input.begin() ;
    auto * target_it = &(head_and_tail[0]) ;
    auto * source_it = &(input[0]) ;
    for ( int i = 0 ; i < front ; i++ )
    {
      *target_it = math_type ( *source_it ) ;
      ++target_it ;
      ++source_it ;
    }
    // source_it = input.end() - back ;
    source_it = &(input[input.shape[0]]) - back ;
    for ( int i = 0 ; i < back ; i++ )
    {
      *target_it = math_type ( *source_it ) ;
      ++target_it ;
      ++source_it ;
    }

    // this buffer is submitted to the 'raw' filter. After the call, the buffer
    // has usable data for the very beginning and end of the signal.

    raw_filter.solve ( head_and_tail , head_and_tail ) ;

    // set up two MultiArrayViews corresponding to the portions of the data
    // we copied into the buffer. The first bit of 'head' and the last bit
    // of 'tail' hold valid data and will be used further down.

    // zimt::view_t < 1 , math_type > head
    //   ( zimt::xel_t<std::size_t,1> ( front ) , head_and_tail.data() ) ;
    // 
    // zimt::view_t < 1 , math_type > tail
    //   ( zimt::xel_t<std::size_t,1> ( back ) , head_and_tail.data() + front ) ;
    
    auto head = head_and_tail.subarray ( 0 , front ) ;
    auto tail = head_and_tail.subarray ( front , front + back ) ;
    
    // head now has runup correct values at the beginning, succeeded by runup
    // invalid values, and tail has tail_size + runup correct values at the end,
    // preceded by runup values which aren't usable.

    // now we create a fake 2D view to the margin of the data. Note how we let
    // the view begin 2 * runup before the end of the first line, capturing the
    // 'wraparound' right in the middle of the view.

    // The fake 2D views hold enough runup on either side of the usable
    // data, so we use boundary conditions which are fast to compute instead of
    // futilely using the boundary conditions pertaining to the original data,
    // which would only have an effect on the runup data which do not end up
    // in the final result at all. We still end up wasting a few cycles, because
    // the filter itself will surround the data with some extrapolated values
    // (as many as is deemed appropriate for the filter's support), but at least
    // the extrapolation won't be coputationally expensive. Getting rid of these
    // extra computations is probably more expensive than accepting this small
    // amound of wasted CPU time.

    typename stripe_handler_type::arg_type
             handler_args_with_bc_guess ( handler_args ) ;

    handler_args_with_bc_guess.bc = zimt::GUESS ;

    // KFJ 2018-02-11 both here, and a bit further down, where 'margin_target'
    // is set up, I had forgotten to multiply the offset which is added to
    // *.data() with the appropriate stride, resulting in memory errors where
    // the stride wasn't 1. since this rarely happens, I did not notice it
    // until now.
    
    zimt::view_t < 2 , in_value_type >
    
      fake_2d_margin ( input.data() +  input.strides[0]
                                       * ( chunk_size - 2 * runup ) ,
                       
                       zimt::xel_t<std::size_t,2> { input.strides[0] ,
                                                    input.strides[0]
                                                    * chunk_size } ,
                       
                       zimt::xel_t<std::size_t,2> { 4 * runup ,
                                                    lanes - 1 }
                     ) ;
  
    // again we create a buffer and filter into the buffer

    zimt::array_t < 2 , out_value_type >
      margin_buffer ( fake_2d_margin.shape ) ;

    separable_filter < zimt::view_t < 2 , in_value_type > ,
                       zimt::view_t < 2 , out_value_type > ,
                       stripe_handler_type >()
      ( fake_2d_margin , margin_buffer , 0 , handler_args_with_bc_guess , njobs ) ;
    
    // now we have filtered data for the margins in margin_buffer,
    // of which the central half is usable, the remainder being runup data
    // which we'll ignore. Here's a view to the central half:
    
    zimt::view_t < 2 , out_value_type > margin
    = margin_buffer.subarray ( zimt::xel_t<std::size_t,2> { runup , 0 } ,
                               zimt::xel_t<std::size_t,2> { 3 * runup , lanes - 1 } ) ;
    
    // we create a view to the target array's margin which we intend
    // to overwrite, but the data will only be copied in from margin
    // after the treatment of the core.

    zimt::view_t < 2 , out_value_type >
    
      margin_target ( output.data() + output.strides[0]
                                      * ( chunk_size - runup ) ,
                      
                      zimt::xel_t<std::size_t,2> { output.strides[0] ,
                                                   output.strides[0]
                                                   * chunk_size } ,
                      
                      zimt::xel_t<std::size_t,2> { 2 * runup ,
                                                   lanes - 1 }
                    ) ;
                      
    // next we 'fake' a 2D array from input and filter it to output, this may
    // be an in-place operation, since we've extracted all margin information
    // earlier and deposited what we need in buffers.
    
    zimt::view_t < 2 , in_value_type >
      fake_2d_source ( input.data() ,
                       zimt::xel_t<std::size_t,2> { input.strides[0] ,
                                                    input.strides[0]
                                                    * chunk_size } ,
                       zimt::xel_t<std::size_t,2> { chunk_size , lanes }
                      ) ;

    zimt::view_t < 2 , out_value_type >
      fake_2d_target ( output.data() ,
                       zimt::xel_t<std::size_t,2> { output.strides[0] ,
                                                    output.strides[0]
                                                    * chunk_size } ,
                       zimt::xel_t<std::size_t,2> { chunk_size , lanes } ) ;

    // now we filter the fake 2D source to the fake 2D target

    separable_filter < zimt::view_t < 2 , in_value_type > ,
                       zimt::view_t < 2 , out_value_type > ,
                       stripe_handler_type >()
      ( fake_2d_source , fake_2d_target , 0 , handler_args_with_bc_guess , njobs ) ;
      
    // we now have filtered data in target, but the stripes along the magins
    // in x-direction (1 runup wide) are wrong, because we applied whatever
    // boundary conditions inherent to the filter, while the data in fact
    // continued from one line end to the next one's beginning.
    // this is why we have the data in 'margin', and we now copy them to the
    // relevant section of 'target'
                
    margin_target = margin ;
    
    // finally we have to fix the first and last few values, which weren't
    // touched by the margin operation (due to margin's offset and length)
    // note how we move back from 'math_type' to 'out_value_type'.
    
    for ( int i = 0 ; i < runup ; i++ )
      output[i] = out_value_type ( head[i] ) ;
    
    int j = tail.size() - tail_size - runup ;
    for ( int i = output.size() - tail_size - runup ;
          i < output.size() ; i++ , j++ )
      output[i] = out_value_type ( tail[j] ) ;

  } // end of first _process_1d() overload
  
  /// specialized processing of nD input/output. We have established
  /// that the data are nD. Now we process the axes in turn, passing
  /// the per-axis handler args.

  void _process_nd ( const input_array_type & input ,
                     output_array_type & output ,
                     const std::vector
                           < typename stripe_handler_type::arg_type >
                           & handler_args ,
                     int njobs ) const
  {
    _process_nd ( input , output , 0 ,
                  handler_args [ 0 ] , njobs ) ;
    
    for ( int axis = 1 ; axis < dimension ; axis++ )
    {
      // note the different argument signature here: the first argument
      // is now 'output', because the run for axis 0 above has deposited
      // it's output there and that's where we need to pick it up for
      // filtering along the other axes - now we're operating in-place.

      _process_nd ( output , output , axis ,
                    handler_args [ axis ] , njobs ) ;
    }
  }
  
  // in the next section we have code processing nD data along
  // a specific axis. The code starts with an operator() overload
  // meant to be called from 'outside'. This is meant for cases
  // where filtering needs to be done differently for different
  // axes. After that we have the actual processing code.
  
  /// this operator() overload for single-axis processing takes
  /// plain arrays for input and output, they may be either 1D or nD.
  /// again we use _on_dimension, now with a different argument
  /// signature (we have 'axis' now). As _on_dimension is a
  /// variadic template, we 'reemerge' in the right place.
  
  void operator() ( const input_array_type & input ,
                    output_array_type & output ,
                    int axis ,
                    const typename stripe_handler_type::arg_type
                      & handler_args ,
                    int njobs = zimt::default_njobs ) const
  {
    _on_dimension ( is_1d_type() ,
                    input , output , axis ,
                    handler_args , njobs ) ;
  }
  
  /// processing of nD input/output. we have established that the data
  /// are nD. now we look at the data type. If the data are multi-channel,
  /// they need to be element-expanded for further processing, which
  /// is done by '_on_expand'. Note that there are two variants of
  /// _on_expand, one for plain arrays and one for stacks of arrays,
  /// which are coded after the operator() overload taking stacks of
  /// arrays further down.

  template < typename ... types >
  void _process_nd ( types ... args ) const
  {
    // we're all set for processing a single axis of data.
    // now we have a dispatch on is_1_channel_type, because if the
    // data are multi-channel, we want to element-expand the arrays.
    
    _on_expand ( is_1_channel_type() , args ... ) ;
  }
  
  /// variant of _on_expand for single arrays. this overload is called
  /// if the data are multi-channel. we element-expand the arrays, then
  /// call the single-channel overload below

  // KFJ 2020-11-19 made this overload into a template of in_t, out_t,
  // because it can be called with in_t being the same as input_array_type,
  // or in_t being output_array_type - the latter happens for axes >= 1,
  // where the input is the output of filtering axis 0 and already of
  // the same type as the output.
    
  template < typename in_t , typename out_t >
  void _on_expand ( std::false_type , // is_1_channel_type() ,
                    const in_t & input ,
                    out_t & output ,
                    int axis ,
                    const typename stripe_handler_type::arg_type
                      & handler_args ,
                    int njobs ) const
  {
    auto source = input.expand_elements() ;
    auto target = output.expand_elements() ;

    // with the element-expanded data at hand, we can now delegate to
    // the routine below, which deals with single-channel data
  
    _on_expand ( std::true_type() , // the expanded arrays are single-channel
                 source ,
                 target ,
                 axis + 1 ,         // processing axis is now one higher
                 handler_args ,
                 njobs ) ;
  }  

  /// Variant of _on_expand for single arrays. This is the single-channel
  /// overload. The arrays now hold fundamentals, either because that was
  /// their original data type or because they have been element-expanded.
  /// Now we finally get to do the filtering.
  /// Note how we introduce input and output as templates, since we can't
  /// be sure of their type: they may or may not have been element-expanded.

  // reinplementation of _on_expand using the new multithreading logic.
  // The original implementation assigned subsets of the workload to
  // individual worker threads. This new version assigns the same
  // task to all workers: fetch batches of lines and process them
  // until there are no more. The new version provides better locality,
  // granularity and code simplicity.

  template < typename in_t , typename out_t >
  void _on_expand ( std::true_type , // is_1_channel_type() ,
                    const in_t & input ,
                    out_t & output ,
                    int axis ,
                    const typename stripe_handler_type::arg_type
                      & handler_args ,
                    int njobs ) const
  {
    // size is the size of a 1-element thick slice perpendicular to
    // the processing axis. We set up a zimt::atomic holding that number,
    // which will be accessed by the worker threads in batches of vsize
    // to be buffered, filtered and written out to the target array.
    // think of the zimt::atomic as a ticket dispenser giving out tickets
    // with successive numbers pertaining to lines needing processing.

    auto size = input.size() / input.shape [ axis ] ;
    zimt::atomic < std::ptrdiff_t > tickets ( size ) ;
    
    // we'll use this worker code, which simply calls 'present' with
    // the arguments it needs, including, as first argument, a pointer to
    // the zimt::atomic above which yields indexes of 1D subarrays. Using
    // a lambda with cach-by-reference here is a neat way of producing the
    // std::function<void()> which 'multithread' expects

    std::function < void() > worker =
    [&]()
    {
      present < in_t , out_t , stripe_handler_type >
        ( &tickets , &input , &output , &handler_args , axis ) ;
    } ;

    // finally we use multithread() to set up njobs worker threads which
    // all run the same code, fetching batches of 1D subarrays until they
    // have all been processed.

    zimt::multithread ( worker , njobs ) ;
  }

  /// this operator() overload for single-axis processing takes
  /// std::vectors ('stacks') of arrays. this is only supported
  /// for nD data. This is a rarely-used variant; throughout zimt
  /// there isn't currently any place where this routine is called,
  /// but it's needed for some special applications. If you are studying
  /// the code, you may safely disregard the remainder of the code in
  /// this class definition; the two _on_expand variants below are
  /// also for the special case of 'stacks' of arrays.

  // With a bit of adapter code, this path could be used for
  // processing vigra's 'chunked arrays': for every axis, put
  // all sequences of chunks collinear to that axis into a
  // std::vector (as MultiArrayViews, not the data themselves),
  // then pass these stacks to this routine. TODO: try
  // As long as one sequence of chunks fits into memory, the
  // process should be efficient, allowing filtering of very large
  // data sets.
  
  void operator() ( const std::vector<input_array_type> & input ,
                    std::vector<output_array_type> & output ,
                    int axis ,
                    const typename stripe_handler_type::arg_type
                      & handler_args ,
                    int njobs = zimt::default_njobs ) const
  {
    static_assert ( ! is_1d_type() ,
                    "processing of stacked 1D arrays is not supported" ) ;
                    
    _process_nd ( input , output , axis ,
                  handler_args , njobs ) ;
  }
  
  /// variant of _on_expand for stacks of arrays.
  /// this overload is called if the data are multi-channel.
  /// we element-expand the arrays, then call the single-channel
  /// overload below
    
  template < typename in_vt , typename out_vt >
  void _on_expand ( std::false_type , // is_1_channel_type() ,
                    const std::vector < in_vt > & input ,
                    std::vector < out_vt > & output ,
                    int axis ,
                    const typename stripe_handler_type::arg_type
                      & handler_args ,
                    int njobs ) const
  {
    typedef zimt::view_t < dimension + 1 , in_ele_type >
      e_input_array_type ;  
      
    typedef zimt::view_t < dimension + 1 , out_ele_type >
      e_output_array_type ;

    // note how we expand the element channel to dimension 0, to make sure
    // that adjacent 1D subarrays will be processed together.

    std::vector<e_input_array_type> source ;
    for ( auto & e : input )
      source.push_back ( e.expand_elements ( 0 ) ) ;
    
    std::vector<e_output_array_type> target ;
    for ( auto & e : output )
      target.push_back ( e.expand_elements ( 0 ) ) ;

    // with the element-expanded data at hand, we can now delegate to
    // the routine below, which deals with single-channel data
  
    _on_expand ( std::true_type() , // the expanded arrays are single-channel
                 source ,
                 target ,
                 axis + 1 ,         // processing axis is now one higher
                 handler_args ,
                 njobs ) ;
  }  

  /// variant of _on_expand for stacks of arrays
  /// this is the single-channel overload. The arrays now hold fundamentals,
  /// either because that was their original data type or because they have
  /// been element-expanded. We end up in this routine for all processing
  /// of stacks and finally get to do the filtering.

  // reinplementation of _on_expand using the new multithreading logic.
  // The original implementation assigned subsets of the workload to
  // individual worker threads. This new version assigns the same
  // task to all workers: fetch batches of lines and process them
  // until there are no more. The new version provides better locality,
  // granularity and code simplicity.

  template < typename in_vt , typename out_vt >
  void _on_expand ( std::true_type , // is_1_channel_type() ,
                    const std::vector < in_vt > & input ,
                    std::vector < out_vt > & output ,
                    int axis ,
                    const typename stripe_handler_type::arg_type
                      & handler_args ,
                    int njobs ) const
  {
    // size is the size of a 1-element thick slice perpendicular to
    // the processing axis. We set up a zimt::atomic holding that number,
    // which will be accessed by the worker threads in batches of vsize
    // to be buffered, filtered and written out to the target array.
    // think of the zimt::atomic as a ticket dispenser giving out tickets
    // with successive numbers pertaining to lines needing processing.

    auto size = input[0].size() / input[0].shape [ axis ] ;
    zimt::atomic < std::ptrdiff_t > tickets ( size ) ;
    
    // we'll use this worker code, which simply calls 'present' with
    // the arguments it needs, including, as first argument, a pointer to
    // the zimt::atomic above which yields indexes of 1D subarrays. Using
    // a lambda with cach-by-reference here is a neat way of producing the
    // std::function<void()> which 'multithread' expects

    std::function < void() > worker =
    [&]()
    {
      vpresent < in_vt , out_vt , stripe_handler_type >
        ( &tickets , &input , &output , &handler_args , axis ) ;
    } ;
    
    // finally we use multithread() to set up njobs worker threads which
    // all run the same code, fetching batches of 1D subarrays until they
    // have all been processed.
    
    zimt::multithread ( worker , njobs ) ;
  }

} ; // struct separable_filter

} ; // namespace detail

/// zimt::filter is the common entry point for filter operations
/// in zimt. This routine does not yet do any processing, it's
/// purpose is to convert it's arguments to 'canonical' format
/// and then call the actual filter code in namespace detail.
/// It also determines the type used for arithmetic operations.
/// The type specification for input and output assures that only
/// arrays with the same dimensionality are accepted, and a static
/// assertion makes sure the number of channels match. canonical
/// form means that input and output value type are either
/// fundamental (for single-channel data) or zimt::xel_t of a
/// fundamental data type. This way, the input and output is
/// presented in a neutral form.

template < typename in_type ,
           typename out_type ,
           unsigned int D ,
           class filter_type ,
           typename ... types >
void filter ( const zimt::view_t < D , in_type > & input ,
              zimt::view_t < D , out_type > & output ,
              types ... args )
{
  // find out the elementary (fundamental) type of in_type and out_type
  // by using vigra's ExpandElementResult mechanism.

  typedef typename zimt::get_ele_t < in_type >
                   :: type in_ele_type ;
                   
  typedef typename zimt::get_ele_t < out_type >
                   :: type out_ele_type ;

  // get the number of channels and make sure it's consistent

  enum { channels = zimt::get_ele_t < in_type > :: size } ;
  
  static_assert ( channels
                  == zimt::get_ele_t < out_type > :: size ,
          "separable_filter: input and output data type must have the same number of channels" ) ;
  
  // produce the canonical types for both data types and arrays

  typedef canonical_type < in_type > canonical_in_value_type ;
  typedef zimt::view_t < D , canonical_in_value_type > cn_in_type ;
  
  typedef canonical_type < out_type > canonical_out_value_type ;
  typedef zimt::view_t < D , canonical_out_value_type > cn_out_type ;

  // call separable_filter with arrays reinterpreted as canonical types,
  // and all other arguments unchecked and unchanged.
  
  detail::separable_filter < cn_in_type ,
                             cn_out_type ,
                             filter_type >()
               ( reinterpret_cast < const cn_in_type & > ( input ) ,
                 reinterpret_cast < cn_out_type & > ( output ) ,
                 args ... ) ;
}

/// amplify is used to copy input to output, optionally applying
/// 'boost' in the process. If the operation is in-place and 'boost'
/// is 1, 'amplify' returns prematurely.

template < unsigned int dimension ,
           typename in_value_type ,
           typename out_value_type ,
           typename math_ele_type >
void amplify ( const zimt::view_t
                     < dimension , in_value_type >  & input ,
               zimt::view_t 
                      < dimension , out_value_type > & output ,
               math_ele_type boost = 1 ,
               int njobs = zimt::default_njobs
             )
{
  // if the operation is in-place and boost is 1,
  // there is nothing to do.

  if (    (void*) ( input.data() ) == (void*) ( output.data() )
       && boost == math_ele_type ( 1 ) )
    return ;

  assert ( input.size() == output.size() ) ;

  if ( std::is_same < in_value_type , out_value_type > :: value )
  {
    output.copy_data ( input ) ;
    return ;
  }
  
  // amplify has built-in type conversion.
  // TODO: amplify with boost == 1 is futile.

  amplify_type < in_value_type , out_value_type > amp ( boost ) ;
  transform ( amp , input , output ) ;
}

END_ZIMT_SIMD_NAMESPACE
HWY_AFTER_NAMESPACE() ;

#endif // sentinel
