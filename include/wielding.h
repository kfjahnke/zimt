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

/*! \file wielding.h

    \brief Implementation of zimt::transform

    wielding.h provides code to process all 1D sub-views of nD views.
    This is similar to using vigra::Navigator, which also iterates over
    1D subarrays of nD arrays. Here, this access is hand-coded to have
    complete control over the process, and to work with range-based
    code rather than the iterator-based approach vigra uses.
    
*/

#ifndef ZIMT_WIELDING_H

#include <atomic>
#include <algorithm>
#include "array.h"
#include "unary_functor.h"
#include "bill.h"

namespace wielding
{
typedef int ic_type ;

/// vs_adapter wraps a zimt::unary_functor to produce a functor which is
/// compatible with the wielding code. This is necessary, because zimt's
/// unary_functors take 'naked' arguments if the data are 1D, while the
/// wielding code always passes TinyVectors. The operation of this wrapper
/// class should not have a run-time effect; it's simply converting references.
/// While it would be nice to simply pass through the unwrapped unary_functor,
/// this would force us to deal with the distinction between data in xel_t
/// and 'naked' fundamentals deeper down in the code, and here is a good central
/// place where we can route to uniform access via xel_t - possibly with
/// only one element.
/// By inheriting from inner_type, we provide all of inner_type's type system
/// which we don't explicitly override.
/// Rest assured: the reinterpret_cast is safe. If the data are single-channel,
/// the containerized version takes up the same meory as the uncontainerized
/// version of the datum. multi-channel data are containerized anyway.

template < class inner_type >
struct vs_adapter
: public inner_type
{
  using typename inner_type::in_ele_v ;
  using typename inner_type::out_ele_v ;
  
  typedef typename inner_type::in_nd_ele_type in_type ;
  typedef typename inner_type::out_nd_ele_type out_type ;
  typedef typename inner_type::in_nd_ele_v in_v ;
  typedef typename inner_type::out_nd_ele_v out_v ;
                              
  vs_adapter ( const inner_type & _inner )
  : inner_type ( _inner )
  { } ;
  
  /// eval overload for unvectorized arguments

  void eval ( const in_type & in ,
                   out_type & out )
  {
    inner_type::eval
      ( reinterpret_cast < const typename inner_type::in_type & > ( in ) ,
        reinterpret_cast < typename inner_type::out_type & > ( out ) ) ;
  }

  /// vectorized evaluation function. This is enabled only if vsize > 1
  
  template < typename = std::enable_if < ( inner_type::vsize > 1 ) > >
  void eval ( const in_v & in ,
                   out_v & out )
  {
    inner_type::eval
      ( reinterpret_cast < const typename inner_type::in_v & > ( in ) ,
        reinterpret_cast < typename inner_type::out_v & > ( out ) ) ;
  }
} ;

// wielding code to co-process two equally shaped views, which is
// also used for 'apply', passing the same view as in- and output.
// The calling code (zimt::transform) has already wrapped 'act',
// the functor converting input values to output values, with
// wielding::vs_adapter, and cast the views to hold xel_t, even if
// the data are single-channel.
// this replaces the older code which was unnecessarily convoluted.

template < class act_t , std::size_t dimension >
void coupled_f ( const act_t & _act ,
                 zimt::view_t < dimension ,
                                typename act_t::in_type
                              > in_view ,
                 zimt::view_t < dimension ,
                                typename act_t::out_type
                              > out_view ,
                 const zimt::bill_t & bill )
{
  // transform has already checked, but we doublecheck:

  assert ( in_view.shape == out_view.shape ) ;

  // we extract the act functor's type system and constants

  typedef typename act_t::in_type in_type ;
  typedef typename act_t::out_type out_type ;
  typedef typename act_t::in_ele_type in_ele_type ;

  typedef typename act_t::out_type out_type ;
  typedef typename act_t::out_ele_type out_ele_type ;

  typedef typename act_t::in_v in_v ;
  typedef typename act_t::out_v out_v ;

  static const std::size_t chn_in = act_t::dim_in ;
  static const std::size_t chn_out = act_t::dim_out ;
  static const std::size_t vsize = act_t::vsize ;

  const auto & axis ( bill.axis ) ; // short notation
  const auto & segment_size ( bill.segment_size ) ; // ditto

  // extract the strides for input and output

  auto in_stride = in_view.strides [ axis ] ;
  auto out_stride = out_view.strides [ axis ] ;

  // create slices holding the start positions of the lines

  auto slice1 = in_view.slice ( axis , 0 ) ;
  auto slice2 = out_view.slice ( axis , 0 ) ;

  // and multi-coordinate iterators over these slices

  zimt::mci_t < dimension - 1 > in_mci ( slice1.shape ) ;
  zimt::mci_t < dimension - 1 > out_mci ( slice2.shape ) ;

  // is this maybe an 'apply'?

  const bool is_apply
    = ( (void*) slice1.origin == (void*) slice2.origin ) ;

  // get the line length and the number of lines

  long length = in_view.shape [ axis ] ;
  auto nr_lines = slice1.size() ;

  // get the number of line segments

  std::size_t last_segment_size = segment_size ;

  std::ptrdiff_t nr_segments = length / segment_size ;
  if ( length % segment_size )
  {
    last_segment_size = length % segment_size ;
    nr_segments++ ;
  }

  // calculate the total number of joblet indexes

  std::ptrdiff_t nr_indexes = nr_lines * nr_segments ;

  // set up the atomic to share out the joblet indexes
  // to the worker threads

  zimt::atomic < std::ptrdiff_t > indexes ( nr_indexes ) ;

  // set up the payload code for 'multithread' which fetches joblet
  // 'tickets' and processes each ticket

  auto worker =
  [&]()
  {
    // variable to hold a joblet index fetched from the atomic 'indexes'

    std::ptrdiff_t joblet_index ;

    // this is only needed for reductions, but the compiler should
    // notice if the copy is unnecessary (copy elision). With the
    // copy, reductions can use the same code as ordinary transforms

    act_t act ( _act ) ; // create per-thread copy of '_act'

    // loop as long as there are joblet indices left

    while ( zimt::fetch_ascending ( indexes , nr_indexes , joblet_index ) )
    {
      // terminate early on cancellation request - the effect is not
      // immediate, but reasonably granular: the check is once per
      // segment.

      if ( bill.p_cancel && bill.p_cancel->load() )
        break ;

      // decode the joblet index into it's components, the current
      // line and the segment within this line

      std::size_t line = joblet_index / nr_segments ;
      std::size_t segment = joblet_index % nr_segments ;

      // this gives us source and target address - first we obtain
      // the lines' base address, then we add the segment's offset

      auto src = slice1.origin + slice1.offset ( in_mci [ line ] ) ;
      src += segment * in_stride * segment_size ;

      auto trg = slice2.origin + slice2.offset ( out_mci [ line ] ) ;
      trg += segment * out_stride * segment_size ;

      // how many values do we have in the current segment?
      // the last segment may be less than segment_size long

      std::size_t nr_values ;

      if ( segment == nr_segments - 1 )
        nr_values = last_segment_size ;
      else
        nr_values = segment_size ;

      // we'll first process batches of vsize values with SIMD code,
      // then the leftovers with scalar code

      auto nr_vectors = nr_values / vsize ;
      auto leftover = nr_values % vsize ;

      // these SIMD variables will hold one batch if input or output,
      // respectively.

      in_v in_buffer ;
      out_v out_buffer ;

      // first we perform a peeling run, processing data vectorized
      // as long as there are enough data to fill the vectorized
      // buffers (md_XXX_data_type).
      // depending on whether the input/output is strided or not,
      // and on the vector size and number of channels,
      // we pick different overloads of 'bunch' and fluff'. The
      // overloads without stride may use InterleavedMemoryWrapper,
      // or, for single-channel data, SIMD load/store operations,
      // which is most efficient. We can only pick the variants
      // using InterleavedMemoryWrapper if vsize is a multiple of
      // the hardware SIMD register size, hence the rather complex
      // conditionals. But the complexity is rewarded with optimal
      // peformance.

      if ( in_stride == 1 )
      {
        if ( out_stride == 1 )
        {
          for ( ic_type a = 0 ; a < nr_vectors ; a++ )
          {
            in_buffer.bunch ( src ) ;
            src += vsize ;
            act.eval ( in_buffer , out_buffer ) ;
            out_buffer.fluff ( trg ) ;
            trg += vsize ;
          }
        }
        else
        {
          for ( ic_type a = 0 ; a < nr_vectors ; a++ )
          {
            in_buffer.bunch ( src ) ;
            src += vsize ;
            act.eval ( in_buffer , out_buffer ) ;
            out_buffer.fluff ( trg , out_stride ) ;
            trg += vsize * out_stride ;
          }
        }
      }
      else
      {
        if ( out_stride == 1 )
        {
          for ( ic_type a = 0 ; a < nr_vectors ; a++ )
          {
            in_buffer.bunch ( src , in_stride ) ;
            src += vsize * in_stride ;
            act.eval ( in_buffer , out_buffer ) ;
            out_buffer.fluff ( trg ) ;
            trg += vsize ;
          }
        }
        else
        {
          for ( ic_type a = 0 ; a < nr_vectors ; a++ )
          {
            in_buffer.bunch ( src , in_stride ) ;
            src += vsize * in_stride ;
            act.eval ( in_buffer , out_buffer ) ;
            out_buffer.fluff ( trg , out_stride ) ;
            trg += vsize * out_stride ;
          }
        }
      }

      // peeling is done, we mop up the remainder with scalar code
      // KFJ 2022-05-19 initially I coded so that an apply would have
      // to take care not to write to out and read in subsequently,
      // but I think the code should rather be defensive and avoid
      // the problem without user code having to be aware of it.
      // hence the test for equality of src and trg.

      if ( leftover )
      {
        if ( is_apply )
        {
          // this is an 'apply', avoid write-before-read
          out_type help ;
          for ( ic_type r = 0 ; r < leftover ; r++ )
          {
            act.eval ( *src , help ) ;
            *trg = help ;
            src += in_stride ;
            trg += out_stride ;
          }
        }
        else
        {
          // this is not an 'apply'
          for ( ic_type r = 0 ; r < leftover ; r++ )
          {
            act.eval ( *src , *trg ) ;
            src += in_stride ;
            trg += out_stride ;
          }
        }
      }
    }
  } ; // end of payload code

  // with the atomic distributing joblet indexes and the payload code
  // established, we call multithread to invoke worker threads to invoke
  // the payload code repeatedly until all joblets are complete

  // TODO: if the views involved aren't 'large', multithreading is
  // counterproductive. So: establish a good heuristic to look at
  // the amount of data and decide whether to multithread at all
  // and, if so, with how many threads.

  zimt::multithread ( worker , bill.njobs ) ;

} // coupled_f

template < class act_t >
void coupled_f ( const act_t & act ,
                 zimt::view_t < 1 ,
                                typename act_t::in_type
                              > in_view ,
                 zimt::view_t < 1 ,
                                typename act_t::out_type
                              > out_view ,
                 const zimt::bill_t & bill
               )
{
  zimt::view_t < 2 , typename act_t::in_type > vi
    ( in_view.origin ,
      { long ( in_view.strides[0] ) ,
        long ( in_view.strides[0] * in_view.shape[0] ) } ,
      { in_view.shape[0] , 1UL } ) ;

  zimt::view_t < 2 , typename act_t::out_type > vo
    ( out_view.origin ,
      { long ( out_view.strides[0] ) ,
        long ( out_view.strides[0] * out_view.shape[0] ) } ,
      { out_view.shape[0] , 1UL } ) ;

  coupled_f ( act , vi , vo , bill ) ;
}

// implementation of 'index-based' transform. class act_t is a
// zimt::unary_functor taking coordinates and producing the data type
// out_view references. This implies that out_view has the same number
// of dimensions as the functor's input has channels: a coordinate into
// that view.

template < class act_t >
void indexed_f ( const act_t & _act ,
                 zimt::view_t < act_t::dim_in ,
                                typename act_t::out_type
                              > out_view ,
                 zimt::bill_t bill
               )
{
  // we extract the act functor's type system and constants

  typedef typename act_t::in_type in_type ;
  typedef typename act_t::in_ele_type in_ele_type ;
  typedef typename act_t::out_type out_type ;
  typedef typename act_t::out_ele_type out_ele_type ;

  typedef typename act_t::in_v in_v ;
  typedef typename act_t::out_v out_v ;

  static const std::size_t chn_in = act_t::dim_in ;
  static const std::size_t chn_out = act_t::dim_out ;
  static const std::size_t vsize = act_t::vsize ;

  const auto & axis ( bill.axis ) ; // short notation
  const auto & segment_size ( bill.segment_size ) ; // ditto

  // extract the strides of the target view

  auto out_stride = out_view.strides [ axis ] ;

  // create a slice holding the start position of the lines

  auto slice2 = out_view.slice ( axis , 0 ) ;

  // and a multi-coordinate iterator over the slice

  zimt::mci_t < chn_in - 1 > out_mci ( slice2.shape ) ;

  // get the line length and the number of lines

  long length = out_view.shape [ axis ] ;
  auto nr_lines = slice2.size() ;

  // get the number of line segments

  std::size_t last_segment_size = segment_size ;

  std::ptrdiff_t nr_segments = length / segment_size ;
  if ( length % segment_size )
  {
    last_segment_size = length % segment_size ;
    nr_segments++ ;
  }

  // calculate the total number of joblet indexes

  std::ptrdiff_t nr_indexes = nr_lines * nr_segments ;

  // set up the atomic to share out the joblet indexes
  // to the worker threads

  zimt::atomic < std::ptrdiff_t > indexes ( nr_indexes ) ;

  // set up the payload code for 'multithread' which fetches joblet
  // 'tickets' and processes each ticket

  auto worker =
  [&]()
  {
    // variable to hold a joblet index fetched from the atomic 'indexes'

    std::ptrdiff_t joblet_index ;

    // this is only needed for reductions, but the compiler should
    // notice if the copy is unnecessary (copy elision). With the
    // copy, reductions can use the same code as ordinary transforms

    act_t act ( _act ) ; // create per-thread copy of '_act'

    // loop as long as there are joblet indices left

    while ( zimt::fetch_ascending ( indexes , nr_indexes , joblet_index ) )
    {
      // terminate early on cancellation request - the effect is not
      // immediate, but reasonably granular: the check is once per
      // segment.

      if ( bill.p_cancel && bill.p_cancel->load() )
        break ;

      // decode the joblet index into it's components, the current
      // line and the segment within this line

      std::size_t line = joblet_index / nr_segments ;
      std::size_t segment = joblet_index % nr_segments ;

      // this gives us the target address - first we obtain the
      // line's base address, then we add the segment's offset

      auto trg = slice2.origin + slice2.offset ( out_mci [ line ] ) ;
      trg += segment * out_stride * segment_size ;

      // how many values do we have in the current segment?
      // the last segment may be less than segment_size long

      std::size_t nr_values ;

      if ( segment == nr_segments - 1 )
        nr_values = last_segment_size ;
      else
        nr_values = segment_size ;

      // we'll first process batches of vsize values with SIMD code,
      // then the leftovers with scalar code

      auto nr_vectors = nr_values / vsize ;
      auto leftover = nr_values % vsize ;

      // these SIMD variables will hold one batch if input or output,
      // respectively. Here, the SIMD input is a vectorized coordinate

      in_v md_crd ;
      out_v buffer ;

      // initialize the vectorized coordinate. This coordinate will
      // remain constant except for the component indexing the
      // processing axis, which will be counted up as we go along.
      // This makes the index calculations very efficient: for one
      // vectorized evaluation, we only need a single vectorized
      // addition where the vectorized coordinate is increased by
      // vsize. As we go along, we also initialize a scalar coordinate
      // which will be used for 'leftovers'

      auto scrd = out_mci [ line ] ;
      std::size_t ofs = segment * segment_size ;
      in_type crd ;

      for ( int d = 0 , ds = 0 ; d < chn_in ; d++ )
      {
        if ( d != axis )
        {
          md_crd[d] = scrd[ds] ;
          crd[d] = scrd[ds] ;
          ++ds ;
        }
        else
        {
          for ( int e = 0 ; e < vsize ; e++ )
            md_crd[d][e] = in_ele_type ( ofs + e ) ;
          crd[d] = ofs ;
        }
      }

      // first we perform a peeling run, processing data vectorized
      // as long as there are enough data to fill the vectorized
      // buffers (md_XXX_data_type).

      if ( out_stride == 1 )
      {
        for ( ic_type a = 0 ; a < nr_vectors ; a++ )
        {
          act.eval ( md_crd , buffer ) ;
          buffer.fluff ( trg ) ;
          trg += vsize ;
          md_crd[axis] += vsize ;
        }
      }
      else
      {
        for ( ic_type a = 0 ; a < nr_vectors ; a++ )
        {
          act.eval ( md_crd , buffer ) ;
          buffer.fluff ( trg , out_stride ) ;
          trg += vsize * out_stride ;
          md_crd[axis] += vsize ;
        }
      }

      // peeling is done, any leftovers are processed one-by-one

      crd[axis] += nr_vectors * vsize ;

      for ( ic_type r = 0 ; r < leftover ; r++ )
      {
        act.eval ( crd , *trg ) ;
        trg += out_stride ;
        crd[axis]++ ;
      }
    }
  } ; // end of payload code

  // with the atomic distributing joblet indexes and the payload code
  // established, we call multithread to invoke worker threads to invoke
  // the payload code repeatedly until all joblets are complete

  // TODO: if the views involved aren't 'large', multithreading is
  // counterproductive. So: establish a good heuristic to look at
  // the amount of data and decide whether to multithread at all
  // and, if so, with how many threads.

  zimt::multithread ( worker , bill.njobs ) ;

} // indexed_f

// overload for 1D

template < class act_t >
void indexed_f ( const act_t & _act ,
                 zimt::view_t < 1 ,
                                typename act_t::out_type
                              > out_view ,
                 zimt::bill_t bill
               )
{
  // we extract the act functor's type system and constants

  typedef typename act_t::in_type in_type ;
  typedef typename act_t::in_ele_type in_ele_type ;
  typedef typename act_t::out_type out_type ;
  typedef typename act_t::out_ele_type out_ele_type ;

  typedef typename act_t::in_v in_v ;
  typedef typename act_t::out_v out_v ;

  static const std::size_t chn_in = act_t::dim_in ;
  static const std::size_t chn_out = act_t::dim_out ;
  static const std::size_t vsize = act_t::vsize ;

  const auto & axis ( bill.axis ) ; // short notation
  const auto & segment_size ( bill.segment_size ) ; // ditto

  assert ( axis == 0 ) ;

  // extract the strides of the target view

  auto out_stride = out_view.strides [ axis ] ;

  // create a slice holding the start position of the lines

  auto slice2 = out_view.slice ( axis , 0 ) ;

  // and a multi-coordinate iterator over the slice

  long out_index = 0 ;

  // get the line length and the number of lines

  long length = out_view.shape [ axis ] ;
  auto nr_lines = 1 ;

  // get the number of line segments

  std::size_t last_segment_size = segment_size ;

  std::ptrdiff_t nr_segments = length / segment_size ;
  if ( length % segment_size )
  {
    last_segment_size = length % segment_size ;
    nr_segments++ ;
  }

  // calculate the total number of joblet indexes

  std::ptrdiff_t nr_indexes = nr_segments ;

  // set up the atomic to share out the joblet indexes
  // to the worker threads

  zimt::atomic < std::ptrdiff_t > indexes ( nr_indexes ) ;

  // set up the payload code for 'multithread' which fetches joblet
  // 'tickets' and processes each ticket

  auto worker =
  [&]()
  {
    // variable to hold a joblet index fetched from the atomic 'indexes'

    std::ptrdiff_t joblet_index ;

    // this is only needed for reductions, but the compiler should
    // notice if the copy is unnecessary (copy elision). With the
    // copy, reductions can use the same code as ordinary transforms

    act_t act ( _act ) ; // create per-thread copy of '_act'

    // loop as long as there are joblet indices left

    while ( zimt::fetch_ascending ( indexes , nr_indexes , joblet_index ) )
    {
      // terminate early on cancellation request - the effect is not
      // immediate, but reasonably granular: the check is once per
      // segment.

      if ( bill.p_cancel && bill.p_cancel->load() )
        break ;

      // decode the joblet index into it's components, the current
      // line and the segment within this line

      std::size_t segment = joblet_index % nr_segments ;

      // this gives us the target address - frst we obtain the
      // line's base address, then we add the segment's offset

      auto trg = out_view.origin ;
      trg += segment * out_stride * segment_size ;

      // how many values do we have in the current segment?
      // the last segment may be less than segment_size long

      std::size_t nr_values ;

      if ( segment == nr_segments - 1 )
        nr_values = last_segment_size ;
      else
        nr_values = segment_size ;

      // we'll first process batches of vsize values with SIMD code,
      // then the leftovers with scalar code

      auto nr_vectors = nr_values / vsize ;
      auto leftover = nr_values % vsize ;

      // these SIMD variables will hold one batch if input or output,
      // respectively. Here, the SIMD input is a vectorized coordinate

      in_v md_crd ;
      out_v buffer ;

      // initialize the vectorized coordinate. This coordinate will
      // remain constant except for the component indexing the
      // processing axis, which will be counted up as we go along.
      // This makes the index calculations very efficient: for one
      // vectorized evaluation, we only need a single vectorized
      // addition where the vectorized coordinate is increased by
      // vsize. As we go along, we also initialize a scalar coordinate
      // which will be used for 'leftovers'

      std::size_t ofs = segment * segment_size ;
      in_type crd ;
      crd[0] = ofs ;
      md_crd[0] = md_crd[0].iota() + ofs ;

      // first we perform a peeling run, processing data vectorized
      // as long as there are enough data to fill the vectorized
      // buffers (md_XXX_data_type).

      if ( out_stride == 1 )
      {
        for ( ic_type a = 0 ; a < nr_vectors ; a++ )
        {
          act.eval ( md_crd , buffer ) ;
          buffer.fluff ( trg ) ;
          trg += vsize ;
          md_crd[0] += vsize ;
        }
      }
      else
      {
        for ( ic_type a = 0 ; a < nr_vectors ; a++ )
        {
          act.eval ( md_crd , buffer ) ;
          buffer.fluff ( trg , out_stride ) ;
          trg += vsize * out_stride ;
          md_crd[0] += vsize ;
        }
      }

      // peeling is done, any leftovers are processed one-by-one

      crd[0] += nr_vectors * vsize ;

      for ( ic_type r = 0 ; r < leftover ; r++ )
      {
        act.eval ( crd , *trg ) ;
        trg += out_stride ;
        crd[0]++ ;
      }
    }
  } ; // end of payload code

  // with the atomic distributing joblet indexes and the payload code
  // established, we call multithread to invoke worker threads to invoke
  // the payload code repeatedly until all joblets are complete

  // TODO: if the views involved aren't 'large', multithreading is
  // counterproductive. So: establish a good heuristic to look at
  // the amount of data and decide whether to multithread at all
  // and, if so, with how many threads.

  zimt::multithread ( worker , bill.njobs ) ;

} // indexed_f

} ; // namespace wielding

#define ZIMT_WIELDING_H
#endif
