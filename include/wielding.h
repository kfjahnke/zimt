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

    \brief abstraction of zimt::transform

    wielding.h provides code to process all 1D sub-views of nD views.
    This is similar to using vigra::Navigator, which also iterates over
    1D subarrays of nD arrays. Here, this access is hand-coded to have
    complete control over the process, and to work with range-based
    code rather than the iterator-based approach vigra uses.
    This is a new attempt at the code and it does not use a
    scalar eval variant: the data are now always processed with
    a vector function, and if any 'odd bits' are left over, an
    overload of the vectorized eval taking a 'cap' argument may be
    invoked, which does process the given input to output just
    as the uncapped overload, but may make use of the cap value
    if the operation is a reduction. The wielding code pads the
    vector with the 'odd bits' with the last available value, so
    that - provided that value is valid - the entire vector holds
    valid input. This approach allows to make no assumptions about
    what valid input might be (like, some neutral element) or
    whether lanes with invalid input may or may not produce an
    exception. The order of the data is - per definition - not
    an issue, there are no guarantees concerning the sequence in
    which values are processed. The invocation of a 'capped' eval
    overload only happens if such an overload is present at all;
    this is determined by looking at the functor passed in,
    dispatching to one of two code paths. If the functor has no
    capped eval overload, the 'ordinary' uncapped eval is always
    used, which is guaranteed to be safe (due to padding). This
    behaviour makes it easier to implement functors, because most
    of the time capped eval is not used at all (it's only for
    reductions) so users should not be forced to supply it.

    The code in wielding.h is an abstraction of the 'transform'
    concept, adding selectable 'get' and 'put' abjects which handle
    production of input and disposing of output of the functor.
    This design widens the scope of possible operations considerably,
    and the implementation of the transform family of functions is
    merely a very specialized use case of the generalized code, which
    resides in zimt::process. For the time being, if user code
    needs the extended features of the wieding code, it has to invoke
    the wielding functions directly; zimt::transform and relatives
    use the established signature.

    I used to keep the 'wielding' code in a separate namespace, but
    decided to move it's content to the zimt namespace for sake of
    simplicity, and to not have any 'dangling' symbols when 'dubbing'
    the zimt namespace, e.g. for multi-ISA constructs.
*/

#ifndef ZIMT_WIELDING_H

#include <atomic>
#include <algorithm>
#include "array.h"
#include "unary_functor.h"
#include "bill.h"
#include "get_t.h"
#include "put_t.h"

namespace zimt
{
typedef std::size_t ic_type ;

/// 'process' is the central function in the wielding namespace
/// and used to implement all 'transform-like' operations. It is
/// a generalization of the 'transform' concept, adding choosable
/// 'get' and 'put' methods generating input for the 'act' functor
/// and disposing of it's output. By factoring these two activities
/// out, the code becomes much more flexible, and by selecting
/// simple appropriate 'get' and 'put' objects, zimt::transform
/// can easily be implemented as a specialized application of the
/// more general code in 'process'.
/// So 'process' uses a three-step process: the 'get' object
/// produces data the 'act' functor can process, the 'act'
/// functor produces output data from this input, and the 'put'
/// object disposes of the output. The sequence implies that
/// the type of data the get_t object provides is the same which
/// the act functor expects as input. The same holds true for the
/// act functor's output, which has to agree with the type of datum
/// the put_t object can dispose of.
/// Some uses of 'process' will not need the entire set of three
/// processing elements, but 'process' nevertheless expects them;
/// they may be 'dunnies', though, which will be optimized away.
/// 'process' 'aggregates' the data so that all of the input will
/// be processed by vector code (unless the data themselves are not
/// vectorizable), so the 'act' functor may omit a scalar eval
/// member function and provide vectorized eval only.
/// The template signature could make an attempt to be more specific,
/// but as it is it's nice and clear and works well with ATD, so that
/// invocations of zimt::process can typically invoke it without
/// specifying any template arguments. I moved the 'notional' shape
/// to the front of the argument list, because it provides the
/// 'scaffolding' for the entire operation, which relies on discrete
/// coordinates to initialize the get_t and put_t objects, calculate
/// the number of full vectors and 'leftovers' etc. - followed by the
/// three 'active' elements, and finally the 'loading bill' which is
/// often left at the default.

template < std::size_t dimension ,
           class get_t ,
           class act_t ,
           class put_t >
void process ( zimt::xel_t < std::size_t , dimension > shape ,
               const get_t & _get ,
               const act_t & _act ,
               const put_t & _put ,
               zimt::bill_t bill = zimt::bill_t() )
{
  // we extract the act functor's type system and constants

  typedef typename act_t::in_type in_type ;
  typedef typename act_t::in_ele_type in_ele_type ;
  typedef typename act_t::out_type out_type ;
  typedef typename act_t::out_ele_type out_ele_type ;

  typedef typename act_t::in_v in_v ;
  typedef typename act_t::in_ele_v in_ele_v ;
  typedef typename act_t::out_v out_v ;
  typedef typename act_t::out_ele_v out_ele_v ;

  typedef typename in_ele_v::mask_type in_mask_type ;

  static const std::size_t chn_in = act_t::dim_in ;
  static const std::size_t chn_out = act_t::dim_out ;
  static const std::size_t vsize = act_t::vsize ;

  const auto & axis ( bill.axis ) ; // short notation
  const auto & segment_size ( bill.segment_size ) ; // ditto

  // create a slice holding the start position of the lines
  // TODO: avoid detour via a view, calc. slice directly

  zimt::view_t < dimension , out_type > out_view ( shape ) ;
  auto slice2 = out_view.slice ( axis , 0 ) ;

  // and a multi-coordinate iterator over the slice

  zimt::mci_t < slice2.dimension > out_mci ( slice2.shape ) ;

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

    // act_t act ( _act ) ; // create per-thread copy of 'act'
    auto act = zimt::grok ( _act ) ; // create per-thread copy of 'act'

    // we need per-thread copies of the get_t and put_t objects:

    get_t get ( _get ) ;
    put_t put ( _put ) ;

    // these SIMD variables will hold one batch if input or
    // output, respectively.

    in_v md_in ;
    out_v md_out ;

    // loop as long as there are joblet indices left

    while ( zimt::fetch_ascending
             ( indexes , nr_indexes , joblet_index ) )
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

      // initialize the discrete coordinate. This coordinate will
      // remain constant except for the component indexing the
      // processing axis, which will be counted up as we go along.
      // This makes the index calculations very efficient: for one
      // vectorized evaluation, we only need a single vectorized
      // addition where the vectorized coordinate is increased by
      // vsize. As we go along, we also initialize a scalar coordinate
      // which will be used for 'leftovers'. Note how these coordinates
      // are only used indirectly by passing them to the get_t and put_t
      // objects, which yield input to, and dispose of output from, the
      // 'act' functor.

      auto scrd = out_mci [ line ] ;
      long ofs = segment * segment_size ;

      // initially, we calculate discrete coordinates in long.

      zimt::xel_t < long , dimension > dcrd ;

      for ( long d = 0 , ds = 0 ; d < dimension ; d++ )
      {
        if ( d != axis )
        {
          dcrd[d] = scrd[ds] ;
          ++ds ;
        }
        else
        {
          dcrd[d] = ofs ;
        }
      }

      put.init ( dcrd ) ;

      if ( nr_vectors == 0 )
      {
        if ( leftover )
        {
          // special case: there are no vectors, only leftover
          // we need a special init which caps the result, and
          // applies 'stuffing' (see below)

          get.init ( md_in , dcrd , leftover ) ;
          act.eval ( md_in , md_out , leftover ) ;
          put.save ( md_out , leftover ) ;
        }
      }
      else
      {
        // we create a local copy of the get_t and call init on
        // it, passing in the discrete coordinate of the beginning of
        // the current line and receiving an initial value of md_in,
        // the vectorized datum suitable for processing with 'act'.

        get.init ( md_in , dcrd ) ;

        // first we perform a peeling run, processing data vectorized
        // as long as there are enough data to fill an entire md_out

        for ( std::size_t v = 0 ; v < nr_vectors ; v++ )
        {
          act.eval ( md_in , md_out ) ;
          put.save ( md_out ) ;
          if ( v < nr_vectors - 1 )
            get.increase ( md_in ) ;
        }

        // if there are any scalar values left over, we use 'capped'
        // operations to process them, 'leftover' serving as cap value.
        // These operations only affect part of the vectorized data.
        // If there are no full vectors at all (see further up), the
        // lanes from the cap up are padded with the last value before
        // the cap (this is affected by passing 'stuff' true) - but
        // this is a rare exception. Most of the time at least one
        // full vectorized datum was processed, so all lanes hold valid
        // data, and 'stuffing' isn't needed. Capped operations are less
        // efficient than uncapped code, falling back to 'goading',
        // but they occur only rarely if the lines along the processing
        // axis are long, and not at all if the segments all divide by
        // the lane count. The capped processing is a recent design
        // decision in favour of cumulating source data until a full
        // vector's worth are available - this did not perform well,
        // probably due to increased register pressure and detrimental
        // memory access patterns. Here, the number of pipeline calls
        // (invocations of the 'act' functor') is slightly larger, but
        // the invocation occurs relating to just the memory which is
        // 'currently at hand' (in cache) and no buffer or scatter
        // indexes are needed. The buffering etc. could only pay off if
        // the pipeline were very long and expensive.

        if ( leftover )
        {
          get.increase ( md_in , leftover , false ) ;
          act.eval ( md_in , md_out , leftover ) ;
          put.save ( md_out , leftover ) ;
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
}

// wielding code to get-process two equally shaped views, which is
// also used for 'apply', passing the same view as in- and output.
// The calling code (zimt::transform) has already wrapped 'act',
// the functor converting input values to output values, with
// vs_adapter, and cast the views to hold xel_t, even if
// the data are single-channel.
// With 'loader' as 'get_t', we can use 'process' to implement
// this function:

template < class act_t , std::size_t dimension >
void coupled_f ( const act_t & act ,
                 zimt::view_t < dimension ,
                                typename act_t::in_type
                              > in_view ,
                 zimt::view_t < dimension ,
                                typename act_t::out_type
                              > out_view ,
                 const zimt::bill_t & bill )
{
  typedef typename act_t::in_ele_type in_ele_type ;
  static const std::size_t chn_in = act_t::dim_in ;
  static const std::size_t chn_out = act_t::dim_out ;
  static const std::size_t vsize = act_t::vsize ;

  const auto & axis ( bill.axis ) ;
  storer < in_ele_type , chn_out , dimension , vsize >
    put ( out_view , axis ) ;

  if ( in_view.strides [ axis ] == 1 )
  {
    unstrided_loader < in_ele_type , chn_in , dimension , vsize >
      get ( in_view , axis ) ;
    process ( out_view.shape , get , act , put , bill ) ;
  }
  else
  {
    loader < in_ele_type , chn_in , dimension , vsize >
      get ( in_view , axis ) ;
    process ( out_view.shape , get , act , put , bill ) ;
  }
}

template < typename T , std::size_t N , std::size_t L >
struct pass_through
: public zimt::unary_functor < zimt::xel_t < T , N > ,
                               zimt::xel_t < T , N > ,
                               L >
{
  template < typename I , typename O >
  void eval ( const I & i , O & o , const std::size_t cap = 0 )
  {
    o = i ;
  }
} ;

template < typename W >
zimt::uf_adapter < W >
uf_adapt ( const W & inner )
{
  return zimt::uf_adapter < W >
    ( inner ) ;
}

} ; // namespace wielding

#define ZIMT_WIELDING_H
#endif
