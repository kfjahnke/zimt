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
#include "get.h"
#include "put.h"

namespace zimt
{
typedef std::size_t ic_type ;

/// 'process' is the central function in zimt and used to
/// implement all 'transform-like' operations. It is  a
/// generalization of the 'transform' concept, adding choosable
/// 'get' and 'put' functors generating input for the 'act'
/// functor and disposing of it's output. By factoring these
/// two activities out, the code becomes much more flexible,
/// and by selecting simple appropriate 'get' and 'put' objects,
/// zimt::transform can easily be implemented as a specialized
/// application of the more general code in 'process'.
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
/// they may be 'dummies', though, which will be optimized away.
/// 'process' 'aggregates' the data so that all of the input will
/// be processed by vector code (unless the data themselves are not
/// vectorizable), so the 'act' functor doesn't need a scalar eval
/// member function. It needs to provide a vectorized eval only,
/// plus - for reductions - a 'capped' eval overload which is
/// invoked if the vectorized datum passed to the 'act' functor
/// has content which was filled in to make up full vectors but
/// shouldn't be fed to the reduction.
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
/// Note the template argument gact_t: this allows passing in functors
/// which take and produce single-channel data in 'naked' form rather
/// than as xel_t of one element.

template < std::size_t D ,
           class get_t ,
           class gact_t ,
           class put_t >
void process ( const zimt::xel_t < std::size_t , D > & shape ,
               const get_t & _get ,
               const gact_t & gact ,
               const put_t & _put ,
               const zimt::bill_t & bill = zimt::bill_t() )
{
  typedef vs_adapter < gact_t > act_t ;

  // we extract the act functor's type system and constants. Note
  // that we extract these data from the 'adapted' functor - the
  // one processing xel_t only. This is what we'll actually use
  // in the per-thread processing code. But calling code is free
  // to pass in act functors which process SIMD vectors of
  // fundamentals. This makes it easier on the calling side,
  // and here is the single point where all code must pass, so
  // this is the best place to do the adaptation.

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

  typedef zimt::xel_t < long , D > crd_t ;

  // set up the lower and upper limit of the operation. This is used
  // if only a window of the 'notional' shape is to be processed. The
  // limits are taken from the bill.

  crd_t lower_limit = decode_bill_vector<D> ( bill.lower_limit ) ;

  crd_t upper_limit ;

  if ( bill.upper_limit.size() )
    upper_limit = decode_bill_vector<D> ( bill.upper_limit ) ;
  else
    upper_limit = shape ;

  // might omit this test to allow reversed operation

  for ( std::size_t i = 0 ; i < D ; i++ )
  {
    if ( lower_limit[i] >= upper_limit[i] )
    {
      auto msg = "lower_limit in bill is not smaller than upper_limit" ;
      throw std::invalid_argument ( msg ) ;
    }
  }

  // set up offsets for the get_t and put_t objects. Normally these
  // offsets are zero - a typical case where they aren't would be
  // reading from and storing to 'cropped' arrays. The offsets are
  // added unconditionally to the discrete coordinate passed to the
  // get_t/put_t object's init functions.

  crd_t get_offset = decode_bill_vector<D> ( bill.get_offset ) ;
  crd_t put_offset = decode_bill_vector<D> ( bill.put_offset ) ;

  // with the limits established, we can set up the state yielding
  // the 1D subarrays for processing. The window of coordinates we'll
  // process in this run is limited by lower_limit and upper_limit:

  auto window_shape = upper_limit - lower_limit ;

  // The lowest coordinates along 'axis' are the ones where we'll
  // start processing for individual 1D subarrays

  auto head_area_shape = window_shape ;
  head_area_shape [ axis ] = 1 ;

  // we set up an iterator over this area

  zimt::mci_t < D > head_mci ( head_area_shape ) ;

  // get the line length and the number of lines

  long length = window_shape [ axis ] ;
  auto nr_lines = head_area_shape.prod() ;

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
    // and the per-thread copies can hold individual state. In vspline,
    // the equivalent functors are const, which makes them 'pure' in a
    // functional programming sense. In zimt, most of the time, the
    // functors will also be constant, but here we rely on the compiler
    // to figure this out and optimize accordingly. In return we gain
    // a universal processing routine which can handle reductions on
    // top of the operations which can be done with a const functor.
    // At the same time we move from the general functor to the
    // adapted version (see vs_adapter)

    act_t act ( gact ) ;

    // we need per-thread copies of the get_t and put_t objects as well,
    // because they may hold individual state, e.g. pointers to memory.

    // TODO: it may be more efficient to replace the get_t and put_t
    // ojects' init functions with c'tor overloads, then hold an
    // lvalue reference to md_in, or md_out, respectively, in the
    // get_t/put_t object and avoid passing the reference in increase
    // or save. Alternatively, init might pass a pointer which could be
    // stored to avoid the reference argument in increase/save.

    get_t get ( _get ) ;
    put_t put ( _put ) ;

    // these SIMD variables will hold one batch if input or output,
    // respectively. The get_t functor will set md_in to contain the
    // next value to be processed, then the 'act' functor is invoked to
    // produce output from md_in and store the output in md_out, and
    // finally, the put_t object is invoked to dispose of the result in
    // md_out.

    in_v md_in ;
    out_v md_out ;

    // loop as long as there are joblet indices left

    while ( zimt::fetch_ascending
             ( indexes , nr_indexes , joblet_index ) )
    {
      // terminate early on cancellation request - the effect is not
      // immediate, but reasonably granular: the check is once per
      // segment. The optimizer can recognize the fact that p_cancel
      // is in fact nullptr (this is the default) and optimize the
      // test away.

      if ( bill.p_cancel && bill.p_cancel->load() )
        break ;

      // decode the joblet index into it's components, the current
      // line and the segment within this line

      std::size_t line ;
      std::size_t segment ;

      // decode the joblet index

      line = joblet_index / nr_segments ;
      segment = joblet_index % nr_segments ;

      // how many values do we have in the current segment?
      // the last segment may be less than segment_size long

      std::size_t nr_values ;

      if ( segment == nr_segments - 1 )
        nr_values = last_segment_size ;
      else
        nr_values = segment_size ;

      // we'll first process batches of vsize values with SIMD code,
      // then the leftovers with a 'capped' operation

      auto nr_vectors = nr_values / vsize ;
      auto leftover = nr_values % vsize ;

      // for segments after the first one, we'll add an offset

      long ofs = segment * segment_size ;

      // The coordinate iterator over the 'head' area gives start
      // coordinates pertaining to the chosen window

      auto dcrd = head_mci [ line ] ;

      // To obtain the start coordinate pertaining to the 'notional'
      // shape, we add the window's lower limit - usually the lower
      // limit is zero.

      dcrd += lower_limit ;

      // and, along the processing axis, an offset (if this is the
      // very first segment, the offset is zero)

      dcrd [ axis ] += ofs ;

      // now we have the start coordinate and we can init the put_t
      // object with it. Note how the coordinate is only used once to
      // initialize the get_t and put_t object via their init function,
      // what these objects 'make of it' is up to them.

      put.init ( dcrd + put_offset ) ;

      if ( nr_vectors == 0 )
      {
        if ( leftover )
        {
          // special case: there are no vectors, only leftover
          // we need a special init which caps the result, and
          // applies 'stuffing' (see below)

          get.init ( md_in , dcrd + get_offset , leftover ) ;
          act.eval ( md_in , md_out , leftover ) ;
          put.save ( md_out , leftover ) ;
        }
      }
      else
      {
        // we pass the discrete coordinate of the beginning of
        // the current line to the get_t object's init function and
        // receive an initial value of md_in, the vectorized datum
        // suitable for as input for 'act'. Note the addition of
        // 'get_offset'. This is an optional datum from the 'loading
        // bill' which can be used to make the get_t object use
        // shifted coordinates - e.g. when loading from a cropped
        // image.

        get.init ( md_in , dcrd + get_offset ) ;

        // first we perform a peeling run, processing data vectorized
        // as long as there are enough data to fill an entire md_in

        for ( std::size_t v = 0 ; v < nr_vectors ; v++ )
        {
          act.eval ( md_in , md_out ) ;
          put.save ( md_out ) ;
          if ( v < nr_vectors - 1 )
            get.increase ( md_in ) ;
        }

        // if there are any scalar values left over, we use a 'capped'
        // operation to process them, 'leftover' serving as cap value.
        // This operation only affects part of the vectorized data.
        // If there are no full vectors at all (see further up), the
        // lanes from the cap are padded with the last value before
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

} ; // namespace wielding

#define ZIMT_WIELDING_H
#endif
