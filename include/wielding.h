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

/// vs_adapter wraps a zimt::unary_functor to produce a functor which is
/// compatible with the wielding code. This is necessary, because zimt's
/// unary_functors take 'naked' arguments if the data are single-channel,
/// while the wielding code always passes xel_t. The operation of this
/// wrapper class should not have a run-time effect; it's simply converting
/// references. Note here that zimt::unary_functor is just an empty
/// shell with a bunch of type declarations and constants: it's there
/// as a base class for user code functors, to provide a uniform
/// interface and to facilitate the coding of the 'simdized xel'
/// types which zimt uses.
/// While it would be nice to simply pass through the unwrapped
/// unary_functor, this would force us to deal with the distinction
/// between data in xel_t and 'naked' fundamentals deeper down in the
/// code, and here is a good central place where we can route to uniform
/// access via xel_t - possibly with only one element. We rely on the
/// optimizer to optimize the xel_t of one value_type - use of that
/// construct is only for syntactic uniformity.
/// Note that the wielding code uses *only* the vectorized eval member
/// function of inner_type. The scalar version may be present but it
/// won't be called by the wielding code. If inner_type is to do a
/// reduction - e.g. cumulate statistics as it is repeatedly applied
/// during the transform - it must implement a separate eval member
/// function taking a third argument, 'cap', which is invoked if a
/// processing thread has odd values left over at the end of it's
/// 'run' which aren't enough to fill an entire vector. The wielding
/// code will pass input to this eval member function which is
/// appropriately padded, so the invocation of the vectorized eval
/// is safe, but the cap value will indicate how many of the vector's
/// lanes hold 'genuine' data - the rest are padding, repeating the
/// last 'genuine' value. The padding is applied so that functors
/// which ignore 'cap' (this should be the majority) can do so without
/// causing exceptions due to unsuitable input, provided that the
/// 'genuine' lanes don't causes such exceptions.
/// If the wrapped functor, 'inner_type', does not have a capped eval
/// overload, vs_adapter will route to inner_type's 'uncapped' eval,
/// ignoring the cap value.
/// This mode of processing 'odd bits' may seem wasteful, but I think
/// that - compared to the usual 'segment sizes' used in zimt, which
/// are several hundred per 'run' - and many af them usually without
/// any odd bits (the segment size is a power of two, odd bits only
/// occur at the end of an entire line, if at all) - the extra effort
/// used for padding is quite negligible. I may add a separate code
/// path for segments with no odd bits if this assumption turns out
/// wrong. For now I am content with a clean, uniform design which
/// avoids even the possibility of exceptions due to unpopulated
/// SIMD lanes when processing the last batch of data in a run, and
/// makes it possible to process all data with SIMD code, rather than
/// performing a separate 'peeling' run with scalar code.

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
  
  void eval ( const in_v & in ,
                   out_v & out )
  {
    inner_type::eval
      ( reinterpret_cast < const typename inner_type::in_v & > ( in ) ,
        reinterpret_cast < typename inner_type::out_v & > ( out ) ) ;
  }

private:

  // To detect if a unary functor has a capped eval overload, I use code
  // adapted from from Valentin Milea's anser to
  // https://stackoverflow.com/questions/87372/check-if-a-class-has-a-member-function-of-a-given-signature
  // when providing a capped eval overload of your eval function, please
  // note that it will only be recognized if one of the three signatures
  // tested below is matched precisely.

  template < class C >
  class has_capped_eval_t
  {
      typedef typename C::in_v in_v ;
      typedef typename C::out_v out_v ;

      // we allow the cap to be passed as const&, &, and by value,
      // the remainder of the signature is 'as usual'

      template < class T , class cap_t >
      static std::true_type testSignature
        (void (T::*)(const in_v&, out_v&, const cap_t&));

      template < class T , class cap_t >
      static std::true_type testSignature
        (void (T::*)(const in_v&, out_v&, cap_t&));

      template < class T , class cap_t >
      static std::true_type testSignature
        (void (T::*)(const in_v&, out_v&, cap_t));

      template <class T>
      static decltype(testSignature(&T::eval)) test(std::nullptr_t);

      template <class T>
      static std::false_type test(...);

  public:
      using type = decltype(test<C>(nullptr));
      static const bool value = type::value;
  };

  // does inner_type have a capped eval function? If so, the adapter
  // will dispatch to it, otherwise it will call eval without cap.
  // Most of the time, inner_type won't have a capped variant - this
  // is only useful for reductions, all evaluations inside the
  // wiedling code are coded so that all vectors passed to eval are
  // padded if necessary and therefore 'technically' full and safe
  // to process. The cap is only a hint that some of the lanes were
  // generated by padding wih the last 'genuine' lane, but a reduction
  // needs to be aware of the fact and ignore the lanes filled with
  // padding. An alternative to this method would be the use of
  // masked code, but this would require additional coding effort
  // to handle the masked/capped vectors, and I much prefer to keep
  // the code as simple and straightforward as I can.

  static const bool has_capped_eval
    = has_capped_eval_t<inner_type>::value ;

  void _eval ( std::true_type ,
               const in_v & in ,
                    out_v & out ,
               const std::size_t & cap )
  {
    inner_type::eval
      ( reinterpret_cast < const typename inner_type::in_v & > ( in ) ,
        reinterpret_cast < typename inner_type::out_v & > ( out ) ,
        cap ) ;
  }

  void _eval ( std::false_type ,
               const in_v & in ,
                    out_v & out ,
               const std::size_t & cap )
  {
    inner_type::eval
      ( reinterpret_cast < const typename inner_type::in_v & > ( in ) ,
        reinterpret_cast < typename inner_type::out_v & > ( out ) ) ;
  }

public:

  // the adapter's capped eval will invoke inner_type's capped eval
  // if it's present, and 'normal' uncapped eval otherwise.

  void eval ( const in_v & in ,
                   out_v & out ,
              const std::size_t & cap )
  {
    _eval ( std::integral_constant < bool , has_capped_eval >() ,
            in , out , cap ) ;
  }

} ;

/// process is now the central function in the wielding namespace
/// and used to implement all 'transform-like' operations. It is
/// a generalization of the 'transform' concept, adding choosable
/// 'get' and 'put' methods which generate input for the functor
/// and dispose of it's output. By factoring these two activities
/// out, the code becomes much more flexible, and by selecting
/// simple appropriate 'get' and 'put' objects, zimt::transform
/// can easily be implemented as a specialized application of the
/// more general code in 'process'.
/// So 'process' uses a three-step process: the 'get' object
/// produces data the 'act' functor can process, the 'act'
/// functor produces output data from this input, and the 'put'
/// object disposes of the output.
/// 'process' 'aggregates' the data so that all of the input will
/// be processed by vector code (unless the data themselves are not
/// vectorizable), so the 'act' functor may omit a scalar eval
/// member function and provide vectorized only.

template < class act_t ,
           std::size_t dimension = act_t::dim_in , // default is dodgy
           class get_t = norm_get_crd < act_t , dimension > ,
           class put_t = norm_put_t < act_t , dimension > >
void process ( const act_t & _act ,
               zimt::view_t < dimension ,
                              typename act_t::out_type
                            > out_view ,
               const get_t & co ,
               const put_t & pt ,
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

  // extract the strides of the target view

  auto out_stride = out_view.strides [ axis ] ;

  // create a slice holding the start position of the lines

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

    act_t act ( _act ) ; // create per-thread copy of '_act'

    // these SIMD variables will hold one batch if input or
    // output, respectively.

    in_v md_in ;
    out_v md_out ;

    auto c = co ;
    put_t p ( pt ) ;

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

      p.init ( dcrd ) ;

      if ( nr_vectors == 0 )
      {
        if ( leftover )
        {
          // special case: there are no vectors, only leftover
          // we need a special init which caps the result, and
          // applies 'stuffing' (see below)

          c.init ( md_in , dcrd , leftover ) ;
          act.eval ( md_in , md_out , leftover ) ;
          p.save ( md_out , leftover ) ; // need partial save
        }
      }
      else
      {
        // we create a local copy of the get_t and call init on
        // it, passing in the discrete coordinate of the beginning of
        // the current line and receiving an initial value of md_in,
        // the vectorized datum suitable for processing with 'act'.

        c.init ( md_in , dcrd ) ;

        // first we perform a peeling run, processing data vectorized
        // as long as there are enough data to fill an entire md_out

        for ( std::size_t a = 0 ; a < nr_vectors ; a++ )
        {
          act.eval ( md_in , md_out ) ;
          p.save ( md_out ) ;
          if ( a < nr_vectors - 1 )
            c.increase ( md_in ) ;
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
          c.increase ( md_in , leftover , false ) ;
          act.eval ( md_in , md_out , leftover ) ;
          p.save ( md_out , leftover ) ;
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

// wielding code to co-process two equally shaped views, which is
// also used for 'apply', passing the same view as in- and output.
// The calling code (zimt::transform) has already wrapped 'act',
// the functor converting input values to output values, with
// vs_adapter, and cast the views to hold xel_t, even if
// the data are single-channel.
// With 'loader' as 'get_t', we can use 'process' to implement
// this function:

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
  typedef typename act_t::in_ele_type in_ele_type ;

  static const std::size_t chn_in = act_t::dim_in ;
  static const std::size_t vsize = act_t::vsize ;

  const auto & axis ( bill.axis ) ;
  norm_put_t < act_t , dimension > pt ( out_view , axis ) ;

  if ( in_view.strides [ axis ] == 1 )
  {
    unstrided_loader < in_ele_type , chn_in , dimension , vsize >
      ld ( in_view , axis ) ;
    process ( _act , out_view , ld , pt , bill ) ;
  }
  else
  {
    loader < in_ele_type , chn_in , dimension , vsize >
      ld ( in_view , axis ) ;
    process ( _act , out_view , ld , pt , bill ) ;
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
