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
    resides in wielding::process. For the time being, if user code
    needs the extended features of the wieding code, it has to invoke
    the wielding functions directly; zimt::transform and relatives
    use the established signature.
*/

#ifndef ZIMT_WIELDING_H

#include <atomic>
#include <algorithm>
#include "array.h"
#include "unary_functor.h"
#include "bill.h"

namespace wielding
{
typedef std::size_t ic_type ;

/// vs_adapter wraps a zimt::unary_functor to produce a functor which is
/// compatible with the wielding code. This is necessary, because zimt's
/// unary_functors take 'naked' arguments if the data are 1D, while the
/// wielding code always passes xel_t. The operation of this wrapper
/// class should not have a run-time effect; it's simply converting
/// references.
/// While it would be nice to simply pass through the unwrapped
/// unary_functor, this would force us to deal with the distinction
/// between data in xel_t and 'naked' fundamentals deeper down in the
/// code, and here is a good central place where we can route to uniform
/// access via xel_t - possibly with only one element.
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

  template < class C >
  class has_capped_eval_t
  {
      typedef typename C::in_v in_v ;
      typedef typename C::out_v out_v ;

      // we allow the cap to be passed as const&, &, and by value

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
  // padding.

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

/// class get_crd is an implementation of a 'get_t' class which the
/// rolling-out code uses to produce input values to the functor 'act'.
///
template < typename T ,    // elementary type
           std::size_t N , // number of channels
           std::size_t D , // dimension of the view/array
           std::size_t vsize = zimt::vector_traits < T > :: vsize >
struct get_crd
{
  typedef zimt::xel_t < T , N > value_t ;
  typedef zimt::simdized_type < value_t , vsize > value_v ;
  typedef typename value_v::value_type value_ele_v ;
  typedef zimt::xel_t < long , D > crd_t ;

  const std::size_t d ;

  // get_crd's c'tor receives the step witdh. Other similar
  // classes might use other c'tor arguments. This get_crd
  // holds no state apart from 'd' and 'step'.

  get_crd ( const std::size_t & _d = 0 )
  : d ( _d )
  { }

  // init is used to initialize the vectorized value to the value
  // it should hold at the beginning of the peeling run. The discrete
  // coordinate 'crd' gives the location of the first value, and the
  // get_crd infers the start value from it. The scalar value is
  // not used until peeling is done, so it isn't initialized here.

  void init ( value_v & cv , const crd_t & crd )
  {
    cv = crd ;
    cv [ d ] += value_ele_v::iota() ;
  }

  // initialize the scalar value from the discrete coordinate.
  // This needs to be done after peeling, the scalar value
  // is not initialized before and not updated during peeling.

  void init ( value_t & c , const crd_t & crd )
  {
    c = crd ;
  }

  // increase modifies it's argument to contain the next value, or
  // next vectorized value, respectively

  void increase ( value_t & trg )
  {
    ++ trg [ d ] ;
  }

  void increase ( value_v & trg )
  {
    trg [ d ] += vsize ;
  }
} ;

// class storer disposes of the results of calling eval. This
// functionality is realized with separate 'put_t' classes to
// increase flexibility, just as the get_t classes with get_crd
// above as an example. Here, init only receives the starting
// coordinate of the 'run', and the c'tor receives a reference
// to a target view and the axis of the run.
// Other put_t classes might dispose of values differently; the
// most extreme would be to ignore the values (e.g. when the
// 'act' functor implements a reduction and the single result
// values are not needed). Such a scenario would allow the code
// to generate reductions from very large 'virtual' arrays, if
// the get_t class generates values without having to load them
// from memory.

template < typename T ,
           std::size_t N ,
           std::size_t M = N ,
           std::size_t L = zimt::vector_traits < T > :: vsize >
struct storer
{
  typedef zimt::xel_t < T , N > value_t ;
  typedef zimt::simdized_type < value_t , L > value_v ;
  typedef typename value_v::value_type value_ele_v ;
  typedef zimt::xel_t < long , M > crd_t ;
  typedef zimt::simdized_type < long , L > crd_v ;
  typedef zimt::xel_t < std::ptrdiff_t , L > sc_indexes_t ;

  const std::size_t d ;
  zimt::view_t < M , value_t > & trg ;
  value_t * p_trg ;
  const std::size_t stride ;

  // get_t's c'tor receives the step witdh. Other similar
  // classes might use other c'tor arguments. This get_t
  // holds no state apart from 'd' and 'step'.

  storer ( zimt::view_t < M , value_t > & _trg ,
           const std::size_t & _d = 0 )
  : trg ( _trg ) , d ( _d ) , stride ( _trg.strides [ _d ] )
  { }

  // init is used to initialize the target pointer

  void init ( const crd_t & crd )
  {
    p_trg = & ( trg [ crd ] ) ;
  }

  // save writes to the current taget pointer and increases the
  // pointer by the amount of value_t written

  void save ( const value_t & v )
  {
    *p_trg = v ;
    p_trg += stride ;
  }

  void save ( const value_v & v )
  {
    v.fluff ( p_trg , stride ) ;
    p_trg += L * stride ;
  }

  // note this 'save' overload: 'leftover' values which remain after
  // 'peeling' are collected in a buffer until a full vector is ready.
  // The data from this buffer need to be stored to the correct loci
  // in memory, which are encoded in 'indexes' as a set of offsets
  // from the view's/array's origin.

  void save ( const value_v & v , const sc_indexes_t & indexes )
  {
    auto po = (T*)(trg.origin) ;
    auto ix = indexes * N ;

    for ( int chn = 0 ; chn < N ; chn++ )
    {
      v[chn].scatter ( po + chn , ix ) ;
    }

    p_trg += L ;
  }
} ;

// For brevity, here's a using declaration deriving the default
// get_t and put_t from the actor's type and the array's/view's
// dimension.

template < typename act_t , std::size_t dimension >
using norm_get_crd
  = wielding::get_crd
  < typename act_t::in_ele_type ,
    act_t::dim_in ,
    dimension ,
    act_t::vsize > ;

template < typename act_t , std::size_t dimension >
using norm_put_t
  = wielding::storer
  < typename act_t::out_ele_type ,
    act_t::dim_out ,
    dimension ,
    act_t::vsize > ;

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
               zimt::bill_t bill ,
               const get_t & co ,
               const put_t & pt )
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

    // buffer for 'leftovers' from the end of the segment,
    // scalar coordinate, target memory pointer and help
    // pointer to beginning of array's data, set of scatter
    // indexes to target locations of leftovers. Note that the
    // scatter indexes are xel_t of ptrdiff_t, not index_type.
    // The resulting scatter will use a goading loop rather than
    // a 'true' hardware gather, but since this only has to be
    // done once per thread (at conclusion), it's no big deal.

    in_type tail [ vsize ] ;
    std::size_t tail_index = 0 ;
    in_type crd ;
    out_type * trg ;
    out_type * trg0 = out_view.origin ;
    zimt::xel_t < std::ptrdiff_t , vsize > sc_indexes = 0 ;

    // these SIMD variables will hold one batch if input or
    // output, respectively. Here, the SIMD input is a
    // vectorized coordinate

    in_v md_crd ;
    out_v buffer ;

    auto c = co ;
    put_t p ( pt ) ;

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

      trg = slice2.origin + slice2.offset ( out_mci [ line ] ) ;
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

      // initialize the vectorized coordinate. This coordinate will
      // remain constant except for the component indexing the
      // processing axis, which will be counted up as we go along.
      // This makes the index calculations very efficient: for one
      // vectorized evaluation, we only need a single vectorized
      // addition where the vectorized coordinate is increased by
      // vsize. As we go along, we also initialize a scalar coordinate
      // which will be used for 'leftovers'

      auto scrd = out_mci [ line ] ;
      long ofs = segment * segment_size ;

      // initially, we calculate discrete coordinates in long.

      typedef zimt::xel_t < long , dimension > dcrd_t ;
      typedef zimt::simdized_type < dcrd_t , vsize > md_dcrd_t ;
      typedef typename md_dcrd_t::value_type dcomp_t ;

      dcrd_t dcrd ;
      md_dcrd_t md_dcrd ;

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

      // we create a local copy of the get_t and call init on
      // it, passing in the discrete coordinate of the beginning of
      // the current line and receiving an initial value of md_crd,
      // the vectorized datum suitable for processing with 'act'.

      c.init ( md_crd , dcrd ) ;
      p.init ( dcrd ) ;

      // first we perform a peeling run, processing data vectorized
      // as long as there are enough data to fill the vectorized
      // buffers (md_XXX_data_type).

      if ( out_stride == 1 )
      {
        for ( ic_type a = 0 ; a < nr_vectors ; a++ )
        {
          act.eval ( md_crd , buffer ) ;
          p.save ( buffer ) ;
          trg += vsize ;
          if ( a < nr_vectors - 1 )
            c.increase ( md_crd ) ;
        }
      }
      else
      {
        for ( ic_type a = 0 ; a < nr_vectors ; a++ )
        {
          act.eval ( md_crd , buffer ) ;
          p.save ( buffer ) ;
          trg += vsize * out_stride ;
          if ( a < nr_vectors - 1 )
            c.increase ( md_crd ) ;
        }
      }

      // peeling is done, any leftovers are cumulated in 'tail'
      // until 'tail' has a full vector's worth of input.

      // update the scalar coordinate to point to the first
      // unprocessed location

      dcrd [ axis ] += nr_vectors * vsize ;
      c.init ( crd , dcrd ) ;

      for ( ic_type r = 0 ; r < leftover ; r++ )
      {
        // save the offset of the target pointer from the
        // array's origin

        sc_indexes [ tail_index ] = trg - trg0 ;

        // save the current scalar coordinate to 'tail'

        tail [ tail_index++ ] = crd ;

        // let the get_t produce the next scalar value

        c.increase ( crd ) ;

        // update target pointer

        trg += out_stride ;

        // if we now have a full vector's worth of data to
        // process, we call the vectorized eval

        if ( tail_index == vsize )
        {
          in_v in ;
          in.bunch ( tail ) ;
          act.eval ( in , buffer ) ;

          // we can't do a simple 'fluff' because the target
          // memory isn't 'regular', instead we have to do a
          // set of scatter operations to the indexes we've
          // saved in sc_indexes.

          p.save ( buffer , sc_indexes ) ;

          // finally we reset tail_index to start at the
          // beginning of 'tail' with the next leftover

          tail_index = 0 ;
        }
      }
    }

    // if we have any leftovers here (tail_index > 0) it
    // can't be a full vector's worth, so now we pad 'tail'
    // with the last 'genuine' value, call capped 'eval'
    // and store the result.

    if ( tail_index )
    {
      in_v in ;

      for ( std::size_t i = tail_index ; i < vsize ; i++ )
      {
        tail [ i ] = tail [ tail_index - 1 ] ;
        sc_indexes [ i ] = sc_indexes [ tail_index - 1 ] ;
      }

      in.bunch ( tail ) ;
      act.eval ( in , buffer , tail_index ) ;

      p.save ( buffer , sc_indexes ) ;
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

// class loader is a get_t which loads data from a zimt::view.
// With this get_t, process can be used to implement coupled_f
// We have two variants which differ only in the use of the argument
// 'stride' when calling 'bunch'. TODO: may not make a performance
// difference to call bunch with stride==1 vs. call without stride,
// in which case the unstrided variant would be superfluous.

template < typename T ,
           std::size_t N ,
           std::size_t M = N ,
           std::size_t L = zimt::vector_traits < T > :: vsize >
struct loader
{
  typedef zimt::xel_t < T , N > value_t ;
  typedef zimt::simdized_type < value_t , L > value_v ;
  typedef typename value_v::value_type value_ele_v ;
  typedef zimt::xel_t < long , M > crd_t ;
  typedef zimt::simdized_type < long , L > crd_v ;

  const std::size_t d ;
  const zimt::view_t < M , value_t > & src ;
  const value_t * p_src ;
  const std::size_t stride ;

  // get_t's c'tor receives the step witdh. Other similar
  // classes might use other c'tor arguments. This get_t
  // holds no state apart from 'd' and 'step'.

  loader ( const zimt::view_t < M , value_t > & _src ,
           const std::size_t & _d = 0 )
  : src ( _src ) , d ( _d ) , stride ( _src.strides [ _d ] )
  { }

  // init is used to initialize the vectorized value to the value
  // it should hold at the beginning of the peeling run. The discrete
  // coordinate 'crd' gives the location of the first value, and the
  // get_t infers the start value from it. The scalar value is
  // not used until peeling is done, so it isn't initialized here.

  void init ( value_v & cv , const crd_t & crd )
  {
    p_src = & ( src [ crd ] ) ;
    cv.bunch ( p_src , stride ) ;
  }

  // Initialize the scalar value from the discrete coordinate.
  // This needs to be done after peeling, the scalar value
  // is not initialized before and not updated during peeling.

  void init ( value_t & c , const crd_t & crd )
  {
    p_src = & ( src [ crd ] ) ;
    c = * p_src ;
  }

  // increase modifies it's argument to contain the next value, or
  // next vectorized value, respectively

  void increase ( value_t & trg )
  {
    ++ p_src ;
    trg = *p_src ;
  }

  void increase ( value_v & trg )
  {
    p_src += L ;
    trg.bunch ( p_src , stride ) ;
  }
} ;

template < typename T ,
           std::size_t N ,
           std::size_t M = N ,
           std::size_t L = zimt::vector_traits < T > :: vsize >
struct unstrided_loader
: public loader < T , N , M , L >
{
  typedef loader < T , N , M , L > base_t ;

  using typename base_t::crd_t ;
  using typename base_t::value_t ;
  using typename base_t::value_v ;
  using base_t::src ;
  using base_t::p_src ;
  using base_t::base_t ;

  void init ( value_v & cv , const crd_t & crd )
  {
    p_src = & ( src [ crd ] ) ;
    cv.bunch ( p_src ) ;
  }

  void init ( value_t & c , const crd_t & crd )
  {
    p_src = & ( src [ crd ] ) ;
    c = * p_src ;
  }

  void increase ( value_t & trg )
  {
    ++ p_src ;
    trg = *p_src ;
  }

  void increase ( value_v & trg )
  {
    p_src += L ;
    trg.bunch ( p_src ) ;
  }
} ;

// wielding code to co-process two equally shaped views, which is
// also used for 'apply', passing the same view as in- and output.
// The calling code (zimt::transform) has already wrapped 'act',
// the functor converting input values to output values, with
// wielding::vs_adapter, and cast the views to hold xel_t, even if
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
    process ( _act , out_view , bill , ld , pt ) ;
  }
  else
  {
    loader < in_ele_type , chn_in , dimension , vsize >
      ld ( in_view , axis ) ;
    process ( _act , out_view , bill , ld , pt ) ;
  }
}

} ; // namespace wielding

#define ZIMT_WIELDING_H
#endif
