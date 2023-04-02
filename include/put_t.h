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

/*! \file get_t.h

    \brief classes to provide input to an 'act' functor

    zimt::process feeds the 'act' functor, working along
    the aggregation axis, while the coordinate's component along the
    other axes remain constant. This effectively 1D operation can
    be coded efficiently, whereas the 'act' functor would typically
    look at it's input as an nD entity and not be aware of the fact
    that only one of the input's components is 'moving'. Once the
    'act' functor is done, it produces a vectorized result which
    is in turn disposed of by a put_t object.
    This header provides a set of classes which are meant to fit into
    this slot. They follow a specific pattern which results from the
    logic in zimt_process, namely the two init and two save overloads.

*/

#ifndef ZIMT_PUT_T_H

namespace zimt
{

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
           std::size_t D ,
           std::size_t L >
struct storer
{
  typedef zimt::xel_t < T , N > value_t ;
  typedef zimt::simdized_type < value_t , L > value_v ;
  typedef typename value_v::value_type value_ele_v ;
  typedef zimt::xel_t < long , D > crd_t ;

  const std::size_t & d ;
  zimt::view_t < D , value_t > & trg ;
  const std::size_t stride ;

  value_t * p_trg ;

  // get_t's c'tor receives the target array and the processing
  // axis. Similar classes might use other c'tor arguments. This
  // get_t holds no variable state apart from p_trg.

  storer ( zimt::view_t < D , value_t > & _trg ,
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

  void save ( const value_v & v )
  {
    v.fluff ( p_trg , stride ) ;
    p_trg += L * stride ;
  }

  // capped save, used for the final batch of data which did not
  // fill out an entire value_v

  void save ( const value_v & v , std::size_t cap )
  {
    v.fluff ( p_trg , stride , cap ) ;
  }
} ;

// vstorer stores vectorized data to a zimt::view_t of vectorized
// values. This is a good option for storing intermediate results,
// because it can use efficient SIMD store operations rather than
// having to interleave the data to store them as xel of T. To
// retrieve the data from such a vectorized storage array, use
// class vloader (see get_t.h).
// The design is so that vstorer can operate within the logic used
// by zimt::process: the discrete coordinate taken by 'init' refers
// to 'where the datum would be in an array of value_t'. The array
// of value_v which the data are stored to has to be suitably sized:
// along the 'hot' axis, the size should be so that, multiplied with
// the lane count, it will be greater or equal the 'notional' size
// of the unvectorized array - for efficiency reasons, class vstorer
// will only store 'full' simdized values, as they are produced by
// the 'act' functor. Along the other axes, the size must be the same
// as the 'notional' size of the equivalent unvectorized array.

template < typename T ,
           std::size_t N ,
           std::size_t D ,
           std::size_t L >
struct vstorer
{
  typedef zimt::xel_t < T , N > value_t ;
  typedef zimt::simdized_type < value_t , L > value_v ;
  typedef typename value_v::value_type value_ele_v ;
  typedef zimt::xel_t < long , D > crd_t ;

  const std::size_t & d ;
  zimt::view_t < D , value_v > & trg ;
  const std::size_t stride ;
  const std::size_t shift ;

  value_v * p_trg ;

  // get_t's c'tor receives the target array and the processing
  // axis. Similar classes might use other c'tor arguments. This
  // get_t holds no variable state apart from p_trg.

  vstorer ( zimt::view_t < D , value_v > & _trg ,
            const std::size_t & _d = 0 )
  : trg ( _trg ) , d ( _d ) , stride ( _trg.strides [ _d ] ) ,
    shift ( std::log2 ( L ) )
  {
    assert ( ( 1 << shift ) == L ) ;
  }

  // init is used to initialize the target pointer

  void init ( const crd_t & crd )
  {
    auto crdv = crd ;
    crdv [ d ] <<= shift ;
    p_trg = & ( trg [ crdv ] ) ;
  }

  // save writes to the current taget pointer and increases the
  // pointer by the amount of value_t written

  void save ( const value_v & v , std::size_t cap = 0 )
  {
    *p_trg = v ;
    p_trg += stride ;
  }
} ;

// discard_result is a put_t which doesn't do anything, it's only
// there to fill the put_t 'slot'. It's useful for situations where
// the results of the 'act' functor aren't needed, e.g. in reductions
// where the 'act' functor performs statistics. Another use case is
// performance tests, where the act functor is invoked repeatedly,
// but only it's performance is of interest and the cost of saving
// it's output should be avoided.

template < typename T ,
           std::size_t N ,
           std::size_t M = N ,
           std::size_t L = zimt::vector_traits < T > :: vsize >
struct discard_result
{
  typedef zimt::xel_t < T , N > value_t ;
  typedef zimt::simdized_type < value_t , L > value_v ;
  typedef typename value_v::value_type value_ele_v ;
  typedef zimt::xel_t < long , M > crd_t ;
  typedef zimt::simdized_type < long , L > crd_v ;
  typedef zimt::xel_t < std::ptrdiff_t , L > sc_indexes_t ;

  void init ( const crd_t & crd )
  { }

  template < typename ... A >
  void save ( A ... args )
  { }
} ;

// For brevity, here's a using declaration deriving the default
// put_t from the actor's type and the array's/view's dimension.

template < typename act_t , std::size_t dimension >
using norm_put_t
  = storer
  < typename act_t::out_ele_type ,
    act_t::dim_out ,
    dimension ,
    act_t::vsize > ;


} ; // namespace zimt

#define ZIMT_PUT_T_H
#endif
