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

// Here's some initial example code which walks through a few of the
// 'transform family' of functions, showing use of zimt at a high level
// and also demonstrating that 'feeding' the functions can be done with
// initializer lists and ATD, which makes it easy to interface with
// non-zimt code.
// Depending on which SIMD backend you want to employ, you can compile
// this code with different compiler options. Here are a few examples,
// using clang++:
// 1. using the 'goading' backend and no external SIMD library
// clang++ -o example example.cc
// 2. using the std::simd backend (requires std::simd headers)
// clang++ -DUSE_STDSIMD -std=c++17 -o example example.cc
// 3. using the Vc backend (requires the Vc library and headers)
// clang++ -DUSE_VC -o example example.cc -lVc
// 4. using the highway backend (requires the highway headers)
// clang++ -DUSE_HWY -o example example.cc
// Optimization with up to -Ofast should be fine. To avoid multithreading,
// add -DZIMT_SINGLETHREAD. Note that per default zimt will multithread
// with pthreads, which you may have to link in on some platforms. g++
// should also work, but is tested less. To get code specific to your
// ISA, add -march=native. For some backends (especially highway),
// additional ISA-specific flags improve performance.

#define WIELDING_SEGMENT_SIZE 32

#include <type_traits>
#include <limits>
#include <array>
#include <functional>

// #include "common.h"
// #include "xel.h"
// #include "vector.h"
// #include "array.h"
// #include "unary_functor.h"
// #include "interleave.h"
// #include "wielding.h"
#include "transform.h"

typedef zimt::xel_t < float , 3 > f3_t ;

// example for a sink functor using an atomic as reduction target.
// When the sink functor goes out of scope, it adds it's partial
// result to the atomic.

template < typename dtype > struct sum_up
: public zimt::unary_functor < dtype , dtype >
{
  typedef zimt::unary_functor < dtype > base_type ;
  using typename base_type::in_type ;
  using typename base_type::in_v ;
  using base_type::vsize ;

  // reference to a thread-safe 'pooling target'

  zimt::atomic < dtype > & collector ;
  dtype sum ;

  // this c'tor is used to create the initial sum_up object and
  // fixes the reference to the 'pooling target'

  sum_up ( zimt::atomic < dtype > & _collector )
  : collector ( _collector )
  { sum = 0 ; }

  // sum_up needs a copy c'tor to propagate the reference to
  // the 'pooling target'. Note that copy assignment won't work.

  sum_up ( const sum_up & other )
  : collector ( other.collector )
  { sum = 0 ; }

  // finally, two eval overloads, one for scalar input and
  // one for SIMD input.

  void eval ( const in_type & v , in_type & dummy )
  { sum += v ; }

  void eval ( const in_v & v , in_v & dummy )
  { sum += v.sum() ; }

  ~sum_up()
  { collector.fetch_add ( sum ) ; }
} ;

// if 'act' is always copied, reductions could be done with the
// regular transform functions.

template < typename T , std::size_t N > struct sum_up_fcpy
: public zimt::unary_functor < zimt::xel_t < T , N > >
{
  typedef zimt::xel_t < T , N > dtype ;

  typedef zimt::unary_functor < dtype > base_type ;
  using typename base_type::in_type ;
  using typename base_type::in_v ;
  using base_type::vsize ;

  // reference to a thread-safe function which will serve to
  // deposit the 'personal score' to a pooled final result

  std::function < void ( dtype ) > yield ;

  dtype * p_score = nullptr ;

  // this c'tor is used to create the initial sum_up_fcpy object
  // and fixes the yielding callback

  sum_up_fcpy ( std::function < void ( const dtype & ) > _yield )
  : yield ( _yield )
  {
    p_score = new dtype ;
    * p_score = 0 ;
  }

  // sum_up_fcpy needs a copy c'tor to propagate the yield function

  sum_up_fcpy ( const sum_up_fcpy & other )
  : yield ( other.yield )
  {
    p_score = new dtype ;
    * p_score = 0 ;
  }

  // now, two operator() overloads, one for scalar input and
  // one for SIMD input. Note the summation: the second overload
  // receives xel_t of N SIMD vectors, so to get a meaningful
  // result with N channels, the SIMD vectors are summed up
  // horizontally and their sums are added to the channels of
  // 'score'.

  void eval ( const in_type & v , in_type & dummy ) const
  {
    ( * p_score ) += v ;
  }

  void eval ( const in_v & v , in_v & dummy ) const
  {
    for ( int ch = 0 ; ch < N ; ch++ )
      (*p_score) [ ch ] += v [ ch ] . sum () ;
  }

  ~sum_up_fcpy()
  {
    yield ( * p_score ) ;
    delete p_score ;
  }
} ;

template < typename T >
struct amp13_t
: public zimt::unary_functor < T , f3_t , 16 >
{
  float factor ;

  amp13_t ( float _factor )
  : factor ( _factor )
  { }

  template < typename I , typename O >
  void eval ( const I & in , O & out ) const
  {
    out = in * factor ;
  }
} ;

void test ( zimt::bill_t bill )
{
  // some data to process. We use f3_t, a xel of three floats, which
  // represents, e.g., an RGB pixel

  f3_t data [ 1000 ] ;
  f3_t data2 [ 1000 ] ;

  // we'll use this readymade functor to convert input values to
  // output values. It's a multiplication with separate factors for
  // each channle.

  // typedef zimt::xel_t < float , 1 > f1_t ;

  zimt::amplify_type < f3_t > amp ( { 2.0 , 3.0 , 4.0 } ) ;
  amp13_t < float > amp1 ( 2.5 ) ;

  // set up zimt::views to the data

  zimt::view_t < 1 , f3_t >
    v1 { data , { 1000 } , { 1 } } ;

  zimt::view_t < 1 , f3_t >
    v1b { data2 , { 1000 } , { 1 } } ;

  zimt::view_t < 3 , f3_t >
    v3 { data , { 1 , 50 , 500 } , { 50 , 10 , 2 } } ;

  zimt::view_t < 3 , f3_t >
    v3b { data2 , { 1 , 50 , 500 } , { 50 , 10 , 2 } } ;

  // initialize the data to 1.0

  v3.set_data ( 1.0f ) ;
  v3b.set_data ( 1.0f ) ;

  // 'apply' the functor 'amp' to the data viewed by v3

  zimt::apply ( amp , v3 , bill ) ;

  // use 'apply' to apply the functor to the data. Here we don't
  // use the zimt::view but pass initializer sequences to demonstrate
  // that the zimt::view can be created 'on the fly', which is handy
  // when calling zimt from code which doesn't use zimt data types.
  // Note how we have to pass the dimensionality of the views, which
  // can't be determined automatically when using initializer sequences

  zimt::apply < 3 >
    ( amp , { data , { 1 , 50 , 500 } , { 50 , 10 , 2 } } , bill ) ;

  // set up an iterator over all nD coordinates 'in' the view

  zimt::mci_t < 3 > mci ( v3.shape ) ;
  zimt::mci_t < 1 > mci1 ( v1.shape ) ;

  // and use it to check that the result is the same throughout

  for ( int k = 0 ; k < 1000 ; k++ )
  {
    assert ( v3 [ mci [ k ] ] == data [ 0 ] ) ;
  }

  // next, we try an 'index-based transform'. this function feeds
  // the *coordinates* of the values in the view as input, and the
  // functor converts them to output. Again we use both forms:

  zimt::transform ( amp , v3 , bill ) ;

  zimt::transform
    ( amp , { data , { 1 , 50 , 500 } , { 50 , 10 , 2 } } , bill ) ;

  // again we check the result. Here we can compare the content of
  // the view with the result of feeding the coordinate to the functor:

  for ( int k = 0 ; k < 1000 ; k++ )
  {
    assert ( v3 [ mci [ k ] ] == amp ( mci [ k ] ) ) ;
  }

  zimt::transform ( amp1 , v1 ) ;

  for ( int k = 0 ; k < 1000 ; k++ )
  {
    assert ( v1 [ mci1 [ k ] ] == f3_t ( 2.5 * mci1 [ k ] [ 0 ] ) ) ;
  }

  // next, we try a two-view-transform, where the input is taken
  // from the first view, fed to the functor, and then stored to
  // the second view. Again, both forms:

  zimt::transform ( amp , v3 , v3b , bill ) ;

  zimt::transform < 3 >
    ( amp , { data , { 1 , 50 , 500 } , { 50 , 10 , 2 } } ,
            { data2 , { 1 , 50 , 500 } , { 50 , 10 , 2 } } , bill ) ;

  // and the test

  for ( int k = 0 ; k < 1000 ; k++ )
  {
    assert ( data2 [ k ] == amp ( data [ k ] ) ) ;
  }

  zimt::transform ( amp , v1 , v1b , bill ) ;

  for ( int k = 0 ; k < 1000 ; k++ )
  {
    assert ( data2 [ k ] == amp ( data [ k ] ) ) ;
  }

  zimt::array_t < 1 , f3_t > a1l ( 10000 ) ;
  amp13_t < long > amp13 ( 2.0 ) ;
  zimt::transform ( amp13 , a1l ) ;

  zimt::transform ( amp13 , data , 2, 500 ) ;

  // next we try a reduction over an entire view, first using
  // an atomic as the final reduction target:

  zimt::atomic < long > collector ( 0 ) ;

  // odd-shaped array of long as input, initialized to 2.0

  zimt::array_t < 3 , long > a3l ( { 17 , 5 , 31 } ) ;
  a3l.set_data ( 2 ) ;

  // create the sink functor, in this case a 'sum_up' object

  sum_up < long > ci ( collector ) ;

  // pass sink functor and array to 'reduce'

  zimt::apply ( ci , a3l , bill ) ;

  // we know what the result should be, so we can test:

  assert ( collector.load() == long ( 2 * a3l.shape.prod() ) ) ;

  // reduction which sums up multichannel data, adding 'personal
  // scores' to the final reduction target with a 'yield' function.
  // The reduction code is multithreaded, and each thread 'scores'
  // to it's own copy of the 'reductor'. When the thread is done,
  // the reductor is destructed, and the d'tor calls 'yield' to
  // add the 'individual score' to the total score.

  // infrastructure to allow thread-safe access to 'collect'

#ifndef ZIMT_SINGLETHREAD
  std::mutex m ;
#endif
  f3_t collect ( 0 ) ;

  // lambda used to 'yield' the personal score to 'collect'

  auto yield = [&] ( const f3_t & v )
  {
#ifndef ZIMT_SINGLETHREAD
    std::lock_guard < std::mutex > lk ( m ) ;
#endif
    collect += v ;
  } ;

  // odd-shaped array of long as input, initialized to {2.0, 2.0, 2.0}

  zimt::array_t < 3 , f3_t > a3f3 ( { 17 , 5 , 31 } ) ;
  a3f3.set_data ( 2 ) ;
  sum_up_fcpy < float , 3 > clf ( yield ) ;

  // pass sink functor and array to 'apply' - sum_up_fcpy has built-in
  // reduction code, the array won't be affected. The general transform
  // code now uses per-thread copies of the functor, so there is no more
  // zimt::reduce.

  collect = 0 ;

  zimt::apply ( clf , a3f3 ) ;

  // check up on the result, again we know the outcome:

  f3_t expected ( 2.0 * a3f3.shape.prod() ) ;
  assert ( collect == expected ) ;

}

int main ( int argc , char * argv[] )
{
  zimt::bill_t bill ;

  // for testing purposes, we try all sorts of combinations of
  // job count and segment size in the 'loading bill', to make
  // sure that the code is robust.

  for ( int times = 0 ; times < 1000 ; times++ )
  {
    bill.njobs = times % 13 ;
    bill.segment_size = times % 103 ;
    test ( bill ) ;
  }
}
