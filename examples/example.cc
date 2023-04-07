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

// Here's some example code which walks through a few of the
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
// ISA, add -march=native on intel/AMD, on other architectures, you can
// use specific flags, e.g -mcpu=apple-m1 for apple's M1 processor.
// For some backends (especially highway), additional ISA-specific flags
// improve performance, e.g for AVX2 add -mavx2 -march=haswell -mpclmul
// -maes. To use the std::simd backend, you'll need a std::simd
// implementation which comes with newer versions of g++ - if that's
// not an option, you can use M. Kretz' original implementation from
// https://github.com/VcDevel/std-simd.

#include "../zimt.h"

// Type for typical xel datum consisting of three float values:

typedef zimt::xel_t < float , 3 > f3_t ;

// simple zimt functor, taking some type T as input and producing
// f3_t as output. Because the operation can use the same code for
// scalar and simdized operation, eval can be a template. Note
// here that zimt uses 'eval' member functions in preference of
// the more conventional operator() overloads. You can add
// 'callability' to a zimt functor by inheriting from a mixin
// with CRTP, see struct callable in unary_functor.h
// The functor is typical for zimt functors: It's set up at
// construction time and remains const threafter. The eval
// member functions are marked const.

template < typename T >
struct amp13_t
: public zimt::unary_functor < T , f3_t , 16 >
{
  const float factor ;

  amp13_t ( const float & _factor )
  : factor ( _factor )
  { }

  template < typename I , typename O >
  void eval ( const I & in , O & out ) const
  {
    out = in * factor ;
  }
} ;

// next we have examples of reduction functors, which serve to
// accumulate - or concentrate - information held in arrays into
// a single value, the 'reduction target'.
// reduction functors need to take heed of the parameter 'cap'
// passed to the vectorized eval to get a correct result - the
// wielding code pads incoming vectors so that the vector is
// safe to handle, but the elements beyond the 'cap' mustn't
// be collected into the reduction result.

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

  // finally, two eval overloads, one for 'uncapped' input and
  // one for 'capped' input. Note how the eval member functions
  // declare but do not produce output. The functor will be
  // used with 'apply' which in turn delegates to transform
  // which expects the functor to have these arguments, but the
  // optimzer will see that the second argument is not used
  // and optimize it away, so having it here is merely a
  // syntactic requirement.

  void eval ( const in_v & v , in_v & )
  {
    sum += v.sum() ;
  }

  void eval ( const in_v & v , in_v & ,
              const std::size_t & cap )
  {
    for ( int e = 0 ; e < cap ; e++ )
      sum += v[e] ;
  }

  // because we're performing a reduction, we need a destructor:
  // The values cumulated in the tread-local copy of this functor
  // have to be communicated to the pooled result in 'collector'.

  ~sum_up()
  {
    collector.fetch_add ( sum ) ;
  }
} ;

// variation using a yield function, which writes to a
// mutex-protected reduction target.

template < typename T , std::size_t N > struct sum_up_fcpy
: public zimt::unary_functor < zimt::xel_t < T , N > >
{
  typedef zimt::xel_t < T , N > dtype ;

  typedef zimt::unary_functor < dtype > base_type ;
  using typename base_type::in_type ;
  using typename base_type::in_v ;
  using base_type::vsize ;

  // reference to a thread-safe function which will serve to
  // deposit the 'personal score' to a pooled final result when
  // the functor is destructed

  std::function < void ( dtype ) > yield ;

  dtype score ;

  // this c'tor is used to create the initial sum_up_fcpy object
  // and fixes the yielding callback

  sum_up_fcpy ( std::function < void ( const dtype & ) > _yield )
  : yield ( _yield )
  {
    score = dtype() ;
  }

  // sum_up_fcpy needs a copy c'tor to propagate the yield function

  sum_up_fcpy ( const sum_up_fcpy & other )
  : yield ( other.yield )
  {
    score = dtype() ;
  }

  // now, two eval overloads, one for scalar input and
  // one for SIMD input. Note the summation: the second overload
  // receives xel_t of N SIMD vectors, so to get a meaningful
  // result with N channels, the SIMD vectors are summed up
  // horizontally and their sums are added to the channels of
  // 'score'.
  // Note how these eval member functions are not const - they
  // modify score, which is a member of the functor. If you try
  // and 'chain' this functor, zimt will (as of this writing)
  // complain and ask for a const functor. To satisfy this demand,
  // you can go via a pointer and allocate the score with new in
  // the c'tor, and delete it in the d'tor. Then the eval functions
  // can be const, because the functor itself is no longer modified.
  // we have two versions for vectorized evaluation, one capped and
  // one uncapped. Here we need to actually use the cap value
  // to get a correct result.

  void eval ( const in_v & v ,
              in_v &  ,
              std::size_t cap )
  {
    for ( int ch = 0 ; ch < N ; ch++ )
    {
      for ( int e = 0 ; e < cap ; e++ )
        score [ ch ] += v [ ch ] [ e ] ;
    }
  }

  void eval ( const in_v & v ,
              in_v & )
  {
    for ( int ch = 0 ; ch < N ; ch++ )
      score [ ch ] += v [ ch ] . sum () ;
  }

  // again, the 'dtor adds the trhead-local result to the total
  // score

  ~sum_up_fcpy()
  {
    yield ( score ) ;
  }
} ;

// This function does the actual test, varying some parameters
// via the 'bill' object which is varied in main().

void test ( zimt::bill_t bill )
{
  // some data to process. We use f3_t, a xel of three floats, which
  // represents, e.g., an RGB pixel

  f3_t data [ 1000 ] ;
  f3_t data2 [ 1000 ] ;

  // we'll use this readymade functor to convert input values to
  // output values. It's a multiplication with separate factors for
  // each channel.

  zimt::amplify_type < f3_t > amp ( { 2.0 , 3.0 , 4.0 } ) ;
  amp13_t < float > amp1 ( 2.5 ) ;

  // set up zimt::views to the data, first two 1D views, then
  // two 3D views. Note how we set up the views with initializer
  // sequences, which is nicely type-neutral. You might as well
  // use xel_t or other fixed-size aggregates like std::array

  zimt::view_t < 1 , f3_t >
    v1 { data , { 1 } , { 1000 } } ;

  zimt::view_t < 1 , f3_t >
    v1b { data2 , { 1 } , { 1000 } } ;

  zimt::view_t < 3 , f3_t >
    v3 { data , { 1 , 50 , 500 } , { 50 , 10 , 2 } } ;

  zimt::view_t < 3 , f3_t >
    v3b { data2 , { 1 , 50 , 500 } , { 50 , 10 , 2 } } ;

  // initialize the data to 1.0
  // set_data expects f3_t, but f3_t does have a c'tor from float,
  // so we can pass a float here

  v3.set_data ( 1.0f ) ;
  v3b.set_data ( 1.0f ) ;

  // 'apply' the functor 'amp' to the data viewed by v3

  zimt::apply ( amp , v3 , bill ) ;

  // use 'apply' to apply the functor to the data. Here we don't
  // use the zimt::view but pass initializer sequences to demonstrate
  // that the zimt::view can be created 'on the fly', which is handy
  // when calling zimt from code which doesn't use zimt data types.
  // Note how we have to pass the dimensionality of the views, which
  // can't be determined automatically when using initializer sequences.

  zimt::apply < 3 >
    ( amp , { data , { 1 , 50 , 500 } , { 50 , 10 , 2 } } , bill ) ;

  // set up an iterator over all nD coordinates 'in' the view

  zimt::mci_t < 3 > mci ( v3.shape ) ;

  // and use it to check that the result is the same throughout

  for ( int k = 0 ; k < 1000 ; k++ )
  {
    assert ( v3 [ mci [ k ] ] == data [ 0 ] ) ;
  }

  // next, we try an 'index-based transform'. this function feeds
  // the *coordinates* of the values in the view as input, and the
  // functor converts them to output. Again we use both forms:

  zimt::transform ( amp , v3 , bill ) ;

  zimt::transform < 3 >
    ( amp , { data , { 1 , 50 , 500 } , { 50 , 10 , 2 } } , bill ) ;

  // again we check the result. Here we can compare the content of
  // the view with the result of feeding the coordinate to the functor:

  for ( int k = 0 ; k < 1000 ; k++ )
  {
    f3_t result ;
    amp.eval ( f3_t ( mci [ k ] ) , result ) ;
    assert ( v3 [ mci [ k ] ] == result ) ;
  }

  zimt::transform ( amp1 , v1 ) ;

  zimt::mci_t < 1 > mci1 ( v1.shape ) ;

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
    f3_t result ;
    amp.eval ( data [ k ] , result ) ;
    assert ( data2 [ k ] == result ) ;
  }

  zimt::transform ( amp , v1 , v1b , bill ) ;

  for ( int k = 0 ; k < 1000 ; k++ )
  {
    f3_t result ;
    amp.eval ( data [ k ] , result ) ;
    assert  ( data2 [ k ] == result ) ;
  }

  // let's try amp13_t

  zimt::array_t < 1 , f3_t > a1l ( 10000 ) ;
  amp13_t < long > amp13 ( 2.0 ) ;
  zimt::transform ( amp13 , a1l ) ;

  // for 1D transforms, there's an overload taking a plain pointer
  // and the stride and length

  zimt::transform ( amp13 , data , 2 , 500 ) ;

  // next we try a reduction over an entire view, first using
  // an atomic as the final reduction target:

  zimt::atomic < long > collector ( 0 ) ;

  // odd-shaped array of long as input, initialized to 2.0

  zimt::array_t < 3 , long > a3l ( { 17 , 5 , 31 } ) ;
  a3l.set_data ( 2 ) ;

  // create the sink functor, in this case a 'sum_up' object

  sum_up < long > ci ( collector ) ;

  // pass sink functor and array to 'apply'

  zimt::apply ( ci , a3l , bill ) ;

  // we know what the result should be, so we can test:

  assert ( collector.load() == long ( 2 * a3l.shape.prod() ) ) ;

  // reduction which sums up multichannel data, adding 'personal
  // scores' to the final reduction target with a 'yield' function.
  // The reduction code is multithreaded, and each thread 'scores'
  // to it's own copy of the 'reductor'. When the thread is done,
  // the reductor is destructed, and the d'tor calls 'yield' to
  // add the 'individual score' to the total score. copies of the
  // 'reductor' which were never used to perform an 'eval' will
  // also yield to 'collect', but their contribution will be zero.

  // infrastructure to allow thread-safe access to 'collect'. If
  // ZIMT_SINGLETHREAD is defined, the mutex protection becomes
  // futile, and, in fact, impossible, because the code is modified
  // so as to not make use of any threading-related code, so that
  // it can compile without pthreads.

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
  // reduction code, the array won't be affected.

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
  // sure that the code is robust. The test succeeds if all of
  // the assertions hold and the program termiates without ouput.

  for ( int times = 0 ; times < 1000 ; times++ )
  {
    bill.njobs = times % 13 ;
    bill.segment_size = times % 103 ;
    test ( bill ) ;
  }
}
