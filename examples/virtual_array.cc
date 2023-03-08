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

// This example uses a 'virtual array' to 'roll out' SIMD code to
// a large number of values without actually reading from or writing
// to memory - the input is created 'from thin air' with a linspace_t
// object - the same which is used in linspace.cc. Then the functor,
// 'crunch', performs some arithmetic on the input, and, to make things
// a little more interesting, cumulates the result. Finally, rather
// than using a put_t object which stores the output to an array, we
// use a custom put_t object which simply ignores it. What do we get?
// We'll set up a 'virtual array' of 1e9 (a billion) pixels, all with
// different virtual values, process them with multithreaded SIMD code
// and cumulate the result. So we can test 'pure' CPU performance
// without memory access, as if we were processing a large array.
// We can then, e.g, compare the result to a result we might get
// from operating on real arrays and see just how much the memory access
// adds to total execution time.

#include "../zimt.h"

// linspace_t is the 'get_t' we'll use for this example. This functor
// generates the input values to the functor - and for this example,
// where the 'act' functor does nothing but route it's input to it's
// output, it actually does all the 'work'.
// This example may seem trivial, but it should hint at the flexibility
// of minz::process and how to use it. And linspace_t can serve as
// a template for more elaborate 'get_t' classes.

template < typename T ,     // elementary type
           std::size_t N ,  // number of channels
           std::size_t L >  // lane count
struct linspace_t
{
  typedef zimt::xel_t < T , N > value_t ;
  typedef zimt::simdized_type < value_t , L > value_v ;
  typedef typename value_v::value_type value_ele_v ;
  typedef zimt::xel_t < long , N > crd_t ;
  typedef zimt::simdized_type < crd_t , L > crd_v ;
  typedef typename crd_v::value_type crd_ele_v ;

  const std::size_t d ;
  const value_t start ;
  const value_t step ;

  // linspace_t's c'tor receives start, step and axis. Note how
  // start and step are N-dimensional; each component gives the
  // intended value for the corresponding axis.

  linspace_t ( const value_t & _start ,
               const value_t & _step ,
               const std::size_t & _d )
  : start ( _start ) ,
    step ( _step ) ,
    d ( _d )
  { }

  // init is used to initialize the vectorized value to the value
  // it should hold at the beginning of the peeling run. The discrete
  // coordinate 'crd' gives the location of the first value, and this
  // function infers the start value from it. The scalar value will
  // not be used until peeling is done, so it isn't initialized here.

  void init ( value_v & cv , const crd_t & crd )
  {
    cv = step * crd + start ;
    cv [ d ] += value_ele_v::iota() * step [ d ] ;
  }

  // initialize the scalar value from the discrete coordinate.
  // This needs to be done once after peeling, the scalar value
  // is not initialized before.

  void init ( value_t & c , const crd_t & crd )
  {
    c = step * crd + start ;
  }

  // increase modifies it's argument to contain the next value, or
  // next vectorized value, respectively

  void increase ( value_t & trg )
  {
    trg [ d ] += step [ d ] ;
  }

  void increase ( value_v & trg )
  {
    trg [ d ] += ( step [ d ] * L ) ;
  }
} ;

// we use a functor 'crunch' which does a bit of arithmetic.

template < typename T , std::size_t N , std::size_t L >
struct crunch
: public zimt::unary_functor < zimt::xel_t < T , N > ,
                               zimt::xel_t < T , N > ,
                               L >
{
  typedef zimt::xel_t < double , N > score_t ;
  score_t score ;

  std::function < void ( const score_t & ) > yield ;

  crunch ( std::function < void ( const score_t & ) > _yield )
  : yield ( _yield )
  {
    score = 0 ;
  }

  // this is the arithmetic we'll perform:

  template < typename I , typename O >
  void _eval ( const I & i , O & o )
  {
    O help ;
    help[0] = i[1] - i[0] ;
    help[1] = i[2] - i[1] ;
    help[2] = i[0] - i[2] ;
    o[0] = help[0] * help[1] ;
    o[1] = help[1] * help[2] ;
    o[2] = help[2] * help[0] ;
  }

  // because we'll cumulate results, we have two eval overloads,
  // one for 'capped' operation and one for 'uncapped' operation.
  // These two versions only differ in how the 'score' is updated.
  // Note that the score is in double precision - 1e9 values are
  // already large enough to be problematic in SP float.

  template < typename I , typename O >
  void eval ( const I & i , O & o , const std::size_t cap )
  {
    _eval ( i , o ) ;
    for ( int ch = 0 ; ch < N ; ch++ )
    {
      for ( int e = 0 ; e < cap ; e++ )
        score [ ch ] += o [ ch ] [ e ] ;
    }
  }

  template < typename I , typename O >
  void eval ( const I & i , O & o )
  {
    _eval ( i , o ) ;
    for ( int ch = 0 ; ch < N ; ch++ )
      score [ ch ] += o [ ch ] . sum () ;
  }

  // for a reduction, we need a d'tor to trigger adding the per-thread
  // result to the total with the 'yield' function.

  ~crunch()
  {
    yield ( score ) ;
  }
} ;

// for this example, we'll use a custom 'put_t' - on object which
// disposes of the data which the 'act' functor produces in a
// 'non-standard' way. What we'll do here is *not* write these data
// to an array/view at all, but simply discard them. While this may
// seem like just 'wasting cycles', it removes memory access from
// the process and only 'exercises' the CPU, so it's good for
// benchmarking raw CPU throughput. We'll 'roll out' the code to
// a large 'virtual array'.
// Because this put_t doesn't actually 'do' anything, we can omit
// pretty much everything - all we need is init() and save() as
// empty functions, so that the code is formally correct to be
// called - the optimizer will do away with the empty functions.

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


int main ( int argc , char * argv[] )
{
  // we'll use a familiar 'get_t' object - it's the same as in the
  // linspace.cc example. Here, we'll start at zero and use 0.01 as
  // step width in every direction

  typedef zimt::xel_t < float , 3 > delta_t ;
  delta_t start { 0.0 , 0.0 , 0.0 } ;
  delta_t step { .01 , .01 , .01 } ;
  linspace_t < float , 3 , 16 > l ( start , step , 0 ) ;

  // this is our 'act' functor's type

  typedef crunch < float , 3 , 16 > act_t ;

  // here's the 'virtual' array: a view with origin == nullptr - the
  // 'origin' will never actually be accessed, because the put_t we'll
  // use does not touch it. But of course we need a valid shape. The
  // strides aren't used either, so they can be set to zero as well.
  // We chosse an 'odd' shape to make sure as much code as possible
  // gets used and that the 'capping' works.

  zimt::xel_t < std::size_t , 3 > shape { 999 , 1011 , 971 } ;
  zimt::xel_t < long , 3 > strides { 0 , 0 , 0 } ; // unused.

  zimt::view_t < 3 , delta_t > a ( nullptr , strides , shape ) ;

  // this is out put_t which simply discards the results.

  discard_result < float , 3 , 3 , 16 > p ;

  // we need a bit of infrastructure to cumulate the results from
  // per-thread copies of the 'act' functor to 'collect'

#ifndef ZIMT_SINGLETHREAD
  std::mutex m ;
#endif

  typedef zimt::xel_t < double , 3 > score_t ;
  score_t collect ( 0 ) ;

  // lambda used to 'yield' the per-thread score to 'collect'
  // in a thread-safe manner

  auto yield = [&] ( const score_t & v )
  {
#ifndef ZIMT_SINGLETHREAD
    std::lock_guard < std::mutex > lk ( m ) ;
#endif
    collect += v ;
  } ;

  // now we're ready to go!

  minz::process ( act_t ( yield ) , a , zimt::bill_t() , l , p ) ;

  // here's the final result over ca. 1e9 pixels:

  std::cout << "collect: " << collect << std::endl ;

  // if you'd like to arrive at the same result by a simple iteration
  // without any multithreading or SIMD, uncomment this code:

  // typedef zimt::xel_t < float , 3 > f3_t ;
  //
  // collect = 0 ;
  //
  // auto f = act_t ( yield ) ;
  //
  // for ( std::size_t i = 0 ; i < shape[0] ; i++ )
  // {
  //   for ( std::size_t j = 0 ; j < shape[1] ; j++ )
  //   {
  //     for ( std::size_t k = 0 ; k < shape[2] ; k++ )
  //     {
  //       f3_t x { i * .01 , j * .01 , k * .01 } ;
  //       f3_t y ;
  //       f._eval ( x , y ) ;
  //       collect += y ;
  //     }
  //   }
  // }
  //
  // std::cout << "collect: " << collect << std::endl ;
}
