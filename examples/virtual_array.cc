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

// we use a functor 'crunch' which does a bit of arithmetic.

template < typename T , std::size_t N , std::size_t L >
struct crunch
: public zimt::unary_functor < zimt::xel_t < T , N > ,
                               zimt::xel_t < T , N > ,
                               L >
{
  typedef zimt::unary_functor < zimt::xel_t < T , N > ,
                                zimt::xel_t < T , N > ,
                                L > base_t ;
  using typename base_t::in_v ;
  using typename base_t::out_v ;

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
  // Note also that - at least for now - we can'T code the eval
  // member functions as templates (which is commonly done in
  // vspline) because the signature test fails to detect the
  // capped eval overload if the capped eval is coded as a template.
  // For zimt code this should be less of an issue, because there
  // is no scalar eval overload, so there is no reason not to pass
  // in_v and out_v expressis verbis.

  void eval ( const in_v & i , out_v & o , const std::size_t & cap )
  {
    _eval ( i , o ) ;
    for ( int ch = 0 ; ch < N ; ch++ )
    {
      for ( int e = 0 ; e < cap ; e++ )
        score [ ch ] += o [ ch ] [ e ] ;
    }
  }

  void eval ( const in_v & i , out_v & o )
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
// The implementation is now in wielding.h

int main ( int argc , char * argv[] )
{
  // we'll use a familiar 'get_t' object - it's the same as in the
  // linspace.cc example. Here, we'll start at zero and use 0.01 as
  // step width in every direction

  typedef zimt::xel_t < float , 3 > delta_t ;
  delta_t start { 0.0 , 0.0 , 0.0 } ;
  delta_t step { .01 , .01 , .01 } ;
  zimt::linspace_t < float , 3 , 3 , 16 > l ( start , step , 0 ) ;

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

  auto gk = zimt::grok ( act_t ( yield ) ) ;

  zimt::process ( a.shape , l , gk + gk ,
                  zimt::discard_result() ) ;

  // here's the final result over ca. 1e9 pixels:

  std::cout << "collect: " << collect << std::endl ;
  std::cout << "array size: " << a.size() << std::endl ;

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
