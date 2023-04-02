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

// Another example demonstrating the direct use of zimt's 'wielding'
// code, which allows what I call 'regular' operations, an abstraction
// of array processing which uses the structure of array processing
// but not necessarily arrays as it's substrate: The processing is
// guided by an n-dimensional coordinate block which is traversed
// along parallel 1D 'lines' or 'strands' which are all parallel to
// one axis of the construct. While it would be quite possible to
// use an 'act' functor accepting coordinates and 'taking it from
// there', zimt allows to pass a 'get_t' object which must be able
// to handle a few basic requests (init, increase) and so generates
// values to be used as input by the 'act' functor. This indirect
// method is similar to using a scheme with separated iterator and
// accessor, but because the aim is to produce maximally efficient
// code using SIMD instructions, get_t objects must also be able
// to produce 'simdized' input data for the 'act' functor. This sounds
// complicated, but once the method is understood it becomes clear
// that writing 'get_t' classes reduces the complexity of the task
// at hand, because most of the functionality is handled by the
// 'wielding' code and the get_t object only has a very limited
// and easily-grasped set of responsiblities. Coding these can
// become arbitrarily complex - as complex as the functionality
// demands - but typically it's simple and straightforward. At the
// 'other end' of processing, the output of the 'act' functor is also
// handled by a dedicated object, which disposes of the results the
// 'act' functor produces from it's input. Again this is similar to
// an accessor, which uses the notion of location in the coordinate
// system as it's itertator. The archetypal put_t object simply
// stores the result data in an array, but again a dedicated put_t
// object can do anything with the data - down to ignoring them.
// The design of the wielding code allows to 'do stuff' in different
// ways - it's quite feasible to let the act functor handle most of
// the processing and use very simple - possibly stock - get_t and
// put_t objects. But the act functor is typically unaware of the
// linear processing of 'strands', which are central to the operation
// of the get_t and put_t objects and allow them to do their task
// very efficiently: their 'awareness' of the linear operation
// can be exploited to 'streamline' their code. A get_t object
// operating in the 'notion' of 3D space will only vary in one
// component of the 3D coordinate. An act functor will not have
// such awareness and instead look at the input 'ad hoc' without
// any way of exploiting the linear processing, so to stick with
// the 3D notion, to the act functor, all components are equally
// relevant and it must process all of them adequately. This is
// of course more resource-intensive than the linear processing in
// the get_t object. The disadvantage may be mitigated by the
// optimizer, but it's safer not to rely on that and exploit any
// coding simplifications which arise from linear processing
// explicitly. At the put_t end of processing, the same holds
// tre. It would be feasible to let the act functor handle, e.g.,
// data storage, but, to stick with the 3D example, the act functor
// would need to generate a target address to write data to from
// all three components of the coordinate, whereas a put_t object
// would be 'aware' of it's 'linear context' and therefore able to
// generate target addresses by just looking at the one varying
// component and it's influence on the target address, which typically
// reduces processing time significantly and also avoids more costly
// operations (division, multiplication, modulo) and gets along with
// simple operations like addition.
// Now for the concrete example. Here, we generate input to the
// act functor consisting of continuous 3D coordinates - sets of
// three floating point numbers which can be interpreted as coordinates
// in 3D space. But these coordinates will represent the sampling of
// a 2D manifold, in this example simply a rectangle in space.
// A rectangle in space can be described by its 'anchor point' 'a',
// two 3D vectors - lets call them 'u' and 'v' and two 'limits'
// - let's call them 'm' and 'n', so that all points P in the
// rectangle obey the formula P = a + k * u + l * v with
// k in 0...m and l in 0...n

#include <iomanip>
#include "../zimt.h"

// rect3d_t is the 'get_t' we'll use for this example. This functor
// generates the input values to the functor - and for this example,
// where the 'act' functor does nothing but route it's input to it's
// output, it actually does all the 'work'.
// This example may seem trivial, but it should hint at the flexibility
// of zimt::process and how to use it. And rect3d_t can serve as
// a template for more elaborate 'get_t' classes.
// TODO: this might be abstracted to arbitrary channel counts and
// put into get_t.h - but maybe it's too specific to be part of the
// library.

template < typename T ,     // elementary type
           std::size_t L >  // lane count
struct rect3d_t
{
  typedef zimt::xel_t < T , 3 > value_t ;
  typedef zimt::simdized_type < value_t , L > value_v ;
  typedef typename value_v::value_type value_ele_v ;
  typedef zimt::xel_t < long , 2 > crd_t ;
  typedef zimt::simdized_type < crd_t , L > crd_v ;
  typedef typename crd_v::value_type crd_ele_v ;
  typedef zimt::xel_t < value_t , 2 > step_t ;

  const std::size_t d ; // processing axis - 0 or 1 for a 2D array
  const value_t a ;     // this is the 'a' in the formula above
  const step_t uv ;     // 'u' and 'v' from the formula

  // rect3d_t's c'tor receives a, u, and v. For this example, we'll
  // use 'cardinal' m and n, So k will vary from 0...m-1 and l will
  // vary from 0...n-1. This does not limit generality: we can simply
  // scale u and v so that the size of the construct comes out as
  // intended.

  rect3d_t ( const value_t & _a ,
             const step_t & _uv ,
             const std::size_t & _d )
  : a ( _a ) ,
    uv ( _uv ) ,
    d ( _d )
  { }

  // init is used to initialize the vectorized value to the value
  // it should hold at the beginning of the peeling run. The discrete
  // coordinate 'crd' gives the location of the first value, and this
  // function infers the start value from it. The scalar value will
  // not be used until peeling is done, so it isn't initialized here.
  // Note how crd is 2D, whereas trg is 3D!

  void init ( value_v & trg , const crd_t & crd )
  {
    trg = a + crd[0] * uv[0] + crd[1] * uv[1] ;
    trg [ 0 ] += value_ele_v::iota() * uv [ d ] [ 0 ] ;
    trg [ 1 ] += value_ele_v::iota() * uv [ d ] [ 1 ] ;
    trg [ 2 ] += value_ele_v::iota() * uv [ d ] [ 2 ] ;
  }

  // 'capped' variant. This is only needed if the current segment is
  // so short that no vectors can be formed at all. We fill up the
  // target value with the last valid datum.

  void init ( value_v & trg ,
              const crd_t & crd ,
              std::size_t cap )
  {
    trg = a + crd[0] * uv[0] + crd[1] * uv[1] ;
    for ( std::size_t e = 1 ; e < cap ; e++ )
    {
      trg [ 0 ] [ e ] += T ( e ) * uv [ d ] [ 0 ] ;
      trg [ 1 ] [ e ] += T ( e ) * uv [ d ] [ 1 ] ;
      trg [ 2 ] [ e ] += T ( e ) * uv [ d ] [ 2 ] ;
    }
    trg.stuff ( cap ) ;
  }

  // increase modifies it's argument to contain the next value, or
  // next vectorized value, respectively

  void increase ( value_v & trg )
  {
    trg [ 0 ] += ( uv [ d ] [ 0 ] * L ) ;
    trg [ 1 ] += ( uv [ d ] [ 1 ] * L ) ;
    trg [ 2 ] += ( uv [ d ] [ 2 ] * L ) ;
  }

  // 'capped' variant. This is called after all vectors in the current
  // segment have been processed, so the lanes in trg beyond the cap
  // should hold valid data, and 'stuffing' them with the last datum
  // before the cap is optional.

  void increase ( value_v & trg ,
                  std::size_t cap ,
                  bool _stuff = true )
  {
    for ( std::size_t e = 0 ; e < cap ; e++ )
    {
      trg [ 0 ] [ e ] += ( uv [ d ] [ 0 ] * L ) ;
      trg [ 1 ] [ e ] += ( uv [ d ] [ 1 ] * L ) ;
      trg [ 2 ] [ e ] += ( uv [ d ] [ 2 ] * L ) ;
    }
    if ( _stuff )
    {
      trg.stuff ( cap ) ;
    }
  }
} ;

int main ( int argc , char * argv[] )
{
  std::cout << std::fixed << std::showpoint
            << std::setprecision(1) ;

  // let's start with a simple 1D linspace.

  zimt::bill_t bill ;

  typedef rect3d_t < float , 4 > rect_t ;
  typedef typename rect_t::value_t value_t ;
  typedef typename rect_t::step_t step_t ;

  // we'll keep the example very simple: the origin of the
  // rectangle will be at (0, 0, 0), the 'u' and 'v' vectors
  // will only have one none-zero component each:

  value_t start = 0 ;
  value_t u { 0.1 , 0.0 , 0.0 } , v { 0.0 , 0.0 , 0.2 } ;
  step_t step { u , v } ;

  rect_t l ( start , step , 0 ) ;
  typedef zimt::pass_through < float , 3 , 4 > act_t ;
  zimt::array_t < 2 , value_t > a ( { 17 , 15 } ) ;
  zimt::norm_put_t < act_t , 2 > p ( a , 0 ) ;

  zimt::process < act_t , 2 > ( act_t() , a , l , p ) ;

  for ( std::size_t y = 0 ; y < 15 ; y++ )
  {
    for ( std::size_t x = 0 ; x < 17 ; x++ )
      std::cout << " " << a [ { x , y } ] ;
    std::cout << std::endl ;
  }
  std::cout << std::endl ;

  // repeat the process with axis 1 - the order of execution
  // is different, but the result is the same.

  rect_t l1 ( start , step , 1 ) ;
  zimt::norm_put_t < act_t , 2 > p1 ( a , 1 ) ;
  bill.axis = 1 ;

  zimt::process < act_t , 2 > ( act_t() , a , l1 , p1 , bill ) ;

  for ( std::size_t y = 0 ; y < 15 ; y++ )
  {
    for ( std::size_t x = 0 ; x < 17 ; x++ )
      std::cout << " " << a [ { x , y } ] ;
    std::cout << std::endl ;
  }
  std::cout << std::endl ;
 }
