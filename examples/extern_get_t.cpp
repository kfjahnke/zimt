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

// This is one part of a two-part example, going together with
// fetches_get_t.cpp. This example demonstrates that 'grokking' a get_t
// object produces an abject which can be passed from one TU to another
// without the need of having the source code of the 'grokked' object
// visible by the code obtaining the 'grokked' object.
// this file has the definition of a get_t class which yields twice
// the 'notional' coordinate - not very useful, but it's simple and
// demonstrates what's going on. Compile this file to an object file,
// like so:
// clang++ -c -o extern_get_t.o extern_get_t.cpp
// Do the same for fetches_get_t.cc, then link the two object files.
// The resulting program will pass the 'grokked' get_t object from
// one TU to the other and use it in main, printing out the result.

#include <zimt/zimt.h>

// definition of the get_t class used for this example. note how this
// definition is local to this TU and not visible in fetch_get_t.cpp
// The definition is the same as zimt::get_crd, the only difference
// is the doubling of the coordinates which are generated.

template < typename T ,    // elementary/fundamental type
           std::size_t N , // number of channels
           std::size_t D , // dimension of the view/array
           std::size_t L = zimt::vector_traits < T > :: vsize >
struct get_twice_crd_t
{
  typedef zimt::xel_t < T , N > value_t ;
  typedef zimt::simdized_type < value_t , L > value_v ;
  typedef typename value_v::value_type value_ele_v ;
  typedef zimt::xel_t < long , D > crd_t ;

  const std::size_t d ;

  // get_crd's c'tor receives the processing axis

  get_twice_crd_t ( const std::size_t & _d = 0 )
  : d ( _d )
  { }

  void init ( value_v & trg , const crd_t & crd ) const
  {
    trg = 2 * crd ;
    trg [ d ] += 2 * ( value_ele_v::iota() ) ;
  }

  void init ( value_v & trg ,
              const crd_t & crd ,
              std::size_t cap ) const
  {
    init ( trg , crd ) ;
    trg.stuff ( cap ) ;
  }

  void increase ( value_v & trg ) const
  {
    trg [ d ] += 2 * L ;
  }

  void increase ( value_v & trg ,
                  std::size_t cap ,
                  bool _stuff = true ) const
  {
    auto mask = ( value_ele_v::IndexesFromZero() < int(cap) ) ;
    trg [ d ] ( mask ) += T ( 2 * L ) ;
    if ( _stuff )
    {
      trg [ d ] ( ! mask ) = trg [ d ] [ cap - 1 ] ;
    }
  }
} ;

// now we pick one concrete instatiation of this template and write
// a function to pass it to the other TU with a grok_get_t object.
// Note that the grok_get_t has a c'tor template which accepts
// compatible get_t; the return statement 'groks' the 'grokkee'.

zimt::grok_get_t < int , 2 , 2 , 16 > fetch_get_t()
{
  return get_twice_crd_t < int , 2 , 2 , 16 > () ;
}
