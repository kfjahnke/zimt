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
// extern_get_t.cpp. This example demonstrates that 'grokking' a get_t
// object produces an abject which can be passed from one TU to another
// without the need of having the source code of the 'grokked' object
// visible by the code obtaining the 'grokked' object.
// this file has the definition of a get_t class which yields twice
// the 'notional' coordinate - not very useful, but it's simple and
// demonstrates what's going on. Compile this file to an object file,
// like so:
// clang++ -c -o fetches_get_t.o fetches_get_t.cpp
// Do the same for extern_get_t.cc, then link the two object files,
// like so:
// clang++ fetches_get_t.o extern_get_t.o
// The resulting program (a.out) will pass the 'grokked' get_t object
// from one TU to the other and use it in main, printing out the result.

#include <zimt/zimt.h>

// declaration of 'fetch_get_t' in the other TU. Note how this here TU
// doesn't know anything of the type of the 'grokkee' itself.

extern zimt::grok_get_t < int , 2 , 2 , 16 > fetch_get_t() ;

int main ( int argc , char * argv[] )
{
  // fetch the 'grokked' get_t bject from the other TU

  auto gt = fetch_get_t() ;

  // now we'll put it to use:

  typedef zimt::xel_t < int , 2 > dtype ;
  zimt::array_t < 2 , dtype > crda ( { 3 , 4 } ) ;

  zimt::process < 2 > ( { 3 , 4 } ,
                        gt ,
                        zimt::pass_through < int , 2 , 16 > () ,
                        zimt::storer < int , 2 , 2 , 16 > ( crda ) ) ;

  // let's see the result

  for ( int y = 0 ; y < 4 ; y++ )
  {
    for ( int x = 0 ; x < 3 ; x++ )
      std::cout << "  " << crda [ { x , y } ] ;
    std::cout << std::endl ;
  }
}
