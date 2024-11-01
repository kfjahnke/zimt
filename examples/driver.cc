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

// To handle code using several SIMD ISAs in one binary (like with
// highway's foreach_target mechanism) I prefer to split the examples
// in two: this TU, the driver, is compiled without any ISA-specific
// flags and contains 'main' which in turn calls a plain old function
// in the 'payload' TU which does the dispatching. The payload TU can
// be compiled in different ways - be it with highway's foreach_target,
// or with a single ISA-specific set of compiler flags.

#define ZIMT_REGISTER_LIST "zimt_register_list_example.h"

#include "../zimt/common.h"
#include <iostream>

// in namespace 'project' (set up in the payload TU, e.g. linspace.cc)
// we have two functions which we may call from here. The first one
// yields a dispatch pointer, via which we can invoke member functions
// of the dispatcher, which are introduced with ZIMT_REGISTER in
// interface.h - currently only a dummy function to test the mechanism.
// The second is the actual payload code, which 'does something'.

namespace project
{
  const zimt::dispatch * const get_dispatch() ;
  int payload ( int argc , char * argv[] ) ;
}

int main ( int argc , char * argv[] )
{
  // Here we use zimt's dispatch mechanism: first, we get a pointer
  // to the dispatcher, then we invoke a member function of the
  // dispatcher - in this case, just the dummy function we've set
  // up to test the mechanism. What's the point? We can call
  // a SIMD-ISA-specific bit of code without having to concern
  // ourselves with figuring out which SIMD ISA to use on the current
  // CPU: this happens via highway's dispatch mechanism, or is fixed
  // at compile time, but in any case we receive a dispatcher pointer
  // routing to the concrete variant.

  auto dp = project::get_dispatch() ;
  std::cout << "dp = " << dp << std::endl ;
  int trg = dp->dummy ( 5.0f ) ;
  std::cout << "get_dispatch returned " << trg << std::endl ;

  int success = project::payload ( argc , argv ) ;
  std::cout << "payload returned " << success << std::endl ;
}
