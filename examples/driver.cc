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
// or with a single ISA-specific set of compiler flags. The definition
// of 'payload' takes the code which was in main() in the previous set
// of examples.

// #include "../zimt/common.h"
// #include "../zimt/simd/simd_tag.h"
// #include <hwy/highway.h>
// #include <iostream>

// in namespace 'project' (set up in the payload TU, e.g. linspace.cc)
// we have two functions which we may call from here. The first one
// yields a dispatch pointer, via which we can invoke member functions
// of the dispatcher, which are introduced with ZIMT_REGISTER in
// interface.h - currently only a dummy function to test the mechanism.
// The second is the actual payload code, which 'does something'.

// namespace project
// {
//   struct dispatch_base
//   {
//     // in dispatch_base and derived classes, we keep two flags.
//     // 'backend' holds a value indicating which of zimt's back-end
//     // libraries is used. 'hwy_isa' is only set when the highway
//     // backend is used and holds highway's HWY_TARGET value for
//     // the given nested namespace.
// 
//     zimt::backend_e backend = zimt::NBACKENDS ;
//     unsigned long hwy_isa = 0 ;
// 
//     // next we have pure virtual member function definitions for
//     // payload code. In this example, we only have one payload
//     // function. This 'driver' is used for example code and we
//     // simply move what was 'main' in examples not using the
//     // multi-SIMD-ISA mechanism to 'payload'. This results in
//     // code which doesn't need to be re-compiled for several
//     // ISAs being re-compiled several times, but this only
//     // produces a slight overhead, if it isn't optimized away
//     // anyway.
// 
//     virtual int payload ( int argc , char * argv[] ) const = 0 ;
//   } ;
// }

int main ( int argc , char * argv[] )
{
  // Here we use zimt's dispatch mechanism: first, we get a pointer
  // to the dispatcher, then we invoke a member function of the
  // dispatcher. What's the point? We can call a SIMD-ISA-specific
  // bit of code without having to concern ourselves with figuring
  // out which SIMD ISA to use on the current CPU: this happens via
  // highway's dispatch mechanism, or is fixed at compile time, but
  // in any case we receive a dispatch_base pointer routing to the
  // concrete variant. project::get_dispatch might even be coded
  // to provide pointers to dispatch objects in separate TUs, e.g.
  // when these TUs use different back-ends or compiler flags. Here,
  // we can remain unaware of how the concrete dispatch object is
  // set up and the pointer obtained.

  auto dp = project::get_dispatch() ;

  // we can get information about the specific dispatch object:

  std::cout << "obtained dispatch pointer " << dp << std::endl ;
  std::cout << "dispatching to back-end   "
            << zimt::backend_name [ dp->backend ] << std::endl ;
#if defined USE_HWY || defined MULTI_SIMD_ISA
  std::cout << "dispatch hwy_isa is       "
            << hwy::TargetName ( dp->hwy_isa ) << std::endl ;
#endif

  // now we call the payload via the dispatch_base pointer.

  int success = dp->payload ( argc , argv ) ;
  std::cout << "payload returned " << success << std::endl ;
}
