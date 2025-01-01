/************************************************************************/
/*                                                                      */
/*    zimt - abstraction layer for SIMD programming                     */
/*                                                                      */
/*            Copyright 2025 by Kay F. Jahnke                           */
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

// driver.cc is has the 'main' function for the test program multi_tu.

#include <iostream>

// Again, we use a separate 'driver ' program, where we put 'main'.
// The definition of class dispatch_base is now in a separate header:

#include "dispatch.h"

namespace project
{
  // we use two object files which aren't 'covered' by highway's
  // foreach_target mechanism. The first one is a 'regular' highway
  // target, but it's not picked by foreach_target automatically.
  // We trigger it's production by inclusion of the EMU128 target
  // in CMakeLists.txt. The second object file is to demonstrate
  // the addition of custom-made object files which may or may not
  // use highway. The epl_test object doesn't use highway and is
  // made from a different source file (e_payload.cc)

  namespace N_EMU128
  {
    const dispatch_base * const _get_dispatch() ;
  } ;

  namespace epl_test
  {
    const dispatch_base * const _get_dispatch() ;
  } ;
} ;

int main ( int argc , char * argv[] )
{
  // obtain a dispatch_base pointer to the ISA-specific code which
  // highway would pick via it's dynamic dispatch. get_dispatch is
  // in dispatch.cc

  auto dp = project::get_dispatch() ;

  // call the payload function via the dispatch_base pointer

  std::cout << "dynamic dispatch: " << dp->payload() << std::endl ;
  std::cout << std::endl ;

  // now we'll get access to all dispatch options we have in this
  // program and try them out. First, we'll collect dispatch_base
  // pointers to all the ISA-specific and additional variants of
  // payload code in a vector:

  std::vector < const dispatch_base * > isa_list ;

  // get_isa_list 'walks through' the variants which highway uses
  // with it's foreach_target mechanism

  project::get_isa_list ( isa_list ) ;

  // we have added two extra variants - one is a regular highway
  // target, but not normally added by foreach_target, and one is
  // made without highway to show that we can add arbitrary variants
  // as long as they can provide a dispatch_base pointer to give
  // access to their payload code.

  isa_list.push_back ( project::N_EMU128::_get_dispatch() ) ;
  isa_list.push_back ( project::epl_test::_get_dispatch() ) ;

  // we try all dispatches. If the 'better' ISAs (better than what's
  // available on the executing CPU) actually had ISA-specific
  // instructions, they would fail (illegal instruction), but since
  // we're only producing a std:string in the payload routines, the
  // full traversal is safe.

  std::cout << "inventory of dispatch options:" << std::endl ;

  for ( auto * p_dsp : isa_list )
  {
    std::cout << "found dispatch pointer "
              << p_dsp << std::endl ;
    std::cout << "  hwy_target           "
              << p_dsp->hwy_target << std::endl ;
    std::cout << "  hwy_target_name      "
              << p_dsp->hwy_target_name << std::endl ;
    std::cout << "  hwy_target_str       "
              << p_dsp->hwy_target_str << std::endl ;

    // we use highway's numeric scheme for HWY_TARGET and emit a
    // warning here if the gleaned target has a hwy_target value
    // below the value produced by dynamic dispatch. This indicates
    // an ISA which the current CPU can't handle. In this example
    // it's safe to call the payload, because our payloads don't
    // use ISA-specific instructions - if they did, calling such
    // a payload would result in an illegal instruction error.

    if ( p_dsp->hwy_target < dp->hwy_target )
    {
      std::cout << "warning: compiled for an ISA which this CPU can't handle"
                << std::endl ;
    }
    std::cout << "testing payload: "
              << p_dsp->payload() << std::endl ;

    std::cout << std::endl ;
  }
  return 0 ;
}
