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

// dispatch.cc uses highway's foreach_target mechanism to declare
// functionality in object code made from 'payload.cc'. This is the
// main difference to the multi_isa example: there, we actually
// produce code with foreach_target, here, just declarations, and
// the definitions in separate object files.

// dispatch.h declares class dispatch_base

#include "dispatch.h"

// we use foreach_target with the usual scaffolding code

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "dispatch.cc" 

#include <hwy/foreach_target.h> 
#include <hwy/highway.h>

HWY_BEFORE_NAMESPACE() ;

namespace project
{
  namespace HWY_NAMESPACE
  {
    // here, we're inside the ISA-specific nested namespace. We declare
    // _get_dispatch. The function definition for that resides in the
    // ISA-specific TU.  

    const dispatch_base * const _get_dispatch() ;
  } ;

  // Now we'll use a bit of trickery to 'extract' the various dispatch
  // options from what the foreach_target mechanism has set up.
  // I haven't found a simple way to directly gain access to this
  // information in a platform-independent way with highway, but
  // the method I use here is universally applicable, if at a slight
  // cost of code space.

  // without further specialization, conduit::glean() returns nullptr.
  // the conduit template is defined in dispatch.h, see there.

  // when conduit is specialized with a target, glean() returns
  // a dispatch_base pointer pointing to an object of the derived
  // class 'dispatch'. We'll use this mechanism to provide our program
  // with a list of dispatch_base pointers to all the ISA-specific
  // dispatch objects occuring in the program - see the code further
  // down; the code is in namespace project after HWY_ONCE.

  template<> struct conduit < HWY_TARGET >
  {
    static const dispatch_base * glean ()
    {
      return HWY_NAMESPACE::_get_dispatch() ;
    }
  } ;

} ;

#if HWY_ONCE

namespace project
{
  // we use a recursive template to fill 'isa_list': The caller
  // calls store_active specialized with the highest possible
  // target. If that target has a specialization of 'conduit',
  // that will return a dispatch_base pointer when it's 'glean'
  // member function is called. If so, we store the result in
  // 'isa_list'. Then we right-shift the target (to the next
  // lower target value) and call store_active specialized to
  // this lower value. The test for TRG==0 ends the recursion
  // - all gleaned dispatch_base pointers are now in 'isa_list'.
  // We 'load' the dispatch object with more information (like,
  // HWY_TARGET, the target's name etc.), so the one pointer is
  // enough for code needing dispatch.

  template <std::size_t TRG> void store_active
    ( std::vector < const dispatch_base * > & isa_list )
  {
    auto const * gleaned = conduit < TRG > :: glean() ;
    if ( gleaned )
      isa_list.push_back ( gleaned ) ;

    if constexpr ( TRG != 0UL )
      store_active < ( TRG >> 1 ) > ( isa_list ) ;
  }

  // we use highway's HWY_DYNAMIC_DISPATCH to automatically pick the
  // best version of _get_dispatch, which will return a dispatch_base
  // pointer pointing to an ISA-specfic dispatch object. We'll call
  // get_dispatch from the main program.

  HWY_EXPORT ( _get_dispatch ) ;

  const dispatch_base * const get_dispatch()
  {
    return HWY_DYNAMIC_DISPATCH(_get_dispatch) () ;
  }

  // here's our function to collect dispatch_base pointers to dispatch
  // objects declared with foreach_target.

  void get_isa_list ( std::vector < const dispatch_base* > & isa_list )
  {
    isa_list.clear() ;
    store_active < 0x8000000000000000 > ( isa_list ) ;
  }

} ;

#endif

HWY_AFTER_NAMESPACE() ;
