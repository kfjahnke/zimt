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

#include <iostream>
#include "dispatch.h"

// This header defines all the macros having to do with targets:

#include <hwy/detect_targets.h>

// glean the target as 'TG_ISA' from outside - this file is intended
// to produce ISA-specific separate TUs containing only binary for
// one given ISA, but assuming that other files of similar structure,
// but for different ISAs will also be made and all linked together
// with more code which actually makes use of the single-ISA TUs.
// 'Slotting in' the target ISA from the build system is enough to
// produce a SIMD-ISA-specific TU - all the needed specifics are
// derived from this single information. detect_targets.h sets
// HWY_TARGET to HWY_STATIC_TARGET, so we #undef it and use the
// target specification from outside instead.

#undef HWY_TARGET
#define HWY_TARGET TG_ISA

// now we #include highway.h - as we would do after foreach_target.h
// in a multi-ISA build. With foreach_target.h, the code is re-included
// several times, each time with a different ISA. Here we have set one
// specific ISA and there won't be any re-includes.

#include <hwy/highway.h>

HWY_BEFORE_NAMESPACE() ;

// now we're inside the #pragma-defined ISA-specific compilation
// environment - just as if we'd passed ISA-specific compiler flags.
// We could now #include headers, or define templates, which would
// all 'fully' cooperate with the payload code, because they would
// share the same environment.

namespace project
{
  namespace HWY_NAMESPACE
  {
    // now we finally define dispatch::payload and _get_dispatch.
    // The definitions are happening in the ISA-specific environment
    // (set up with HWY_BEFORE_NAMESPACE), wich is what we need, once
    // for each in-play ISA.

    struct dispatch
    : public dispatch_base
    {
      std::string payload() const
      {
        // finally, the 'payload code' itself. we echo the name
        // of the current target ISA:

        std::string echo = "call to payload in " ;
        echo += hwy::TargetName ( HWY_TARGET ) ;
        return echo ;
      }

      // we add a c'tor for class 'dispatch' where we set ISA-specific
      // member variables inside the dispatch object, which were declared
      // as members of dispatch_base. Because class dispatch inherits from
      // class dispatch_base, these ISA-specific values will be found via
      // a dispatch_base pointer. With this information we can limit the
      // 'gleaning' process to the provision of a dispatch_base pointer
      // and then extract it's 'metadata' via these member variables.
      // HWY_TARGET_STR isn't #defined in every compilation of payload.cc,
      // so we only set hwy_target_str where it's #defined.

      dispatch()
      {
        hwy_target = HWY_TARGET ;
        hwy_target_name = hwy::TargetName ( HWY_TARGET ) ;
#ifdef HWY_TARGET_STR
        hwy_target_str = HWY_TARGET_STR ;
#endif
      }

    } ;

    // _get_dispatch returns a dispatch_base pointer (pointer to base class)
    // to the dispatch object specific to this nested namespace.

    const dispatch_base * const _get_dispatch()
    {
      static dispatch d ;
      return &d ;
    }
  } ;
} ;

HWY_AFTER_NAMESPACE() ;
