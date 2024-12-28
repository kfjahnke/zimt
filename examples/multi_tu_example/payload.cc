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
        int payload ( int argc , char * argv[] ) const
        {
          // finally, the 'payload code' itself. Just to show that
          // all our efforst have put us in the right environment,
          // we echo the name of the current target ISA:

          std::cout << "paylod: target = "
                    << hwy::TargetName ( HWY_TARGET )
                    << std::endl ;
          return 0 ;
        }
      } ;

    const dispatch_base * const _get_dispatch()
    {
      static dispatch d ;
      return &d ;
    }
  } ;
} ;

HWY_AFTER_NAMESPACE() ;
