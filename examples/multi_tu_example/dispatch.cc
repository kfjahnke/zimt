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
    // dispatch::payload, but it will be only defined in a separate
    // ISA-specfic TU. We also declare the 'local' _get_dispatch. The
    // function definition for that will also be in the ISA-specific TU.  

    struct dispatch
    : public dispatch_base
    {
      int payload ( int argc , char * argv[] ) const ;
    } ;

    const dispatch_base * const _get_dispatch() ;
  } ;

#if HWY_ONCE

  // we use highway's HWY_DYNAMIC_DISPATCH to automatically pick the
  // best version of _get_dispatch, which will return a dispatch_base
  // pointer pointing to an ISA-specfic dispatch object. We'll call
  // get_dispatch from the main program.

  HWY_EXPORT ( _get_dispatch ) ;

  const dispatch_base * const get_dispatch()
  {
    return HWY_DYNAMIC_DISPATCH(_get_dispatch) () ;
  }

#endif

} ;

HWY_AFTER_NAMESPACE() ;
