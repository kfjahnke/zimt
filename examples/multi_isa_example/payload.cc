// we want to echo stuff to the console

#include <iostream>

// here we include the new header 'project.h'

#include "project.h"

// we start out with a bit of 'scaffolding' code to use highway's 
// oreach_target mechanism

// 'clear' HWY_TARGET_INCLUDE is it's set already

#undef HWY_TARGET_INCLUDE

// then set it to the name of this very file

#define HWY_TARGET_INCLUDE "payload.cc"

// the compiler now 'knows' which file we want to 'submit' to the
// foreach_target mechanism and we can #include the 'magic header'

#include <hwy/foreach_target.h>

// next we include highway.h, which gives us access to highway
// library code.

#include <hwy/highway.h>

// that's it for scaffolding. The foreach_target mechanism works by
// repeatedly #including HWY_TARGET_INCLUDE, each time using different
// compiler settings, specific to each supported SIMD ISA. It sets the
// macro 'HWY_NAMEPSACE', so we now switch to the SIMD-ISA-specific
// code by continuing in it's own specific nested namespace:

namespace project
{
  namespace HWY_NAMESPACE
  {
    // here we are inside the SIMD-ISA-specific namespace, but we
    // dont' want to know *which specific one* that might be - we
    // keep the code general.

    // Before we continue as before, let's define a concrete class
    // derived from the 'dispatch_base' class we have in 'project.h'.
    // It's important that the derived class is in the SIMD-ISA-specific
    // namespace!

    struct dispatch
    : public project::dispatch_base
    {
      // here, we provide an override of the pure virtual member
      // function in the base class with a concrete implementation.
      // Note the HWY_ATTR prefix - highway needs this to make the
      // compiler compile the code correctly with SIMD capability.

      HWY_ATTR void payload() const
      {
        // let's do something ISA-specific
      
        std::cout << "hello from linked-in payload for target "
                  << hwy::TargetName ( HWY_TARGET )
                  << std::endl ;
      }
    } ;

    // We still use highway's dispatch mechanism - but now we don't
    // HWY_EXPORT the payload function any more - we'll access it via
    // the dispatch class. The only function we submit to HWY_EXPORT
    // for now is the _get_disptach function. This function returns
    // a pointer to an object of class project::dispatch_base - the
    // base class. But the return statement inside the function takes
    // a dispatch object local to the current nested namespace. C++
    // semantics automatically casts that to the desired base class
    // pointer, which is returned to the caller.
    
    project::dispatch_base * _get_dispatch()
    {
      static dispatch d ;
      return &d ;
    }
  } ;
    
  #if HWY_ONCE

  // now we tell highway that we will want to dispatch to the SIMD-ISA-
  // specific variant of _get_dispatch

  HWY_EXPORT ( _get_dispatch ) ;

  // here comes the definition of the function 'get_dispatch', now 'one
  // level up', in namespace 'project' without further qualifiers. This 
  // is meant 'for export', just as it's declared and expected in
  // 'driver.cc'. Driver.cc can now acquire a dispatch base class pointer,
  // and calling the virtual member function through it will invoke the
  // SIMD-ISA-specific variants.
  
  project::dispatch_base * get_dispatch()
  {
    return HWY_DYNAMIC_DISPATCH(_get_dispatch) () ;
  }

  #endif
}
