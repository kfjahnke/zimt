// dispatch.h has the class definition of dispatch_base and the
// declaration of 'get_dispatch' which yields the dispatch_base
// pointer at run-time.

#ifndef DISPATCH_BASE
#define DISPATCH_BASE

struct dispatch_base
{
  virtual int payload ( int argc , char * argv[] ) const = 0 ;
} ;

// get_dispatch will yield a dispatch_base pointer to the ISA-specific
// payload code best suited for the CPU currently running the code.
// The definition of this function is in 'dispatch.cc'.

namespace project
{
  extern const dispatch_base * const get_dispatch() ;
} ;

#endif // for #ifndef DISPATCH_BASE
