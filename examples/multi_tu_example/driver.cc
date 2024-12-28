// Again, we use a separate 'driver ' program, where we put 'main'.
// The definition of class dispatch_base is now in a separate header:

#include "dispatch.h"

int main ( int argc , char * argv[] )
{
  // obtain a dispatch_base pointer to the ISA-specific code.
  // get_dispatch is in dispatch.cc

  auto dp = project::get_dispatch() ;

  // call the payload function via the dispatch_base pointer

  int success = dp->payload ( argc , argv ) ;

  return success ;
}
