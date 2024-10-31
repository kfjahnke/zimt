#include "project.h"

int main ( int argc , char * argv[] )
{
  // Now we'll not call the payload function, but get_dispatch
  
  auto * dp = project::get_dispatch() ;

  // If we now call 'payload' - via dp - we get the ISA-specific version:
  
  dp->payload() ;
}
