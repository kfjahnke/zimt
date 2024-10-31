// We use a 'sentinel' to make sure a TU 'sees' this code precisely once

#ifndef PROJECT_H
#define PROJECT_H

namespace project
{
  // We want a 'dispatch' object with - for now - one member function.
  // This is the 'payload' function we had as a free function - now
  // it's a pure virtual member function of class dispatch_base.

  struct dispatch_base
  {
    // we define the macro 'SIMD_REGISTER' to generate pure virtual
    // meber function declarations. The invocations of this macro are
    // taken from 'interface.h'.

    #define SIMD_REGISTER(RET,NAME,...) \
      virtual RET NAME ( __VA_ARGS__ ) const = 0 ;

    #include "interface.h"

    #undef SIMD_REGISTER
  } ;

  // We have one free function: get_dispatch will yield a dispatch base
  // class pointer.

  dispatch_base * get_dispatch() ;
} ;

#endif // to #ifndef PROJECT_H
