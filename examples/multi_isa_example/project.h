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
    // we declare 'payload' as a *pure virtual member function'. That looks
    // scary, but it makes it impossible to invoke the 'payload' member
    // of struct dispatch itself - only derived classes with an actual
    // implementation can be used to invoke their specific implementation.

    virtual void payload() const = 0 ;
  } ;

  // We have one free function: get_dispatch will yield a dispatch base
  // class pointer.

  dispatch_base * get_dispatch() ;
} ;

#endif // to #ifndef PROJECT_H
