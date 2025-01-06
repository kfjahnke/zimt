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
#include <vector>

// dispatch.h has the class definition of dispatch_base and the
// declaration of 'get_dispatch' which yields the dispatch_base
// pointer at run-time.

#ifndef DISPATCH_BASE
#define DISPATCH_BASE

struct dispatch_base
{
  // the most important member of class dispatch_base is the 'payload'.
  // Here we only have a single payload function, but in a concrete
  // program there might be many. Here, in the base class, we declare
  // the as pure virtual members.

  virtual std::string payload() const = 0 ;

  // further members in dispatch_base will be populated in the derived
  // 'dispatch' classes' c'tors to provide metadata: The main program
  // can glean a dispatch_base pointer and then use this information.
  
  std::size_t hwy_target ;
  std::string hwy_target_name ;
  std::string hwy_target_str ;
} ;

namespace project
{
  // without further specialization, conduit::glean() returns nullptr.
  // specialized with a HWY_TARGET value, it will yield a dispatch_base
  // pointer to a dispatch object in the ISA_specific nested namespace;
  // see also the comments in dispatch.cc.

  template < std::size_t TRG >
  struct conduit
  {
    static dispatch_base * glean ()
    {
      return nullptr ;
    }
  } ;

  // these functions serve to provide access to ISA-specific code.
  // the first one yields a dispatch_base pointer to a dispatch object
  // in the ISA which highway picks via it's dynamic dispatch. This is
  // the dispatch pointer you'll typically use. The second one fills a
  // std::vector of dispatch_base pointers with pointers to available
  // dispatch objects - one for each ISA. You can use this information
  // to inspect what's available or use a different ISA to the one which
  // highway would choose - the dispatch_base pointer also points to
  // 'metadata' like the HWY_TARGET value for the ISA, to help you
  // figure out what it's good for.

  extern const dispatch_base * const get_dispatch() ;
  extern void get_isa_list ( std::vector < const dispatch_base * > & ) ;
} ;

#endif // for #ifndef DISPATCH_BASE
