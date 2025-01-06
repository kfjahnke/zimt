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

// example of an 'external' payload - one which isn't managed by highway.
// Such payloads can be configured arbitrarily - here we merely provide
// a specific payload routine and a customized dispatch c'tor, and we'll
// access these from driver.cc when we 'walk through' the ISAs.

#include <iostream>
#include "dispatch.h"

namespace project
{
  namespace epl_test
  {
    struct dispatch
    : public dispatch_base
      {
        std::string payload() const
        {
          std::string echo = "hello from payload in epl_test" ;
          return echo ;
        }

        dispatch()
        {
          // we use a very high value for hwy_target here, but not
          // a power of two. The idea is to provide a value which
          // allows the main program to figure out whether this
          // variant will execute on the current CPU. see driver.cc

          hwy_target = 0x8000000000000001 ;
          hwy_target_name = "epl_test: not a highway target!" ;
        }

      } ;

    const dispatch_base * const _get_dispatch()
    {
      static dispatch d ;
      return &d ;
    }
  } ;
} ;
