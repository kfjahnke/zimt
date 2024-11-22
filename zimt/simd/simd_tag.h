/************************************************************************/
/*                                                                      */
/*    zimt - abstraction layer for SIMD programming                     */
/*                                                                      */
/*            Copyright 2023 by Kay F. Jahnke                           */
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

/*! \file simd_tag.h

    \brief tag inherited by all SIMD data types

*/

// Note: initially I used a separate namespace 'simd', but now I've
// decided to keep everything in namespace zimt, especially since the
// use of highway's foreach_target mechanism introduces several nested
// namespaces which further complicate matters, so now I stick with
// zimt and the nested HWY_NAMESPACE namespaces for SIMD-ISA-specific
// code, which correspond with highway's namespace scheme.

#ifndef SIMD_TAG_H
#define SIMD_TAG_H

#include <string>

namespace zimt
{
  // to mark all variations of SIMD data types, we'll derive them
  // from simd_tag, and therefore also from simd_flag.

  class simd_flag { } ;

  // So far, we have four backends. Note that the use of highway's
  // foreach_target mechanism works only with the GOADING and HWY
  // backends and requires #defining MULTI_SIMD_ISA. Without this
  // definition, the zimt code will assume a specific target SIMD
  // ISA which depends on compiler flags at compile time, and in
  // this mode of compilation, the other two backends can be used
  // as well. The problem with the latter two backends is that I
  // haven't found a way to re-compile them using the foreach_target
  // mechanism - their code seems to be written so as to assume a
  // single SIMD ISA and I didn't mange to even patch is so as to
  // fit in with a multiple-reinclusion scheme.

  enum backend_e { GOADING , VC , HWY , STDSIMD , NBACKENDS } ;

  // For diagnostic output:

  const std::string backend_name[] { "GOADING" ,
                                     "Vc" ,
                                     "highway" ,
                                     "std::simd" ,
                                     "unknown" } ;

  // now we can code the tag:

  template < typename T , std::size_t N , backend_e B >
  struct simd_tag
  : public simd_flag
  {
    typedef T value_type ;
    static const std::size_t vsize = N ;
    static const backend_e backend = B ;
  } ;

} ;

#endif
