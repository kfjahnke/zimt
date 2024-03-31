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

/*! \file simd.h

    \brief provides the concrete SIMD data types

    This header includes the SIMD back-end headers and makes them
    available to zimt. Which back-end types are used depends on
    preprocessor #defines and the idea is to settle for one of the
    back-ends for the program. User code is also free to use the
    back-end headers in the simd folder without 'mediation' by this
    header, and to mix them as desired ('cherrypicking').

*/

#ifndef ZIMT_SIMD_H // sentinel

namespace simd
{
template < typename T >
struct allocator_traits
{
  typedef std::allocator < T > type ;
} ;
} ;

// we have two back-ends which can provide data types for the entire
// range zimt intends to cover: arbitrary fundamentals and any number
// of lanes. One of these is 'zimt's own' - a simple container type
// with arithmetic capability and SIMD-typical member functions. This
// type does not use any explicit SIMD instructions but instead relies
// on autovectorization. The other is based on the fixed-size vectors
// available from std::simd. This type does provide 'genuine' SIMD code
// for a growing amount of functionality. It requires C++17.

#if defined USE_STDSIMD

// std::simd provides a 'complete' SIMD type template accepting all
// fundamentals and lane counts. So we use the type derived from the
// fixed-size std::simd vector (std_simd_type) instead of gen_simd_type

#include "simd/std_simd_type.h"

#else

// For the goading implementation and the Vc- and highway-based code,
// we need the goading implementation of gen_simd_type - for the latter
// two backends, it will serve as the fallback type if the SIMD library
// does not provide a suitable type.

#include "simd/gen_simd_type.h"

#endif

// On top of the 'generic' SIMD type - either gen_simd_type or
// std_simd_type - we have two back-ends which provide more elaborate
// SIMD capabilities for a subset of the entire zimt range. These
// back-ends can provide the entire functionality if only the allowed
// fundamentals and lane counts are used.
// USE_VC and USE_HWY are mutually exclusive.

#if defined USE_VC

#include "simd/vc_simd_type.h"

#elif defined USE_HWY

#include "simd/hwy_simd_type.h"

#endif

namespace zimt
{
using namespace simd ;

#ifdef USE_STDSIMD

template < typename U , std::size_t M >
using gen_simd_type = std_simd_type < U , M > ;

#endif

#ifndef ZIMT_VECTOR_NBYTES

#if defined USE_VC

#define ZIMT_VECTOR_NBYTES (2*sizeof(Vc::Vector<float>))

#else

// note that hwy_simd_type.h #defines this value already, so with
// the hwy back-end, the lane count depends on the value used there,
// currently four hardware vectors' worth.

#define ZIMT_VECTOR_NBYTES 64

#endif

#endif

/// traits class simd_traits provides three traits:
///
/// - 'hsize' holds the hardware vector width if applicable (used only with Vc)
/// - 'type': template yielding the vector type for a given vectorization width
/// - 'default_size': the default vectorization width to use for T

/// default simd_traits: without further specialization, T will be vectorized
/// as a gen_simd_type. This way, *all* types will be vectorized, there is
/// no fallback to scalar code for certain types. Scalar code will only be
/// produced if the vectorization width is set to 1 in code taking this
/// datum as a template argument. Note that the type which simd_traits produces
/// for sz == 1 is T itself, not a simd_type of one element.

template < typename T >
struct simd_traits
{
  template < size_t sz > using type =
    typename std::conditional < sz == 1 ,
                                T ,
                                gen_simd_type < T , sz >
                              > :: type ;

  static const size_t hsize = 0 ;

  // the default vector size picked here comes out as 16 for floats,
  // which is twice the vector size used by AVX2.

  enum { default_size =   sizeof ( T ) > ZIMT_VECTOR_NBYTES
                        ? 1
                        : ZIMT_VECTOR_NBYTES / sizeof ( T ) } ;
} ;

// next, for some SIMD backends, we specialize simd_traits for a given
// set of fundamentals. fundamental T which we don't mark this way
// will be routed to use gen_simd_type, the 'goading' implementation.

#if defined USE_VC

template < typename T , std::size_t N >
struct allocator_traits < vc_simd_type < T , N > >
{
  typedef Vc::Allocator < vc_simd_type < T , N > >
    type ;
} ;

// in Vc ML discussion M. Kretz states that the set of types Vc can vectorize
// (with 1.3) is consistent throughout all ABIs, so we can just list the
// acceptable types without having to take the ABI into account.
// So, for these types we specialize 'simd_traits', resulting in the use of
// the appropriate Vc::SimdArray.

#define VC_SIMD(T) \
template<> struct simd_traits<T> \
{ \
  static const size_t hsize = Vc::Vector < T > :: size() ; \
  template < size_t sz > using type = \
    typename std::conditional \
             < sz == 1 , \
               T , \
               vc_simd_type < T , sz > \
             > :: type ; \
  enum { default_size =   sizeof ( T ) > ZIMT_VECTOR_NBYTES \
                        ? 1 \
                        : ZIMT_VECTOR_NBYTES / sizeof ( T ) } ; \
} ;

VC_SIMD(float)
VC_SIMD(double)
VC_SIMD(int)
VC_SIMD(unsigned int)
VC_SIMD(short)
VC_SIMD(unsigned short)

#undef VC_SIMD

#elif defined USE_HWY

// with highway, we route pretty much all fundamentals to hwy_simd_type;
// highway seems to have a wider 'feeding spectrum' of fundamentals.
// additionally, route only vector widths which are a power of two
// to produce hwy_simd_type. hwy_simd_type does not handle other
// vector sizes.

#define HWY_SIMD(T) \
template<> struct simd_traits<T> \
{ \
  static const size_t hsize = 0 ; \
  template < size_t sz > using type = \
    typename std::conditional \
             < sz == 1 , \
               T , \
               typename std::conditional \
               < ( sz & ( sz - 1 ) ) == 0 , \
                 hwy_simd_type < T , sz > , \
                 zimt::gen_simd_type < T , sz > \
               > :: type \
             > :: type ; \
  enum { default_size =   sizeof ( T ) > ZIMT_VECTOR_NBYTES \
                        ? 1 \
                        : ZIMT_VECTOR_NBYTES / sizeof ( T ) } ; \
} ;

HWY_SIMD(float)
HWY_SIMD(double)
HWY_SIMD(long)
HWY_SIMD(unsigned long)
HWY_SIMD(int)
HWY_SIMD(unsigned int)
HWY_SIMD(short)
HWY_SIMD(unsigned short)
HWY_SIMD(signed char)
HWY_SIMD(unsigned char)

#undef HWY_SIMD

#endif

} ; // namespace zimt

#define ZIMT_SIMD_H
#endif
