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

/*! \file vector.h

    \brief code for horizontal vectorization in zimt
    
    zimt currently has three ways of approaching vectorization:
  
    - no vectorization. Scalar code is less complex since it does not
      have to aggregate the data into vectorization-friendly parcels,
      and for some data types, the performance is just as good as with
      vectorization. Use of scalar code results from setting the
      vectorization width to 1. This is usually a template argument
      going by the name 'vsize'.
  
    - Use of Vc for vectorization. This requires the presence of Vc
      during compilation and results in explicit vectorization for all
      elementary types Vc can handle. Vc provides code for several
      operations which are outside the scope of autovectorization,
      most prominently hardware gather and scatter operations, and the
      explicit vectorization with Vc makes sure that vectorization is
      indeed used whenever possible, rather than having to rely on the
      compiler to recognize the opportunity. Use of Vc has to be
      explicitly activated by defining USE_VC during compilation.
      Using this option usually produces the fastest code. The downside
      is the dependence on an external library which may or may not
      actually implement the intended vector operations with vector
      code for a given target: Newer processors may not yet be supported,
      or support may be implemented for part of the instructions only.
      Also, the Vc version coming from the distro's packet management
      may not be up-to-date. Building processing pipelines based on
      Vc::SimdArray is, on the other hand, straightforward - the type
      is well-thought-out and there is good library support for many
      operations. Use of Vc triggers use of fallback code for elementary
      types which Vc can't vectorize - such types are pseudo-vectorized:
  
    - The third option is to produce code which is designed to be
      easily recognized by the compiler as amenable to autovectorization.
      This option is implemented in simd_type.h, which defines an
      arithmetic type 'zimt::simd_type' holding data in a C vector.
      This is a technique I call 'goading': data are processed in small
      aggregates of vector friendly size, resulting in inner loops
      which oftentimes are recognized by the autovectorization stage,
      resulting in hardware vector code if the compiler flags allow
      for it and the compiler can generate code for the intended target.
      Since this approach relies entirely on the compiler's capability
      to autovectorize the (deliberately vectorization-friendly) code,
      the mileage varies. If it works, this is a clean and simple
      solution. A disadvantage is the use of class simd_type for
      vectorization, which is mildly exotic and very much a zimt
      creature - building processing pipelines using this type will
      not be as effortless as using Vc::SimdArray. As long as you're
      not building your own functors to be used with zimt's family
      of transform-like functions, the precise mode of vectorization
      remains an internal issue and you needn't concern yourself with
      with it beyond choosing whether you want zimt to use Vc or not,
      and choosing a suitable vectorization width if the default does
      not suit you. Class zimt::simd_type can 'vectorize' every
      fundamental and is used as fallback type when Vc use is allowed
      but Vc can't provide a vectorized data type, like for 'long double'
      data, so it will be used even with Vc active when the need arises.
  
    It's important to understand that using SIMD is not simply mapping,
    say, pixels of three floats to a vector of three floats - that would
    be 'vertical' vectorization, which is represented by zimt's *scalar*
    code. Instead, zimt is coded to use *horizontal* vectorization,
    which produces vector data fitting the size of the vector unit's
    registers, where each element held by the vector has exactly the same
    meaning as every other: rather than vectors holding, like, the colour
    channels of a pixel, we have a 'red', a 'green' and a 'blue' vector
    holding, say, eight floats each. Horizontal vectorization is best
    explicitly coded, and if it is coded explicitly, the code structure
    itself suggests vectorization to the compiler. Using code like Vc
    gives more structure to this process and adds capabilities beyond the
    scope of autovectorization, but having the horizontal vectorization
    manifest in the code's structure already goes a long way, and if the
    'structurally' vectorized code autovectorizes well, that may well be
    'good enough' as it is. In my experience, it is often significantly
    faster than scalar code - provided the processor has vector units.

    So it turns out that successful vectorization is, to a large degree,
    a *conceptual change* making the intended vectorization explicit by
    choosing appropriate data types. I am indebted to Matthis Kretz, the
    author of the Vc library, who has opened my eyes to this fact with his
    thesis: 'Extending C++ for explicit data-parallel programming via SIMD
    vector types'.

    With zimt::simd_type 'in the back hand' zimt code can rely on
    the presence of a vectorized type for every fundamental, and, by
    extension, vectorized 'xel' data - i.e. vectorized pixels, voxels
    etc. which are implemented as zimt::xel_ts of vectorized
    fundamentals. This allows zimt to be coded so that it relies
    on vectorization, but not necessarily on Vc: Vc is an option to
    provide extra-fast, tailormade vector code for some operations,
    but when it can't be used, zimt's own vector code will be used
    instead, providing *the same interface*. This makes maintainance
    much easier compared to a scenario where, without Vc, the code
    would have to fall back to a scalar version - as indeed it did in
    early zimt versions, giving me plenty of headaches.

    Note that this header is included by zimt/common.h, so this code
    is available throughout zimt.
*/

#ifndef ZIMT_VECTOR_H
#define ZIMT_VECTOR_H

#include "common.h"

// we have several SIMD back-ends to provide SIMD code, which are all
// activated by preprocessor #defines. 'Proper' SIMD code may be
// generated with Vc (USE_VC), highway (USE_HWY) and std::simd
// (USE_STDSIMD). Only one of these should be defined - it may be
// possible to mix them, but this is uncharted territory. If none
// of the 'proper' SIMD backends are specified, zimt will fall
// back to it's own 'goading' implementation, which codes SIMD
// operations as small loops, hoping that they will be autovectorized.

#include "simd_type.h"

#if defined USE_STDSIMD

#include "std_simd_type.h"

#elif defined USE_VC

#include "vc_simd_type.h"

#elif defined USE_HWY

#include "hwy_simd_type.h"

#endif

namespace zimt
{
#ifndef ZIMT_VECTOR_NBYTES

#if defined USE_VC

#define ZIMT_VECTOR_NBYTES (2*sizeof(Vc::Vector<float>))

#else

#define ZIMT_VECTOR_NBYTES 64

#endif

#endif

/// traits class simd_traits provides three traits:
///
/// - 'hsize' holds the hardware vector width if applicable (used only with Vc)
/// - 'type': template yielding the vector type for a given vectorization width
/// - 'default_size': the default vectorization width to use for T

/// default simd_traits: without further specialization, T will be vectorized
/// as a COMMON_SIMD_TYPE. This way, *all* types will be vectorized, there is
/// no fallback to scalar code for certain types. Scalar code will only be
/// produced if the vectorization width is set to 1 in code taking this
/// datum as a template argument. Note that the type which simd_traits produces
/// for sz == 1 is T itself, not a simd_type of one element.

#if defined USE_STDSIMD

// if USE_STDSIMD is defined, we route all types T to std_simd_type,
// because there won't be any specializations of simd_traits.

#define COMMON_SIMD_TYPE std_simd_type

#else

// otherwise, we define the template using zimt::simd_type as the
// fallback type and specialize where the backend can provide.

#define COMMON_SIMD_TYPE simd_type

#endif

template < typename T >
struct simd_traits
{
  template < size_t sz > using type =
    typename std::conditional < sz == 1 ,
                                T ,
                                COMMON_SIMD_TYPE < T , sz >
                              > :: type ;
                              
  static const size_t hsize = 0 ;
  
  // the default vector size picked here comes out as 16 for floats,
  // which is twice the vector size used by AVX2.

  enum { default_size =   sizeof ( T ) > ZIMT_VECTOR_NBYTES
                        ? 1
                        : ZIMT_VECTOR_NBYTES / sizeof ( T ) } ;
} ;

#undef COMMON_SIMD_TYPE

// next, for some SIMD backends, we specialize simd_traits for a given
// set of fundamentals. fundamental T which we don't mark this way
// will be routed to use zimt::simd_type, the 'goading' implementation.

#if defined USE_VC

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
                 zimt::simd_type < T , sz > \
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

/// with the definition of 'simd_traits', we can proceed to implement
/// 'vector_traits':
/// struct vector_traits is a traits class fixing the types used for
/// vectorized code in zimt. These types go beyond mere vectors of
/// fundamentals: most of the time, the data zimt has to process
/// are *not* fundamentals, but what I call 'xel' data: pixels, voxels,
/// stereo sound samples, etc. - so, small aggregates of a fundamental
/// type. vector_traits defines how fundamentals and 'xel' data are to
/// be vectorized.
/// with the types defined by vector_traits, a system of type names is
/// introduced which uses a set of patterns:
/// - 'ele' stands for 'elementary', the type of an aggregate's component
/// - 'nd' stands for 'n-dimensional', a type of an aggregate of one or
///    more components of a given elementary type.
/// - 'v' suffix indicates a 'simdized' type, zimt uses Vc::SimdArrays
///   and zimt::xel_ts of Vc::SimdArrays if Vc is used and the type
///   can be used with Vc::SimdArray, and the equivalent types using
///   zimt::simd_type instead of Vc::SimdArray otherwise.
/// the unspecialized definition of class vector_traits will vectorize
/// by concatenating instances of T into the type simd_traits produces,
/// taking, per default, as many T as the default_size given there.
/// This will work with any type T, even though it makes most sense with
/// fundamentals.

template < typename T ,
           size_t _vsize = 0 ,
           typename Enable = void >
struct vector_traits
{
  // T is not 'element-expandable', so 'dimension' is 1 and T is ele_type
  enum { dimension = 1 } ;
  typedef T ele_type ;
  
  // find the right vectorization width
  enum { size = _vsize == 0
                ? simd_traits < ele_type > :: default_size
                : _vsize } ;

  enum { vsize = size } ;
  enum { hsize = simd_traits < ele_type > :: hsize } ;
  
  // produce the 'synthetic' type,
  typedef zimt::xel_t < ele_type , 1 > nd_ele_type ;
  
  // the vectorized type
  template < typename U , size_t sz >
  using vector = typename simd_traits < U > :: template type < sz > ;
  
  typedef vector < ele_type , vsize > ele_v ; 
  
  // and the 'synthetic' vectorized type
  typedef zimt::xel_t < ele_v , 1 > nd_ele_v ;
  
  // for not 'element-expandable' T, we produce ele_v as 'type'
  typedef ele_v type ;
} ;

/// specialization of vector_traits for 'element-expandable' types.
/// These types are recognized by vigra's ExpandElementResult mechanism,
/// resulting in the formation of a 'vectorized' version of the type.
/// These data are what I call 'xel' data. As explained above, vectorization
/// is *horizontal*, so if T is, say, a pixel of three floats, the type
/// generated here will be a TinyVector of three vectors of vsize floats.

template < typename T , size_t _vsize >
struct vector_traits
       < T ,
         _vsize ,
         typename std::enable_if
                  < zimt::is_element_expandable < T > :: value
                  > ::type
       >
{
  // T is 'element-expandable' - meaning it can be element-expanded
  // with vigra's ExpandElementResult mechanism. We use that to obtain
  // the elementary type and the dimension of T. Note that, if T is
  // fundamental, the resulting traits are the same as they would be
  // for the unspecialized case. What we're interested in here are
  // multi-channel types; that fundamentals are routed through here
  // is just as good as if they were routed through the unspecialized
  // case above.

  enum { dimension = zimt::get_ele_t < T > :: size } ;
  typedef typename zimt::get_ele_t < T > :: type ele_type ;
  
  // given the elementary type, we define nd_ele_type as a zimt::xel_t
  // of ele_type. This is the 'synthetic' type.

  typedef zimt::xel_t < ele_type , dimension > nd_ele_type ;
  
  // next we glean the number of elements a 'vector' should contain.
  // if the template argument 'vsize' was passed as 0, which is the default,
  // We use the default vector size which simd_traits provides. For
  // explicitly specified _vsize we take the explicitly specified value.

  enum { size = _vsize == 0
                ? simd_traits < ele_type > :: default_size
                : _vsize } ;

  // I prefer to use 'vsize' as it is more specific than mere 'size'
                
  enum { vsize = size } ;
  
  // hardware vector register size, if applicable - only used with Vc
  
  enum { hsize = simd_traits < T > :: hsize } ;

  // now we obtain the template for a vector of a given size. This will
  // be either Vc::SimdArray or zimt::simd_type
  
  template < typename U , size_t sz >
  using vector = typename simd_traits < U > :: template type < sz > ;
  
  // using this template and the vectorization width we have established,
  // we obtain the vectorized data type for a component:

  typedef vector < ele_type , vsize > ele_v ; 
  
  // nd_ele_v is the 'synthetic' vectorized type, which is always a
  // TinyVector of the vectorized component type, possibly with only
  // one element:
  
  typedef zimt::xel_t < ele_v , dimension > nd_ele_v ;
  
  // finally, 'type' is the 'canonical' vectorized type, meaning that if
  // T is a fundamental we produce the component vector type itself, but if
  // it is some aggregate (like a TinyVector) we produce a TinyVector of the
  // component vector data type. So if T is float, 'type' is a vector of float,
  // If T is a TinyVector of one float, 'type' is a TinyVector of one vector
  // of float.
  
  typedef typename std::conditional
    < std::is_fundamental < T > :: value ,
      ele_v ,
      nd_ele_v
    > :: type type ;
    
} ;

/// this alias is used as a shorthand to pick the vectorized type
/// for a given type T and a size N from 'vector_traits':

template < typename T , size_t N >
using simdized_type = typename vector_traits < T , N > :: type ;

// In order to avoid syntax which is specific to a specific vectorization
// method, I use some the free function 'assign' for assignments, which
// avoids member functions of the vector objects. While this produces some
// notational inconvenience, it allows a formulation which is independent
// of the vector type used. this way I can use Vc::SimdArray as a target
// of an assignment from another vectorized data type, which would be
// impossible with operator=, which has to be a member function.

// the fallback is to use an assignment via operator=. Most of the time,
// this is defined and works as expected.

template < typename T , typename U >
void assign ( T & t , const U & u )
{
  t = T ( u ) ;
}

// assignment between two zimt::xel_ts of equal size. This 'rolls
// out' the per-element assignment

template < typename T , typename U , int N >
void assign ( zimt::xel_t < T , N > & t ,
              const zimt::xel_t < U , N > & u )
{
  for ( int i = 0 ; i < N ; i++ )
    zimt::assign ( t [ i ] , u [ i ] ) ;
}

// conditional assignment as a free function. This is helpful to code
// uniformly, avoiding the idiomatic difference between
// if ( p ) t = s ; and t ( p ) = s ;

template < typename VT1 , typename PT , typename VT2 >
void assign_if ( VT1 & target ,
                 const PT & predicate ,
                 const VT2 & source )
{
  target ( predicate ) = source ;
}

template < typename T >
void assign_if ( T & target ,
            const bool & predicate ,
            const T & source )
{
  if ( predicate )
    target = source ;
}

} ; // end of namespace zimt

#endif // #ifndef ZIMT_VECTOR_H
