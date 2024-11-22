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
*/
    
#if defined(ZIMT_VECTOR_H) == defined(HWY_TARGET_TOGGLE)
  #ifdef ZIMT_VECTOR_H
    #undef ZIMT_VECTOR_H
  #else
    #define ZIMT_VECTOR_H
  #endif

#include "simd.h"

HWY_BEFORE_NAMESPACE() ;
BEGIN_ZIMT_SIMD_NAMESPACE(zimt)

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
///   zimt::gen_simd_type instead of Vc::SimdArray otherwise.
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
  // be either Vc::SimdArray or zimt::gen_simd_type
  
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
    assign ( t [ i ] , u [ i ] ) ;
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

END_ZIMT_SIMD_NAMESPACE
HWY_AFTER_NAMESPACE() ;

#endif // for sentinel
