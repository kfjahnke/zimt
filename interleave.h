/************************************************************************/
/*                                                                      */
/*    zimt - abstraction layer for SIMD programming                     */
/*                                                                      */
/*            Copyright 2023 by Kay F. Jahnke                           */
/*                                                                      */
/*    The git repository for this software is at                        */
/*                                                                      */
/*    https://bitbucket.org/kfj/zimt                                    */
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

/*! \file interleave.h

    \brief Implementation of 'bunch' and 'fluff'

    The two function templates 'bunch' and 'fluff' provide code to
    access interleaved memory holding 'xel' data: stuff like pixels,
    or coordinates - data types which consist of several equally-typed
    fundamentals. 'bunch' fetches data from interleaved memory and
    deposits them in a set of vectors, and 'fluff' does the reverse
    operation. There are single-channel variants routing to the
    more efficient simple load/store operation. Strided data will
    be handled correctly, but the specialized library code which
    realizes the access as a load/shuffle or shuffle/store does
    require unstrided data, so it will only be used if the stride
    is one (measured in 'xel' units') - strides larger than one will
    be routed to the less specialized code.

    de/interleaving is a common operation, and speeding it up does
    usually pay off. The most basic approach used here is 'goading':
    the memory access is coded as a small loop, hoping the compiler
    will 'get it' and autovectorize the operation. Mileage will vary.
    'One step up' is the use of 'regular gather/scatter' - a gather
    or scatter operation with fixed indices. This may still route to
    'goading' code if the current ISA does not provide gather/scatter.
    The best perfromance will usually arise from routing to dedicated
    de/interleaving code, like Vc's InterleavedMemoryWrapper or
    highway's StoreInterleaved function templates.

    Because the access to interleaved memory is a recognizably
    separate operation, I have factored out the code to this header.
    The code is used extensively by wielding.h.
*/

#ifndef INERLEAVE_H
#define INERLEAVE_H

#include "xel.h"

#ifdef USE_HWY

// if we have highway, we can use some of the specialized functions
// to access interleaved memory.

#include "hwy_simd_type.h"

namespace zimt
{
  using hwy::HWY_NAMESPACE::StoreInterleaved2 ;
  using hwy::HWY_NAMESPACE::StoreInterleaved3 ;
  using hwy::HWY_NAMESPACE::StoreInterleaved4 ;
  using hwy::HWY_NAMESPACE::LoadInterleaved2 ;
  using hwy::HWY_NAMESPACE::LoadInterleaved3 ;
  using hwy::HWY_NAMESPACE::LoadInterleaved4 ;
} ;

#endif

namespace wielding
{

typedef int ic_type ;

#ifdef USE_VC

// if we have Vc, we'll use Vc::InterleavedMemoryWrapper if possible.
// This takes some coding effort to get the routing right.

namespace detail
{

// Here we have some collateral code to use Vc's InterleavedMemoryWrapper.
// This is a specialized way of accessing interleaved but unstrided data,
// which uses several SIMD loads, then reshuffles the data. This should
// be quicker than using a set of gather operations.

// fetch of interleaved, but unstrided data located at _data
// into a TinyVector of zimt::simdized_types using InterleavedMemoryWrapper.
// uses SimdArrays containing K full hardware SIMD Vectors

template < typename T , size_t N , size_t K , size_t ... seq >
void fetch ( zimt::xel_t
             < zimt::simdized_type < T , K * Vc::Vector<T>::size() > , N > & v ,
             const zimt::xel_t < T , N > * _data ,
             const size_t & sz ,
             Vc::index_sequence < seq ... > )
{
  const Vc::InterleavedMemoryWrapper < const zimt::xel_t < T , N > ,
                                       Vc::Vector<T> > data ( _data ) ;

  // as_v1_type is a type holding K Vc::Vector<T> in a TinyVector.
  // we reinterpret the incoming reference to have as_v1_type
  // as value_type - instead of an equally-sized SimdArray. With
  // this interpretation of the data we can use the
  // InterleavedMemoryWrapper, which operates on Vc::Vectors
  // only.

  // KFJ 2018-02-20 given VS as the size of a Vc::Vector<T>, I had initially
  // coded as if a SimdArray<T,VS*K> had a size of VS*K, so just as much as
  // K Vc::Vector<T> occupy. This is not necessarily so, the SimdArray may
  // be larger. Hence this additional bit of size arithmetics to make the
  // reinterpret_cast below succeed for all K, which calculates the number
  // of Vc::Vectors, nv, which occupy the same space as the SimdArray

  enum { nv =   sizeof ( zimt::simdized_type < T , K * Vc::Vector < T > :: size() > )
              / sizeof ( Vc::Vector < T > ) } ;
                    
  typedef typename zimt::xel_t < Vc::Vector < T > , nv > as_v1_type ;
  typedef typename zimt::xel_t < as_v1_type , N > as_vn_type ;
  
  as_vn_type & as_vn = reinterpret_cast < as_vn_type & > ( v ) ;
  
  // we fill the SimdArrays in as_vn round-robin. Note the use of
  // Vc::tie - this makes the transition effortless.
  
  for ( size_t k = 0 ; k < K ; k++ )
  {
    Vc::tie ( as_vn [ seq ] [ k ] ... )
      = ( data [ sz + k * Vc::Vector<T>::size() ] ) ;
  }
}

template < typename T , size_t N , size_t K , size_t ... seq >
void stash ( const zimt::xel_t
             < zimt::simdized_type < T , K * Vc::Vector<T>::size() > , N > & v ,
             zimt::xel_t < T , N > * _data ,
             const size_t & sz ,
             Vc::index_sequence < seq ... > )
{
  Vc::InterleavedMemoryWrapper < zimt::xel_t < T , N > ,
                                 Vc::Vector<T> > data ( _data ) ;
  
  // we reinterpret the incoming reference to have as_v1_type
  // as value_type, just as in 'fetch' above.

  // KFJ 2018-02-20 given VS as the size of a Vc::Vector<T>, I had initially
  // coded as if a SimdArray<T,VS*K> had a size of VS*K, so just as much as
  // K Vc::Vector<T> occupy. This is not necessarily so, the SimdArray may
  // be larger. Hence this additional bit of size arithmetics to make the
  // reinterpret_cast below succeed for all K, which calculates the number
  // of Vc::Vectors, nv, which occupy the same space as the SimdArray

  enum { nv =   sizeof ( zimt::simdized_type < T , K * Vc::Vector < T > :: size() > )
              / sizeof ( Vc::Vector < T > ) } ;
                    
  typedef typename zimt::xel_t < Vc::Vector < T > , nv > as_v1_type ;
  typedef typename zimt::xel_t < as_v1_type , N > as_vn_type ;
  
  const as_vn_type & as_vn = reinterpret_cast < const as_vn_type & > ( v ) ;
  
  // we unpack the SimdArrays in as_vn round-robin. Note, again, the use
  // of Vc::tie - I found no other way to assign to data[...] at all.
  
  for ( size_t k = 0 ; k < K ; k++ )
  {
    data [ sz + k * Vc::Vector<T>::size() ]
      = Vc::tie ( as_vn [ seq ] [ k ] ... ) ;
  }
}

} ; // end of namespace detail

// here we have the versions of bunch and fluff using specialized
// Vc operations to access the buffer. These routines take Vc data
// types, and they are only present if USE_VC is defined at all.
// Further down we have less specific signatures which will be chosen
// if either Vc is not used at all or if the data types passed are
// not Vc types.

/// bunch picks up data from interleaved, strided memory and stores
/// them in a data type representing a package of vector data.

/// The first overload of 'bunch' uses a gather operation to obtain
/// the data from memory. This overload is used if the source data
/// are strided and are therefore not contiguous in memory. It's
/// also used if unstrided data are multi-channel and the vector width
/// is not a multiple of the hardware vector width, because I haven't
/// fully implemented using Vc::InterleavedMemoryWrapper for SimdArrays.
/// This first routine can be used for all situations, the two overloads
/// below are optimizations, increasing performance for specific
/// cases.

template < typename ele_type , std::size_t chn , std::size_t vsz >
void bunch ( const zimt::xel_t < ele_type , chn > * const & src ,
              zimt::xel_t < zimt::vc_simd_type < ele_type , vsz > , chn > & trg ,
              const ic_type & stride )
{
  typedef typename zimt::vc_simd_type < ele_type , vsz > :: index_type index_type ;
  typedef typename index_type::value_type ix_t ;
  ix_t _chn = chn ;
  index_type ix = index_type::IndexesFromZero() * stride * _chn ;
  
  for ( int d = 0 ; d < chn ; d++ )
    trg[d].gather ( ((ele_type*)src) + d , ix ) ;
}

/// overload for unstrided single-channel data.
/// here we can use an SIMD load, the implementation is very
/// straightforward, and the performance gain is large.

template < typename ele_type , std::size_t vsz >
void bunch ( const zimt::xel_t < ele_type , 1 > * const & src ,
              zimt::xel_t < zimt::vc_simd_type < ele_type , vsz > , 1 > & trg ,
              std::true_type
            )
{
  trg[0].load ( (const ele_type*) src ) ;
}

/// the third overload, which is only enabled if vsz is a multiple
/// of the SIMD vector capacity, delegates to detail::fetch, which
/// handles the data acquisition with a Vc::InterleavedMemoryWrapper.
/// This overload is only for unstrided multichannel data.

template < typename ele_type , std::size_t chn , std::size_t vsz >
typename std::enable_if < vsz % Vc::Vector<ele_type>::size() == 0 > :: type 
bunch ( const zimt::xel_t < ele_type , chn > * const & src ,
        zimt::xel_t < zimt::vc_simd_type < ele_type , vsz > , chn > & trg ,
        std::false_type
      )
{
  enum { K = vsz / Vc::Vector<ele_type>::size() } ;
  
  detail::fetch < ele_type , chn , K >
    ( trg , src , 0 , Vc::make_index_sequence<chn>() ) ;
}

/// reverse operation: a package of vectorized data is written to
/// interleaved, strided memory. We have the same sequence
/// of overloads as for 'bunch'.

template < typename ele_type , std::size_t chn , std::size_t vsz >
void fluff ( const zimt::xel_t < zimt::vc_simd_type < ele_type , vsz > , chn > & src ,
              zimt::xel_t < ele_type , chn > * const & trg ,
              const ic_type & stride )
{
  typedef typename zimt::vc_simd_type < ele_type , vsz > :: index_type index_type ;
  typedef typename index_type::value_type ix_t ;
  ix_t _chn = chn ;
  index_type ix = index_type::IndexesFromZero() * stride * _chn ;

  for ( int d = 0 ; d < chn ; d++ )
    src[d].scatter ( ((ele_type*)trg) + d , ix ) ;
}

template < typename ele_type , std::size_t vsz >
void fluff ( const zimt::xel_t < zimt::vc_simd_type < ele_type , vsz > , 1 > & src ,
              zimt::xel_t < ele_type , 1 > * const & trg ,
              std::true_type
            )
{
  src[0].store ( (ele_type*) trg ) ;
}

template < typename ele_type , std::size_t chn , std::size_t vsz >
typename std::enable_if < vsz % Vc::Vector<ele_type>::size() == 0 > :: type 
fluff ( const zimt::xel_t < zimt::vc_simd_type < ele_type , vsz > , chn > & src ,
        zimt::xel_t < ele_type , chn > * const & trg ,
        std::false_type
      )
{
  enum { K = vsz / Vc::Vector<ele_type>::size() } ;
  
  detail::stash < ele_type , chn , K >
    ( src , trg , 0 , Vc::make_index_sequence<chn>() ) ;
}

#endif // USE_VC

// when not processing Vc data , bunch and fluff use rgather/rscatter
// if availabe, and otherwise 'goading' for buffering and unbuffering.
// If the data are single-channel and unstrided, SIMD load/store
// operations are used which are the fastest:

// data are unstrided and single-channel, issue a SIMD load operation

template < typename target_type , typename ele_type >
void bunch ( const zimt::xel_t < ele_type , 1 > * const & src ,
             target_type & trg ,
             std::true_type
           )
{
  trg[0].load ( reinterpret_cast < const ele_type * const > ( src ) ) ;    
}

// data are unstrided and single-channel, issue a SIMD store

template < typename ele_type , typename source_type >
void fluff ( const source_type & src ,
             zimt::xel_t < ele_type , 1 > * const & trg ,
             std::true_type
           )
{
  src[0].store ( reinterpret_cast < ele_type * const > ( trg ) ) ;    
}

template < typename target_type , typename ele_type , std::size_t chn >
void _bunch ( const zimt::xel_t < ele_type , chn > * const & src ,
              target_type & trg ,
              const ic_type & stride )
{
  const ele_type * p_src = reinterpret_cast < const ele_type * const > ( src ) ;
  std::size_t estride = stride * chn ;
  for ( int ch = 0 ; ch < chn ; ch++ )
  {
    trg[ch].rgather ( p_src , estride ) ;
    ++p_src ;
  }
}

template < typename ele_type , typename source_type , std::size_t chn >
void _fluff ( const source_type & src ,
              zimt::xel_t < ele_type , chn > * const & trg ,
              const ic_type & stride
            )
{
  ele_type * p_trg = reinterpret_cast < ele_type * const > ( trg ) ;
  std::size_t estride = stride * chn ;
  for ( int ch = 0 ; ch < chn ; ch++ )
  {
    src[ch].rscatter ( p_trg , estride ) ;
    ++p_trg ;
  }
}

template < typename target_type , typename ele_type , std::size_t chn >
void bunch ( const zimt::xel_t < ele_type , chn > * const & src ,
             target_type & trg ,
             const ic_type & stride )
{
  _bunch ( src , trg , stride ) ;
}

template < typename ele_type , typename source_type , std::size_t chn >
void fluff ( const source_type & src ,
             zimt::xel_t < ele_type , chn > * const & trg ,
             const ic_type & stride
           )
{
  _fluff ( src , trg , stride ) ;
}

#ifdef USE_HWY

// for the time being, highway can interleave 2, 3, and 4-channel
// data. We route the code with overloads, because there is no
// generalized code for an arbitrary number of channels.
// code 'fluffing' xel data with more than four channels will route
// to one of the templates above, using gather/scatter or goading.

template < typename T , std::size_t vsz >
void fluff ( const zimt::xel_t
                < zimt::hwy_simd_type < T , vsz > , 2 > & src ,
              zimt::xel_t < T , 2 > * const & trg ,
              const ic_type & stride )
{
  typedef typename zimt::hwy_simd_type < T , vsz > :: D D ;
  if ( stride == 1 )
  {
    T * p_trg = (T*) trg ;
    for ( std::size_t n = 0 , i = 0 ; n < vsz ; ++i , n += src[0].L() )
    {
      StoreInterleaved2 ( src[0].yield ( i ) ,
                          src[1].yield ( i ) ,
                          D() ,
                          p_trg ) ;
      p_trg += 2 * src[0].L() ;
    }
  }
  else
  {
    _fluff ( src , trg , stride ) ;
  }
}
  
template < typename T , std::size_t vsz >
void fluff ( const zimt::xel_t
                < zimt::hwy_simd_type < T , vsz > , 3 > & src ,
              zimt::xel_t < T , 3 > * const & trg ,
              const ic_type & stride )
{
  typedef typename zimt::hwy_simd_type < T , vsz > :: D D ;
  if ( stride == 1 )
  {
    T * p_trg = (T*) trg ;
    for ( std::size_t n = 0 , i = 0 ; n < vsz ; ++i , n += src[0].L() )
    {
      StoreInterleaved3 ( src[0].yield ( i ) ,
                          src[1].yield ( i ) ,
                          src[2].yield ( i ) ,
                          D() ,
                          p_trg ) ;
      p_trg += 3 * src[0].L() ;
    }
  }
  else
  {
    _fluff ( src , trg , stride ) ;
  }
}
  
template < typename T , std::size_t vsz >
void fluff ( const zimt::xel_t
                < zimt::hwy_simd_type < T , vsz > , 4 > & src ,
              zimt::xel_t < T , 4 > * const & trg ,
              const ic_type & stride )
{
  typedef typename zimt::hwy_simd_type < T , vsz > :: D D ;
  if ( stride == 1 )
  {
    T * p_trg = (T*) trg ;
    for ( std::size_t n = 0 , i = 0 ; n < vsz ; ++i , n += src[0].L() )
    {
      StoreInterleaved4 ( src[0].yield ( i ) ,
                          src[1].yield ( i ) ,
                          src[2].yield ( i ) ,
                          src[3].yield ( i ) ,
                          D() ,
                          p_trg ) ;
      p_trg += 4 * src[0].L() ;
    }
  }
  else
  {
    _fluff ( src , trg , stride ) ;
  }
}

template < typename T , std::size_t vsz >
void bunch ( const zimt::xel_t < T , 2 > * const & src ,
             zimt::xel_t
                < zimt::hwy_simd_type < T , vsz > , 2 > & trg ,
             const ic_type & stride )
{
  const T * p_src = (T*) src ;
  typedef typename zimt::hwy_simd_type < T , vsz > :: vec_t vec_t ;
  typedef  typename zimt::hwy_simd_type < T , vsz > :: D D ;
  if ( stride == 1 )
  {
    const T * p_src = (const T*) src ;
    vec_t c0 , c1 ;
    for ( std::size_t n = 0 , i = 0 ; n < vsz ; ++i , n += trg[0].L() )
    {
      LoadInterleaved2 ( D() , p_src , c0 , c1 ) ;
      trg[0].take ( i , c0 ) ;
      trg[1].take ( i , c1 ) ;
      p_src += 2 * trg[0].L() ;
    }
  }
  else
  {
    _bunch ( src , trg , stride ) ;
  }
}
  
template < typename T , std::size_t vsz >
void bunch ( const zimt::xel_t < T , 3 > * const & src ,
             zimt::xel_t
                < zimt::hwy_simd_type < T , vsz > , 3 > & trg ,
             const ic_type & stride )
{
  const T * p_src = (T*) src ;
  typedef typename zimt::hwy_simd_type < T , vsz > :: vec_t vec_t ;
  typedef  typename zimt::hwy_simd_type < T , vsz > :: D D ;
  if ( stride == 1 )
  {
    const T * p_src = (const T*) src ;
    vec_t c0 , c1 , c2 ;
    for ( std::size_t n = 0 , i = 0 ; n < vsz ; ++i , n += trg[0].L() )
    {
      LoadInterleaved3 ( D() , p_src , c0 , c1 , c2 ) ;
      trg[0].take ( i , c0 ) ;
      trg[1].take ( i , c1 ) ;
      trg[2].take ( i , c2 ) ;
      p_src += 3 * trg[0].L() ;
    }
  }
  else
  {
    _bunch ( src , trg , stride ) ;
  }
}
  
template < typename T , std::size_t vsz >
void bunch ( const zimt::xel_t < T , 4 > * const & src ,
             zimt::xel_t
                < zimt::hwy_simd_type < T , vsz > , 4 > & trg ,
             const ic_type & stride )
{
  const T * p_src = (T*) src ;
  typedef typename zimt::hwy_simd_type < T , vsz > :: vec_t vec_t ;
  typedef  typename zimt::hwy_simd_type < T , vsz > :: D D ;
  if ( stride == 1 )
  {
    const T * p_src = (const T*) src ;
    vec_t c0 , c1 , c2 , c3 ;
    for ( std::size_t n = 0 , i = 0 ; n < vsz ; ++i , n += trg[0].L() )
    {
      LoadInterleaved4 ( D() , p_src , c0 , c1 , c2 , c3 ) ;
      trg[0].take ( i , c0 ) ;
      trg[1].take ( i , c1 ) ;
      trg[2].take ( i , c2 ) ;
      trg[3].take ( i , c3 ) ;
      p_src += 4 * trg[0].L() ;
    }
  }
  else
  {
    _bunch ( src , trg , stride ) ;
  }
}

#endif

} ; // namespace wielding

#endif // #ifndef INERLEAVE_H
