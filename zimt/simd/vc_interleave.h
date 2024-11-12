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

/*! \file vc_interleave.h

    \brief de/interleaving with Vc
*/

#if defined(ZIMT_VC_INTERLEAVE_H) == defined(HWY_TARGET_TOGGLE)
  #ifdef ZIMT_VC_INTERLEAVE_H
    #undef ZIMT_VC_INTERLEAVE_H
  #else
    #define ZIMT_VC_INTERLEAVE_H
  #endif

HWY_BEFORE_NAMESPACE() ;
BEGIN_ZIMT_SIMD_NAMESPACE(zimt)

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
void fetch ( XEL
             < vc_simd_type < T , K * Vc::Vector<T>::size() > , N > & v ,
             const XEL < T , N > * _data ,
             const size_t & sz ,
             Vc::index_sequence < seq ... > )
{
  const Vc::InterleavedMemoryWrapper < const XEL < T , N > ,
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

  enum { nv =   sizeof ( vc_simd_type < T , K * Vc::Vector < T > :: size() > )
              / sizeof ( Vc::Vector < T > ) } ;

  typedef typename XEL < Vc::Vector < T > , nv > as_v1_type ;
  typedef typename XEL < as_v1_type , N > as_vn_type ;

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
void stash ( const XEL
             < vc_simd_type < T , K * Vc::Vector<T>::size() > , N > & v ,
             XEL < T , N > * _data ,
             const size_t & sz ,
             Vc::index_sequence < seq ... > )
{
  Vc::InterleavedMemoryWrapper < XEL < T , N > ,
                                 Vc::Vector<T> > data ( _data ) ;

  // we reinterpret the incoming reference to have as_v1_type
  // as value_type, just as in 'fetch' above.

  // KFJ 2018-02-20 given VS as the size of a Vc::Vector<T>, I had initially
  // coded as if a SimdArray<T,VS*K> had a size of VS*K, so just as much as
  // K Vc::Vector<T> occupy. This is not necessarily so, the SimdArray may
  // be larger. Hence this additional bit of size arithmetics to make the
  // reinterpret_cast below succeed for all K, which calculates the number
  // of Vc::Vectors, nv, which occupy the same space as the SimdArray

  enum { nv =   sizeof ( vc_simd_type < T , K * Vc::Vector < T > :: size() > )
              / sizeof ( Vc::Vector < T > ) } ;

  typedef typename XEL < Vc::Vector < T > , nv > as_v1_type ;
  typedef typename XEL < as_v1_type , N > as_vn_type ;

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

/// de/interleave overloads, which are only enabled if vsz is a multiple
/// of the SIMD vector capacity. delegates to detail::fetch, which
/// handles the data acquisition with a Vc::InterleavedMemoryWrapper.
/// This overload is only for unstrided multichannel data and overrides
/// the deinterleave template in xel.h which uses gather/scatter

template < typename ele_type , std::size_t chn , std::size_t vsz >
typename std::enable_if < vsz % Vc::Vector<ele_type>::size() == 0 > :: type
deinterleave ( const XEL < ele_type , chn > * const & src ,
        XEL < vc_simd_type < ele_type , vsz > , chn > & trg )
{
  enum { K = vsz / Vc::Vector<ele_type>::size() } ;

  detail::fetch < ele_type , chn , K >
    ( trg , src , 0 , Vc::make_index_sequence<chn>() ) ;
}

template < typename ele_type , std::size_t chn , std::size_t vsz >
typename std::enable_if < vsz % Vc::Vector<ele_type>::size() == 0 > :: type
interleave ( const XEL < vc_simd_type < ele_type , vsz > , chn > & src ,
        XEL < ele_type , chn > * const & trg )
{
  enum { K = vsz / Vc::Vector<ele_type>::size() } ;

  detail::stash < ele_type , chn , K >
    ( src , trg , 0 , Vc::make_index_sequence<chn>() ) ;
}

END_ZIMT_SIMD_NAMESPACE
HWY_AFTER_NAMESPACE() ;

#endif // sentinel
