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

/*! \file hwy_interleave.h

    \brief de/interleaving with highway
*/

#if defined(HWY_INTERLEAVE_H) == defined(HWY_TARGET_TOGGLE)
  #ifdef HWY_INTERLEAVE_H
    #undef HWY_INTERLEAVE_H
  #else
    #define HWY_INTERLEAVE_H
  #endif
  
// namespace simd
// {

// HWY_BEFORE_NAMESPACE();
// 
// namespace zimt {
// 
// namespace HWY_NAMESPACE {  // required: unique per target

BEGIN_ZIMT_SIMD_NAMESPACE(zimt)

namespace hn = hwy::HWY_NAMESPACE ;

// for the time being, highway can interleave 2, 3, and 4-channel
// data. We route the code with overloads, because there is no
// generalized code for an arbitrary number of channels.
// code de/interleaving xel data with more than four channels will
// be routed to one of the templates in xel.h, using gather/scatter

template < typename T , std::size_t vsz >
void interleave ( const XEL < hwy_simd_type < T , vsz > , 2 > & src ,
                        XEL < T , 2 > * const & trg )
{
  typedef typename hwy_simd_type < T , vsz > :: D D ;
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

template < typename T , std::size_t vsz >
void interleave ( const XEL < hwy_simd_type < T , vsz > , 3 > & src ,
                        XEL < T , 3 > * const & trg )
{
  typedef typename hwy_simd_type < T , vsz > :: D D ;
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

template < typename T , std::size_t vsz >
void interleave ( const XEL < hwy_simd_type < T , vsz > , 4 > & src ,
                        XEL < T , 4 > * const & trg )
{
  typedef typename hwy_simd_type < T , vsz > :: D D ;
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

template < typename T , std::size_t vsz >
void deinterleave ( const XEL < T , 2 > * const & src ,
                          XEL < hwy_simd_type < T , vsz > , 2 > & trg )
{
  typedef typename hwy_simd_type < T , vsz > :: vec_t vec_t ;
  typedef  typename hwy_simd_type < T , vsz > :: D D ;
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

template < typename T , std::size_t vsz >
void deinterleave ( const XEL < T , 3 > * const & src ,
                          XEL < hwy_simd_type < T , vsz > , 3 > & trg )
{
  typedef typename hwy_simd_type < T , vsz > :: vec_t vec_t ;
  typedef  typename hwy_simd_type < T , vsz > :: D D ;
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

template < typename T , std::size_t vsz >
void deinterleave ( const XEL < T , 4 > * const & src ,
                          XEL < hwy_simd_type < T , vsz > , 4 > & trg )
{
  typedef typename hwy_simd_type < T , vsz > :: vec_t vec_t ;
  typedef  typename hwy_simd_type < T , vsz > :: D D ;
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

} ;

} ;

HWY_AFTER_NAMESPACE();

#endif // sentinel
