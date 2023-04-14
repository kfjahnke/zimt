/************************************************************************/
/*                                                                      */
/*    zimt - abstraction layer for SIMD programming                     */
/*                                                                      */
/*            Copyright 2022 - 2023 by Kay F. Jahnke                    */
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

/*! \file hwy_simd_type.h

    \brief SIMD type using highway

    This is a new, tentative implementation of gen_simd_type
    using highway (https://github.com/google/highway). highway
    provides code to work with hardware SIMD in a portable way,
    but it's still very close to the hardware, and does not
    provide support for vectors larger than the hardware's register
    size. gen_simd_type, on the other hand, is a fixed-size
    construct which may well exceed the hardware size. The 'goading'
    implementation of gen_simd_type uses small loops over a
    POD C vector to implement the functionality - hoping that the
    compiler will 'get it' and autovectorize the code. This here
    implementation is also based on a POD C vector, but the
    functionality is implemented (wherever this seems feasible or
    sensible) by using highway SIMD code. In a way it's enforcing
    by explicit code what 'ordinary' gen_simd_type hopes to
    get from the compiler via autovectorization, and since the
    compiler's 'insight' into the code is limited, the explicit
    approach tends to come out on top, producing SIMD binary more
    often (and in more efficient variants) than the goading
    approach.

    Some of the functionality is implemented by simple goading
    routines. This is either because this is deemed acceptable
    (e.g. printing a simd_t to the console is not in any way
    time critical, nor can it benefit from SIMD code) - or
    because I haven't yet tackled writing 'proper' SIMD code for
    the functionality in question. This state of affairs also reflects
    my implementation strategy: I started out with the 'ordinary'
    gen_simd_type and replaced more and more of the goading
    code by 'proper' SIMD code.

    'Backing' the SIMD vectors like that is only one way of handling
    the SIMD types in the background, but has the advantage of, first,
    being compatible with the goading code (so one can 'go over the
    memory' or 'fall back to scalar') and, second, being general, so
    that both sized and sizeless vectors can be implemented with the
    same code. The disadvantage is that the compiler may not find all
    opportunities for keeping the SIMD code 'afloat' in a set of
    registers, but may at times resort to actually creating and using
    the underlying POD C array, rather than optimizing it away.

    Nevertheless, this implementation seems to tend towards 'proper'
    SIMD code rather than towards the goading implementation.
*/

#ifndef HWY_SIMD_TYPE_H
#define HWY_SIMD_TYPE_H

#include <iostream>
#include <functional>
#include <type_traits>
#include <assert.h>

#include "gen_simd_type.h"

#include <hwy/highway.h>
#include <hwy/contrib/math/math-inl.h>
#include <hwy/aligned_allocator.h>
#include <hwy/print-inl.h>
#include "hwy_atan2.h"

HWY_BEFORE_NAMESPACE();

namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE ;

using namespace simd ;

/// mask type for simd_t. This is a type which holds a set of masks
/// stored in uint8_t, as the highway mask storing function provides.
/// So this type is memory-backed, just like simd_t. Template arguments
/// are the corresponding simd_t's tag type and it's lane count.
/// highway is strict about which vectors and masks can interoperate,
/// and only allows 'direct' interoperation if the types involved
/// 'match' in size. Masks pertaining to vectors of differently-sized T
/// aren't directly interoperable because they don't have the same
/// lane count. One requires k masks of one type and k * 2 ^ i of the
/// other. Here, we follow a different paradigm: The top-level objects
/// we're dealing with have a fixed 'vsize', the number of lanes they
/// hold. This should be a power of two. The paradigm is that objects
/// with equal vsize should be interoperable, no matter what lane count
/// the hardware vectors have which are used to implement their
/// functionality. This makes user code simpler: users pick a vsize
/// which they use for a body of code, all vector-like objects use the
/// common vsize, and the implementation of the vector-like objects
/// takes care of 'rolling out' the operations to hardware vectors.
/// At times this produces what I call 'friction' - if the underlying
/// hardware vectors and masks are not directly compatible, code is
/// needed to interoperate them, and this code can at times be slow.
/// So the recommendation for users is to avoid 'friction' by avoiding
/// mixing differently-sized types, but with the given paradigm, this
/// is a matter of performance tuning rather than imposing constraints
/// on code structure. Some of the 'friction' might be mitigated by
/// additional code using highway's up- and down-scaling routines,
/// but for now the code rather uses 'goading' with small loops over
/// the backing memory, relying on the compiler to handle this
/// efficiently.

template < typename D , std::size_t _vsize >
struct HWY_ALIGN mchunk_t
{
  typedef typename hn::TFromD < D > T ;
  typedef typename hn::Vec < D > vec_t ;
  typedef T value_type ;

  static const std::size_t vsize = _vsize ;

  // pessimistic estimate: we can be certain that vsize bytes
  // will suffice: that would be enough even if there were only
  // one lane per vector. The advantage of this size is that we
  // can avoid some calculations to figure out offsets, the
  // disadvantage is quite high memory use - up to eight times
  // as much as a tightly packed set of bits would consume.
  // This is less of an issue than one might think, though,
  // because the compiler may be able to optimize this memory
  // away, so it doesn't become manifest.

  static const std::size_t mask_bytes = vsize ;

private:

  HWY_ALIGN uint8_t inner [ mask_bytes ] ;

public:

// if we're not using sizeless vectors, the number of lanes is constexpr,
// which can help the compiler produce more efficient binary.

#ifdef HWY_HAVE_SCALABLE
  std::size_t L() const
  {
    return Lanes ( D() ) ;
  }
#else
  static constexpr std::size_t L()
  {
    return Lanes ( D() ) ;
  }
#endif

  // direct access to the data for cheating

  uint8_t * data()
  {
    return inner ;
  }

  const uint8_t * data() const
  {
    return inner ;
  }

  // The 'underlying' hardware mask type. This is entirely determined
  // by the tag type, D, of the vector this mask pertains to.

  typedef hn::Mask < D > vmask_type ;
  
  // access to the memory 'as' masks
  // we load the mask from the memory position which would apply
  // if there were only one lane per vector. This will certainly
  // be a valid choice, and as long as we're consistent and don't
  // mind wasting some space, it's okay. We might even store one
  // byte per lane. The gaps between i/o positions simply remain
  // unused and even undefined - we 'waste' memory for the sake of
  // making access code as efficient as possible.

  vmask_type yield ( const std::size_t & i ) const
  {
    return hn::LoadMaskBits( D() , inner + i * L() ) ;
  }

  void take ( const std::size_t & i , const vmask_type & rhs )
  {
    hn::StoreMaskBits( D() , rhs , inner + i * L() ) ;
  }

  // SaveToBytes 'offloads' the mask to memory holding uint8_t,
  // so that each bit in the 'flattened' mask corresponds to
  // one byte in the memory. true mask bits set the corresponding
  // memory byte to 0xFF, false mask bits set it to 0x00.

  void SaveToBytes ( uint8_t * p_trg ) const
  {
    std::size_t n_lanes = Lanes ( D() ) ;
    std::size_t n_mask = vsize / n_lanes ;
    for ( std::size_t i = 0 ; i < n_mask ; i++ )
    {
      std::size_t ofs = i * n_lanes ;
      uint8_t bit = 1 ;
      for ( std::size_t k = 0 ; k < n_lanes ; k++ )
      {
        std::size_t byte = k / 8 ;
        if ( inner [ ofs + byte ] & bit )
          *p_trg = 0xff ;
        else
          *p_trg = 0x00 ;
        ++p_trg ;
        bit <<= 1 ;
        if ( bit == 0 )
          bit = 1 ;
      }
    }
  }

  // reverse operation: this loads the mask from bytes in memory.

  void LoadFromBytes ( const uint8_t * p_trg )
  {
    std::size_t n_lanes = Lanes ( D() ) ;
    std::size_t n_mask = vsize / n_lanes ;
    for ( std::size_t i = 0 ; i < n_mask ; i++ )
    {
      std::size_t ofs = i * n_lanes ;
      uint8_t bit = 1 ;
      for ( std::size_t k = 0 ; k < n_lanes ; k++ )
      {
        std::size_t byte = k / 8 ;
        if ( *p_trg )
          inner [ ofs + byte ] |= bit ;
        else
          inner [ ofs + byte ] &= ~bit ;
        ++p_trg ;
        bit <<= 1 ;
        if ( bit == 0 )
          bit = 1 ;
      }
    }
  }

  // transfer moves masking information from one mchunk_t to another.
  // If both mchunk_t have compatible backing memory, this routine is
  // futile - the operation can be achieved by simply copying the
  // backing memory (inner) - but otherwise, using this routine works
  // like first using SaveToBytes, then LoadFromBytes - but it does so
  // without needing the buffer.

  template < typename D1 , typename D2 >
  void transfer ( const mchunk_t < D1 , vsize > & in_mask ,
                        mchunk_t < D2 , vsize > & out_mask )
  {
    const std::size_t in_n_lanes = Lanes ( D1() ) ;
    const std::size_t out_n_lanes = Lanes ( D2() ) ;
    std::size_t in_m = 0 ;
    std::size_t out_m = 0 ;
    std::size_t in_ofs = 0 ;
    std::size_t out_ofs = 0 ;
    std::size_t in_l = 0 ;
    std::size_t out_l = 0 ;
    uint8_t in_bit = 1 ;
    uint8_t out_bit = 1 ;

    const uint8_t * p_in = in_mask.data() ;
    uint8_t * p_out = out_mask.data() ;

    for ( std::size_t e = 0 ; e < vsize ; e++ )
    {
      if ( p_in [ in_ofs ] & in_bit )
        p_out [ out_ofs ] |= out_bit ;
      else
        p_out [ out_ofs ] &= ~out_bit ;

      if ( ++in_l == in_n_lanes )
      {
        in_l = 0 ;
        ++in_m ;
        in_ofs = in_m * in_n_lanes ;
        in_bit = 1 ;
      }
      else
      {
        in_bit <<= 1 ;
        if ( in_bit == 0 )
        {
          ++in_ofs ;
          in_bit = 1 ;
        }
      }
      
      if ( ++out_l == out_n_lanes )
      {
        out_l = 0 ;
        ++out_m ;
        out_ofs = out_m * out_n_lanes ;
        out_bit = 1 ;
      }
      else
      {
        out_bit <<= 1 ;
        if ( out_bit == 0 )
        {
          ++out_ofs ;
          out_bit = 1 ;
        }
      }
    }
  }

  // mask construction and assignment

  mchunk_t() = default ;
  mchunk_t ( const mchunk_t & ) = default ;
  mchunk_t & operator= ( const mchunk_t & ) = default ;

  // create an all-true or all-false mask

  mchunk_t ( bool v )
  {
    vmask_type m = FirstN ( D() , v ? vsize : 0 ) ;
    for ( std::size_t n = 0 , i = 0 ; n < vsize ; ++i , n += L() )
    {
      take ( i , m ) ;
    }
  }

  // augmented assignments are defined to allow boolean arithmetic
  // with masks. The augmented assigments are subsequently used to
  // define the corresponding binary operators.

  #define OPEQ_FUNC(OP,OPFN) \
    mchunk_t & OP ( const mchunk_t & rhs ) \
    { \
      for ( std::size_t n = 0 , i = 0 ; n < vsize ; ++i , n += L() ) \
        take ( i , OPFN ( yield ( i ) , rhs.yield ( i ) ) ) ; \
      return *this ; \
    }

    OPEQ_FUNC(operator&=,hn::And)
    OPEQ_FUNC(operator|=,hn::Or)
    OPEQ_FUNC(operator^=,hn::Xor)

  #undef OPEQ_FUNC

  // assignment from another mchunk_t of the same type is covered.
  // assignment from equally-sized, but differently-typed mchunk_t
  // needs more logic: equal vsize does not mean equal use of the
  // backing memory. this is only equal if sizeof(T) is the same,
  // so there are two variants:
  // the first one is used if the backing memory has the same layout,
  // the second one if the layout differs. The dispatch is below.

private:

  template < typename D2 >
  void _assign ( const mchunk_t < D2 , vsize > & rhs , std::true_type )
  {
    // identical layout. We can simply pretend rhs is of equal type

    const auto & trhs ( reinterpret_cast < const mchunk_t & > ( rhs ) ) ;
    *this = trhs ;
  }

  template < typename D2 >
  void _assign ( const mchunk_t < D2 , vsize > & rhs , std::false_type )
  {
    // different layout. This requires 'transfer' and is potentially
    // slow, so best avoided - but it provides the logic to make
    // objects of equal vsize interoperable.

    transfer ( rhs , *this ) ;
  }

public:

  // assignment from an mchunk_t which represents masks of a different
  // type. This top-level routine checks whether the masks pertain
  // to equally-sized data types, in which case the 'backing' memory
  // has identical layout. It then dispatches to the appropriate
  // variant of _assign, above

  template < typename D2 >
  mchunk_t & operator= ( const mchunk_t < D2 , vsize > & rhs )
  {
    typedef typename
      std::conditional < sizeof ( T ) == sizeof ( hn::TFromD < D2 > ) ,
                         std::true_type ,
                         std::false_type > :: type eq_t ;

    _assign ( rhs , eq_t() ) ;
    return *this ;
  }

  // We use the operator= template above to produce a corresponding c'tor

  template < typename D2 >
  mchunk_t ( const mchunk_t < D2 , vsize > & rhs )
  {
    *this = rhs ;
  }

  // next we have the binary operators, which delegate to the
  // augmented assignments

  #define OP_FUNC(OPFUNC,OPEQ) \
    template < typename D2 > \
    mchunk_t OPFUNC ( const mchunk_t < D2 , vsize > & rhs ) const \
    { \
      mchunk_t help ( *this ) ; \
      help OPEQ rhs ; \
      return help ; \
    }

  OP_FUNC(operator&,&=)
  OP_FUNC(operator|,|=)
  OP_FUNC(operator^,^=)
  OP_FUNC(operator&&,&=)
  OP_FUNC(operator||,|=)

  #undef OP_FUNC

  // the only unary operator for masks is the inversion, user may
  // use unary ! or ~.

  #define OP_FUNC(OPFUNC,OP) \
    mchunk_t OPFUNC() const \
    { \
      mchunk_t help ; \
      for ( std::size_t n = 0 , i = 0 ; n < vsize ; ++i , n += L() ) \
        help.take ( i , OP ( yield ( i ) ) ) ; \
      return help ; \
    }

  OP_FUNC(operator!,hn::Not)
  OP_FUNC(operator~,hn::Not)

  #undef OP_FUNC

  // finally, reductions for masks.

  bool none_of() const
  {
    vmask_type help ;
    bool result = true ;
    for ( std::size_t n = 0 , i = 0 ; n < vsize ; ++i , n += L() )
    {
      help = yield ( i ) ;
      result &= hn::AllFalse ( D() , help ) ;
    }
    return result ;
  }

  bool any_of() const
  {
    return ! none_of() ;
  }

  bool all_of() const
  {
    vmask_type help ;
    bool result = true ;
    for ( std::size_t n = 0 , i = 0 ; n < vsize ; ++i , n += L() )
    {
      help = yield ( i ) ;
      result &= hn::AllTrue ( D() , help ) ;
    }
    return result ;
  }

  // echo the mask to a std::ostream

  friend std::ostream & operator<< ( std::ostream & osr ,
                                     mchunk_t it )
  {
    uint8_t buffer [ vsize ] ;
    it.SaveToBytes ( buffer ) ;
    osr << "(" ;
    for ( std::size_t i = 0 ; i < vsize ; i++ )
      osr << ( buffer [ i ] ? "1" : "0" ) ;
    osr << ")" ;
    return osr ;
  }

} ;

// free versions of reductions. It's often necessary to determine whether
// a mask is completely full or empty, or has at least some non-false
// members. The code might be extended to test arbitrary vectors rather
// than only masks. As it stands, to apply the functions to an
// arbitrary vector, use a construct like 'any_of ( v == 0 )' instead of
// 'any_of ( v )'.

template < typename D , std::size_t N >
bool any_of ( const mchunk_t<D,N> & arg )
{
  return arg.any_of() ;
}

template < typename D , std::size_t N >
bool all_of ( const mchunk_t<D,N> & arg )
{
  return arg.all_of() ;
}

template < typename D , std::size_t N >
bool none_of ( const mchunk_t<D,N> & arg )
{
  return arg.none_of() ;
}

/// class template simd_t provides a fixed-size container type for
/// small-ish sets of fundamentals which are stored in a POD C vector.
/// This implementation uses highway to code the loops more efficiently.
/// It mimicks Vc::SimdArray, just like gen_simd_type does, and
/// The code is derived from zimt::simd_array, changing the workhorse
/// code from simple loops to the use of highway functions.
/// The resulting type, with it's 'container-typical' interface, slots
/// in well with the higher-level constructs used in zimt/lux and,
/// at the same time, 'contains' the SIMD implementation in this class,
/// so that it's use doesn't need to be known outside.
/// As an arithmetic type, simd_t provides many mathematical operators
/// and some functions - most of them are realized by calling corresponding
/// highway functions, but some (still) rely on loops, either because they
/// aren't performance-critical or because there is no highway code to be
/// had for the purpose. Some methods are (currently) exclusive to this
/// class, but may be ported to other SIMD interface classes; apart from
/// the original 'goading' class gen_simd_type, there is also an
/// implementation using std::simd in pv/zimt/std_simd_type.h
/// The lane count for a simd_t in this body of code should be a
/// power of two, and it should be at least as large as the hardware lane
/// count of the smallest fundamental used in vectorized form. To cover
/// all eventualities, the hardware lane count of a vector of unsigned
/// char (uint8_t) is a good choice. This choice is to avoid that simd_t
/// objects of small T remain partly empty when a given small vsize is
/// chosen to cater for vectors with larger T. At times, this will lead
/// to overly high register pressure, and the overall performance may
/// benefit from allowing partially filled simd_t via a smaller vsize,
/// which is feasible because simd_t uses highway vectors with
/// CappedTag.

// forward declaration of class template simd_t

template < typename _value_type ,
           std::size_t _vsize >
struct HWY_ALIGN simd_t ;

// next we have conversion functions. I decided to code them as free
// functions, which makes formulation easier because both source and
// target can be template arguments. Conversion with highway is quite
// involved because there is no generic definition, instead there are
// a bunch of conversions which highway can do (as documented in the
// quick reference) but the set is not complete. So we have to code
// so that the available ones will be used and the other ones are
// relized by 'goading' - going over the backing arrays with a loop.
// To avoid repetition, we use macros for three cases: conversion with
// ConvertTo, DemoteTo and PromoteTo. All conversions which are not
// explicitly coded will fall back to goading.

// catch-all template for conversions, uses goading

template < typename src_t , typename trg_t , std::size_t vsize >
void convert ( const simd_t < src_t , vsize > & src ,
                     simd_t < trg_t , vsize > & trg )
{
  auto p_src = src.data() ;
  auto p_trg = trg.data() ;
  for ( std::size_t i = 0 ; i < vsize ; i++ )
    p_trg[i] = p_src[i] ;
}

// the remainder of the conversions uses three highway functions
// only, so we write macros for the three types of conversion to
// make it easier to see the big picture. We use CV_PROMOTE here
// so as not to clash with the macro PROMOTE in common.h

#define CV_PROMOTE(SRC,TRG) \
template < std::size_t vsize > \
void convert ( const simd_t < SRC , vsize > & src , \
                     simd_t < TRG , vsize > & trg ) \
{ \
  typedef hn::CappedTag < TRG , vsize > D ; \
  typedef hn::Rebind < SRC , D > ud_t ; \
\
  for ( std::size_t n = 0 , i = 0 ; n < vsize ; ++i , n += trg.L() ) \
  { \
    auto underfilled = src.template dt_yield < ud_t > ( i ) ; \
    auto promoted = hn::PromoteTo ( D() , underfilled ) ; \
    trg.take ( i , promoted ) ; \
  } \
}

#define CV_CONVERT(SRC,TRG) \
template < std::size_t vsize > \
void convert ( const simd_t < SRC , vsize > & src , \
                     simd_t < TRG , vsize > & trg ) \
{ \
  for ( std::size_t n = 0 , i = 0 ; n < vsize ; ++i , n += trg.L() ) \
    trg.take ( i , hn::ConvertTo ( hn::CappedTag < TRG , vsize > () , \
                                   src.yield ( i ) ) ) ; \
}

#define CV_DEMOTE(SRC,TRG) \
template < std::size_t vsize > \
void convert ( const simd_t < SRC , vsize > & src , \
                     simd_t < TRG , vsize > & trg ) \
{ \
  typedef hn::CappedTag < SRC , vsize > X ; \
  typedef hn::Rebind < TRG , X > ud_t ; \
 \
  for ( std::size_t n = 0 , i = 0 ; n < vsize ; ++i , n += hn::Lanes ( ud_t() ) ) \
  { \
    const auto demoted = hn::DemoteTo ( ud_t() , src.yield ( i ) ) ; \
    trg.template dt_take < ud_t > ( i , demoted ) ; \
  } \
}

// I've 'scraped' the highway quick reference for the possible
// type combinations to be used with PromoteTo, DemoteTo and
// ConvertTo. I ignore half floats for now. Here's the 'scrape':

// PromoteTo: (bf16,f32) (f16,f32) (f32,f64) (i16,i32) (i32,i64)
//            (i8,i16) (i8,i32) (u16,i32) (u16,u32) (u32,u64)
//            (u8,i16), (u8,i32) (u8,u16) (u8,u32)
//
// DemoteTo: (f32,bf16) (f32,f16) (f64,f32) (f64,i32) (i16,i8)
//           (i16,u8) (i32,i16) (i32,i8) (i32,u16) (i32,u8)
//           (i64,i16) (i64,i32) (i64,i8) (i64,u16) (i64,u32)
//           (i64,u8) (u16,i8) (u16,u8) (u32,i16) (u32,i8)
//           (u32,u16) (u32,u8) (u64,i16) (u64,i32) (u64,i8)
//           (u64,u16) (u64,u32) (u64,u8)
//
// ConvertTo: (f32,i32) (f64,i64) (i32,f32) (i64,f64)

// Converted to my macros, here are the specialized functions:

// PromoteTo

// PROMOTE(bfloat16,float)
// PROMOTE(float16,float)
CV_PROMOTE(float,double)
CV_PROMOTE(short,int)
CV_PROMOTE(int,long)
CV_PROMOTE(signed char,short)
CV_PROMOTE(signed char,int)
CV_PROMOTE(unsigned short,int)
CV_PROMOTE(unsigned short,unsigned int)
CV_PROMOTE(unsigned int,unsigned long)
CV_PROMOTE(unsigned char,short)
CV_PROMOTE(unsigned char,int)
CV_PROMOTE(unsigned char,unsigned short)
CV_PROMOTE(unsigned char,unsigned int)

// DemoteTo

// DEMOTE(float,bfloat16)
// DEMOTE(float,float16)
CV_DEMOTE(double,float)
CV_DEMOTE(double,int)
CV_DEMOTE(short,signed char)
CV_DEMOTE(short,unsigned char)
CV_DEMOTE(int,short)
CV_DEMOTE(int,signed char)
CV_DEMOTE(int,unsigned short)
CV_DEMOTE(int,unsigned char)
CV_DEMOTE(long,short)
CV_DEMOTE(long,int)
CV_DEMOTE(long,signed char)
CV_DEMOTE(long,unsigned short)
CV_DEMOTE(long,unsigned int)
CV_DEMOTE(long,unsigned char)
CV_DEMOTE(unsigned short,signed char)
CV_DEMOTE(unsigned short,unsigned char)
CV_DEMOTE(unsigned int,short)
CV_DEMOTE(unsigned int,signed char)
CV_DEMOTE(unsigned int,unsigned short)
CV_DEMOTE(unsigned int,unsigned char)
CV_DEMOTE(unsigned long,short)
CV_DEMOTE(unsigned long,int)
CV_DEMOTE(unsigned long,signed char)
CV_DEMOTE(unsigned long,unsigned short)
CV_DEMOTE(unsigned long,unsigned int)
CV_DEMOTE(unsigned long,unsigned char)

// ConvertTo

CV_CONVERT(float,int)
CV_CONVERT(double,long)
CV_CONVERT(int,float)
CV_CONVERT(long,double)

#undef CV_PROMOTE
#undef CV_DEMOTE
#undef CV_CONVERT

// template for conversions from double to other types which are
// not covered by the overloads above. For now we ignore half floats
// and code as if float and double were the only fp types.
// These conversions use three steps, which may add up to more time
// than 'goading' would use, so this should be TODO tested.

template < typename T , std::size_t vsize >
void convert ( const simd_t < double , vsize > & src ,
                     simd_t < T , vsize > & trg )
{
  static_assert ( std::is_integral < T > :: value , "int only...!" ) ;
  simd_t < long , vsize > l_src = src ;
  simd_t < int , vsize > i_src = l_src ;
  convert ( i_src , trg ) ;
}

template < typename T , std::size_t vsize >
void convert ( const simd_t < T , vsize > & src ,
                     simd_t < double , vsize > & trg )
{
  static_assert ( std::is_integral < T > :: value , "int only...!" ) ;
  simd_t < int , vsize > i_src = src ;
  simd_t < long , vsize > l_src = i_src ;
  convert ( l_src , trg ) ;
}

// conversion to and from gen_simd_type of equal T

template < typename T , std::size_t vsize >
void convert ( const gen_simd_type < T , vsize > & src ,
                     simd_t < T , vsize > & trg )
{
  src.store ( trg.data() ) ;
}

template < typename T , std::size_t vsize >
void convert ( const simd_t < T , vsize > & src ,
                     gen_simd_type < T , vsize > & trg )
{
  trg.load ( src.data() ) ;
}

// conversion to and from gen_simd_type of different T
// This uses goading, because we can't be sure that src_t can be
// handled by highway.

template < typename src_t , typename trg_t , std::size_t vsize >
void convert ( const gen_simd_type < src_t , vsize > & src ,
                     simd_t < trg_t , vsize > & trg )
{
  auto p_trg = trg.data() ;
  for ( std::size_t i = 0 ; i < vsize ; i++ )
    p_trg[i] = src[i] ;
}

template < typename src_t , typename trg_t , std::size_t vsize >
void convert ( const simd_t < src_t , vsize > & src ,
                gen_simd_type < trg_t , vsize > & trg )
{
  auto p_src = src.data() ;
  for ( std::size_t i = 0 ; i < vsize ; i++ )
    trg[i] = p_src[i] ;
}

// now comes the template class simd_t which implements the core
// of the functionality, the SIMD data type 'standing in' for
// gen_simd_type when USE_HWY is defined.

template < typename _value_type ,
           std::size_t _vsize >
struct HWY_ALIGN simd_t
: public simd_tag < _value_type , _vsize , HWY >
{
  typedef simd_tag < _value_type , _vsize , HWY > tag_t ;
  using typename tag_t::value_type ;
  using tag_t::vsize ;
  using tag_t::backend ;

  typedef std::size_t size_type ;

  typedef value_type T ;

  // A typical choice of vsize would be 'as many as there are
  // bytes in a hardware vector'. This insures that all simd_t
  // are interoperable without having to use half- or quarter-filled
  // vectors for size compatibility.

  static const int vbytes = sizeof ( value_type ) * vsize ;

  // we make sure vsize is a power of two - simd_traits in vector.h
  // does route non-power-of-two sizes to gen_simd_type, but
  // user code may not use simd_traits. Best to be safe!

  static_assert ( ( vsize & ( vsize - 1 ) ) == 0 ,
    "only use powers of two as lane count for highway-based simd_t" ) ;

  // we use CappedTag to make sure that interfacing with memory
  // will never exceed the bounds of the 'backing' memory. This
  // may waste some register space: if T were, e.g. uint8_t and
  // vsize 32, while a register could hold 64 bytes, half the
  // register would remain empty.

  typedef hn::CappedTag < value_type , vsize > D ;
  typedef typename hn::Vec < D > vec_t ;

  // just to make sure the class does the right thing

  void info() const
  {
    std::cout << "value_type has "
              << sizeof(value_type) << " bytes" << std::endl ;
    std::cout << "simd_t has  "
              << sizeof(simd_t) << " bytes" << std::endl ;
    std::cout << "HWY_MAX_BYTES: "
              << HWY_MAX_BYTES << " bytes" << std::endl ;
    std::cout << "Lane count is  "
              << L() << std::endl ;    
    std::cout << "MaxLanes is    "
              << MaxLanes(D()) << std::endl ;    
    std::cout << "vsize is       "
              << vsize << " value_type" << std::endl ;
  }

#ifdef HWY_HAVE_SCALABLE

  std::size_t L() const
  {
    return Lanes ( D() ) ;
  }

#else

  static constexpr std::size_t L()
  {
    return Lanes ( D() ) ;
  }
  
#endif

  // derive types used for masks and index vectors.

  typedef hn::Vec < hn::RebindToSigned < D > > hw_index_type ;
  typedef hn::DFromV < hw_index_type > DI ;
  typedef hn::TFromD < DI > TI ;
  typedef simd_t < TI , vsize > index_type ;

  static_assert ( std::is_same < hw_index_type ,
                                 typename index_type::vec_t
                               > :: value ,
                  "index type mismatch" ) ;

  // definition of the type for masks. Since simd_t holds the
  // equivalent of potentially several vectors, the mask type has
  // to hold the equivalent of as many masks.

  typedef mchunk_t < D , vsize > mask_type ;
  typedef mchunk_t < D , vsize > MaskType ;

private:

  // the 'backing' memory: storage of data is in a simple C array.
  // This array is private, and the only access to it is via member
  // functions.

  HWY_ALIGN value_type inner [ vsize ] ;

public:

  // provide the size as a constexpr. This is possible because
  // the size is indeed known at compile time, even though the
  // underlying vectors may be sizeless - possibly capped.

  static constexpr size_type size()
  {
    return vsize ;
  }

  // 'back door' for cheating

  T * data()
  {
    return inner ;
  }

  const T * data() const
  {
    return inner ;
  }

  // interface to the 'backing' memory 'as' highway vectors.
  // This is a key function. I have opted to use operator[] for access
  // to individual lanes, in keeping with standard container semantics,
  // And to avoid the vector types 'leaking out'. If user code wants
  // to use the vectorized interface, it should do so via yield and
  // take, as does the code inside this class.


  vec_t yield ( const std::size_t & i ) const
  {
    return hn::Load ( D() , inner + i * L() ) ;
  }

  void take ( const std::size_t & i , const vec_t & rhs )
  {
    hn::Store ( rhs , D() , inner + i * L() ) ;
  }

  // dt_yield and dt_take are variants which are used to load
  // and store under-filled vectors, which happens during type
  // conversions to/from simd_types with a differently-sized
  // T, involving promotion/demotion.

  template < typename DT >
  hn::Vec < DT > dt_yield ( const std::size_t & i ) const
  {
    return hn::Load ( DT() , inner + i * Lanes ( DT() ) ) ;
  }

  template < typename DT >
  void dt_take ( const std::size_t & i , const hn::Vec < DT > & rhs )
  {
    hn::Store ( rhs , DT() , inner + i * Lanes ( DT() ) ) ;
  }

  // broadcast functions to help with functionality which is not
  // available ready-made, and to help rolling out vector code to
  // chunks, which need the operation repeated over the set of
  // constituent vectors. The functions ending on _vf are vector
  // functions and applied to the constituent vectors, using the
  // access functions yield and take.
  // broadcast functions ending in plain _f are scalar functions and
  // they are rolled out over the array of value_type, 'inner'. value_typehis type of
  // operation may well be recognized by the optimizer and result
  // in 'proper' SIMD code.
  // In all cases, *this is the receiving end of the operation and
  // contains the result of the repeated execution of the function
  // passed to 'broadcast'.

  typedef std::function < value_type() > gen_f ;
  typedef std::function < value_type ( const value_type & ) > mod_f ;
  typedef std::function < value_type ( const value_type & , const value_type & ) > bin_f ;

  simd_t & broadcast ( gen_f f )
  {
    for ( std::size_t i = 0 ; i < size() ; i++ )
    {
      inner[i] = f() ;
    }
    return *this ;
  }

  simd_t & broadcast ( mod_f f )
  {
    for ( std::size_t i = 0 ; i < size() ; i++ )
    {
      inner[i] = f ( inner[i] ) ;
    }
    return *this ;
  }

  simd_t & broadcast ( bin_f f , const simd_t & rhs )
  {
    for ( std::size_t i = 0 ; i < size() ; i++ )
    {
      inner[i] = f ( inner[i] , rhs.inner[i] ) ;
    }
    return *this ;
  }

  typedef std::function < vec_t() > gen_vf ;
  typedef std::function < vec_t ( const vec_t & ) > mod_vf ;
  typedef std::function < vec_t ( const vec_t & , const vec_t & ) > bin_vf ;

  // broadcast a vector generator function
  
  simd_t & vbroadcast ( gen_vf f )
  {
    for ( std::size_t n = 0 , i = 0 ; n < vsize ; ++i , n += L() )
      take ( i , f() ) ;
    return *this ;
  }

  // broadcast a vector modulator

  simd_t & vbroadcast ( mod_vf f )
  {
    for ( std::size_t n = 0 , i = 0 ; n < vsize ; ++i , n += L() )
      take ( i , f ( yield ( i ) ) ) ;
    return *this ;
  }

  // broadcast a vector binary function

  simd_t & vbroadcast ( bin_vf f , const simd_t & rhs )
  {
    for ( std::size_t n = 0 , i = 0 ; n < vsize ; ++i , n += L() )
      take ( i , f ( yield ( i ) , rhs.yield ( i ) ) ) ;
    return *this ;
  }

  // c'tor from T, using hn::Set to provide a vector as initializer

  simd_t ( const T & x )
  {
    for ( std::size_t n = 0 , i = 0 ; n < vsize ; ++i , n += L() )
      take ( i , hn::Set ( D() , x ) ) ;
  }

  simd_t() = default ;

  // assignment from another chunk or a T

  simd_t & operator= ( const simd_t & rhs )
  {
    for ( std::size_t n = 0 , i = 0 ; n < vsize ; ++i , n += L() )
      take ( i , rhs.yield ( i ) ) ;
    return *this ;
  }

  simd_t & operator= ( const simd_t && rhs )
  {
    for ( std::size_t n = 0 , i = 0 ; n < vsize ; ++i , n += L() )
      take ( i , rhs.yield ( i ) ) ;
    return *this ;
  }

  simd_t & operator= ( const T & rhs )
  {
    for ( std::size_t n = 0 , i = 0 ; n < vsize ; ++i , n += L() )
      take ( i , hn::Set ( D() , rhs ) ) ;
    return *this ;
  }

  simd_t ( const simd_t & x )
  {
    *this = x ;
  }

  simd_t ( const simd_t && x )
  {
    *this = x ;
  }

  // operator[] is mapped to ordinary element access to the backing
  // memory. It's assumed that user code will avoid using it for
  // performance-critical code, but it's 'nice to have' and easily
  // coded:

  value_type & operator[] ( const size_type & i )
  {
    return inner[i] ;
  }

  value_type operator[] ( const size_type & i ) const
  {
    return inner[i] ;
  }

  // for conversions between different simd_types and to and from
  // gen_simd_type, we 'break out' to free functions.

  // conversion from one simd_t to another

  template < typename U >
  simd_t & operator= ( const simd_t < U , vsize > & rhs )
  {
    convert ( rhs , *this ) ;
    return *this ;
  }

  // assignment from a gen_simd_type on the rhs

  template < typename U >
  simd_t & operator= ( const gen_simd_type < U , vsize > & rhs )
  {
    convert ( rhs , *this ) ;
    return *this ;
  }

  // conversion to a gen_simd_type
  // TODO: is this ever called?

  template < typename U >
  operator gen_simd_type < U , vsize > ()
  {
    gen_simd_type < U , vsize > result ;
    convert ( *this , result ) ;
    return result ;
  }

  template < typename U >
  simd_t ( const simd_t < U , vsize > & rhs )
  {
    *this = rhs ;
  }

  template < typename U >
  simd_t ( const gen_simd_type < U , vsize > & rhs )
  {
    *this = rhs ;
  }

  // because all simd_t objects are of a distinct size
  // explicitly coded in template arg vsize, we can initialize
  // from an initializer_list. This is probably not very fast,
  // but it's nice to have for experimentation.

  simd_t ( const std::initializer_list < value_type > & rhs )
  {
    assert ( rhs.size() == vsize ) ; // TODO: prefer constexpr
    value_type * trg = inner ;
    for ( const auto & src : rhs )
      *trg++ = src ;
  }

  // implement copysign, isnegative, isfinite, isnan, setQnan, setZero
  // These functions are present in Vc::SimdArray. They are not currently
  // provided in other simd_t variants, I used them to code a direct
  // port of Vc's atan2 function.
  // For lux, this function is performance-critical, so I made several
  // attempts at porting it. A 'straight' port of the Vc code is possible
  // with these added functions, but it operates at the simd_t level,
  // which does not seem to optimize well. So I 'translated' the code
  // to use highway at the vec_t level - see hwy_atan2.h

  static simd_t copysign ( simd_t value ,
                              const simd_t & sign_source )
  {
    for ( std::size_t n = 0 , i = 0 ; n < vsize ; ++i , n += value.L() )
    {
      value.take ( i , hn::CopySign ( value.yield ( i ) ,
                                      sign_source.yield ( i ) ) ) ;
    }
    return value ;
  }

  // there might be a more efficient way to do this, but for now:

  static mask_type isnegative ( const simd_t & rhs )
  {
    return ( rhs < value_type(0) ) ;
  }

  static mask_type isfinite ( const simd_t & rhs )
  {
    mask_type result ;
    for ( std::size_t n = 0 , i = 0 ; n < vsize ; ++i , n += result.L() )
    {
      result.take ( i , hn::IsFinite ( rhs.yield ( i ) ) ) ;
    }
    return result ;
  }

  static mask_type isnan ( const simd_t & rhs )
  {
    mask_type result ;
    for ( std::size_t n = 0 , i = 0 ; n < vsize ; ++i , n += result.L() )
    {
      result.take ( i , hn::IsNaN ( rhs.yield ( i ) ) ) ;
    }
    return result ;
  }

  simd_t & setQnan ( const mask_type & m )
  {
    for ( std::size_t n = 0 , i = 0 ; n < vsize ; ++i , n += L() )
    {
      take ( i , IfThenElse ( m.yield ( i ) ,
                              NaN ( D() ) ,
                              yield ( i ) ) ) ;
    }
    return *this ;
  }

  simd_t & setZero ( const mask_type & m )
  {
    for ( std::size_t n = 0 , i = 0 ; n < vsize ; ++i , n += L() )
    {
      take ( i , IfThenElse ( m.yield ( i ) ,
                              Zero ( D() ) ,
                              yield ( i ) ) ) ;
    }
    return *this ;
  }

  // produce a simd_t filled with T rising from zero

  static const simd_t iota()
  {
    simd_t result ;
    auto v = hn::Iota ( D() , 0 ) ;
    for ( std::size_t n = 0 , i = 0 ; n < vsize ; ++i , n += result.L() )
    {
      result.take ( i , v ) ;
      v = hn::Add ( v , hn::Set ( D() , T ( Lanes ( D() ) ) ) ) ;
    }
    return result ;
  }

  // mimick Vc's IndexesFromZero. This function produces an index
  // vector filled with indexes starting with zero. Because hwy
  // uses as many bits for the index as value_type has, we make
  // sure the indexes can fit.

  static const index_type IndexesFromZero()
  {
    typedef typename index_type::value_type IT ;
    static const IT ceiling = std::numeric_limits < IT > :: max() ;
    static_assert ( ( vsize - 1 ) <= std::size_t ( ceiling ) ,
                    "value_type too small" ) ;

    return index_type::iota() ;
  }

  // variant which starts from a different starting point and
  // optionally uses steps other than one. This is handy to
  // generate gather/scatter indexes to access strided data.
  // to avoid trouble, we look at the maximum index we can produce,
  // and add an assertion to make sure the indexes we expect will
  // fit the range.

  static const index_type IndexesFrom ( const std::size_t & start ,
                                        const std::size_t & step = 1 )
  {
    typedef typename index_type::value_type IT ;
    static const IT ceiling = std::numeric_limits < IT > :: max() ;

    assert (    start + ( vsize - 1 ) * step
             <= std::size_t ( ceiling ) ) ;

    return ( index_type::iota() * IT(step) ) + IT(start) ;
  }

  // functions Zero and One produce simd_t objects filled with
  // 0, or 1, respectively

  static const simd_t Zero()
  {
    return simd_t ( value_type ( 0 ) ) ;
  }

  static const simd_t One()
  {
    return simd_t ( value_type ( 1 ) ) ;
  }

  // echo the vector to a std::ostream, read it from an istream
  // this also goes 'over the memory' because it's not deemed
  // performance-critical

  friend std::ostream & operator<< ( std::ostream & osr ,
                                     simd_t it )
  {
    std::size_t l = it.L() ;
    osr << "(* " ;
    for ( size_type i = 0 ; i < vsize - 1 ; i++ )
      osr << it [ i ] << ( i % l == l - 1 ? " | " : ", " ) ;
    osr << it [ vsize - 1 ] << " *)" ;
    return osr ;
  }

  friend std::istream & operator>> ( std::istream & isr ,
                                     simd_t it )
  {
    for ( size_type i = 0 ; i < vsize ; i++ )
      isr >> it [ i ] ;
    return isr ;
  }

  // memory access functions, which load and store vector data.
  // We start out with functions transporting data from memory into
  // the simd_t. Some of these operations have corresponding
  // c'tors which use the member function to initialize 'inner'.

  // load from unaligned memory. We're defensive here, if the
  // calling code is sure that the memory is appropriately aligned,
  // it can use load_aligned (below)

  void load ( const value_type * const & p_src )
  {
    for ( std::size_t n = 0 , i = 0 ; n < vsize ; ++i , n += L() )
      take ( i , hn::LoadU ( D() , p_src + i * L() ) ) ;
  }

  void load_aligned ( const value_type * const & p_src )
  {
    for ( std::size_t n = 0 , i = 0 ; n < vsize ; ++i , n += L() )
      take ( i , hn::Load ( D() , p_src + i * L() ) ) ;
  }

  // now the reverse operations, storing to memory

  void store ( value_type * const  p_trg ) const
  {
    for ( std::size_t n = 0 , i = 0 ; n < vsize ; ++i , n += L() )
      hn::StoreU ( yield ( i ) , D() , p_trg + i * Lanes ( D() ) ) ;
  }

  void store_aligned ( value_type * const & p_trg ) const
  {
    for ( std::size_t n = 0 , i = 0 ; n < vsize ; ++i , n += L() )
      hn::Store ( yield ( i ) , D() , p_trg + i * Lanes ( D() ) ) ;
  }

// on AVX512, there do not seem to be any gather/scatter operations
// for short or byte values, so I use goading to implement them.

#ifdef FLV_AVX512f

  // to gather larger-than-short data, use hwy g/s

  void _gather ( const value_type * const & p_src ,
                 const index_type & indexes ,
                 std::false_type )
  {
    for ( std::size_t n = 0 , i = 0 ; n < vsize ; ++i , n += L() )
      take ( i , hn::GatherIndex ( D() , p_src ,
             indexes.yield ( i ) ) ) ;
  }

  void _scatter ( value_type * const & p_trg ,
                  const index_type & indexes ,
                  std::false_type ) const
  {
    for ( std::size_t n = 0 , i = 0 ; n < vsize ; ++i , n += L() )
      hn::ScatterIndex ( yield ( i ) , D() , p_trg ,
                         indexes.yield ( i ) ) ;
  }

  // to gather/scatter shorts or bytes, use goading

  void _gather ( const value_type * const & p_src ,
                 const index_type & indexes ,
                 std::true_type )
  {
    for ( std::size_t i = 0 ; i < vsize ; i++ )
      inner [ i ] = p_src [ indexes [ i ] ] ;
  }

  void _scatter ( value_type * const & p_trg ,
                  const index_type & indexes ,
                  std::true_type ) const
  {
    for ( std::size_t i = 0 ; i < vsize ; i++ )
      p_trg [ indexes [ i ] ] = inner [ i ] ;
  }

  // these are the dispatching routines testing for small value_type

  void gather ( const value_type * const & p_src ,
                const index_type & indexes )
  {
    typedef std::integral_constant
              < bool , sizeof ( value_type ) <= 2 > is_small_t ;

    _gather ( p_src , indexes , std::false_type() ) ; // is_small_t() ) ;
  }

  void scatter ( value_type * const & p_trg ,
                 const index_type & indexes ) const
  {
    typedef std::integral_constant
              < bool , sizeof ( value_type ) <= 2 > is_small_t ;

    _scatter ( p_trg , indexes , is_small_t() ) ;
  }

#else // #ifdef FLV_AVX512f

  // other ISAs seem to have no problem scattering all value_types

  // gather with 'proper' index type, which is derived by type inference
  // from the tag type D which defines the underlying vector type. This
  // type is quite specific in highway: it is a vector of signed integers
  // with the same number of bits as the fundamental type of the vector
  // that is indexed. Calling code should obtain index_type from simd_t
  // (it's public), but it may provide indexes in other forms, which are
  // routed to the template further down.

  void gather ( const value_type * const & p_src ,
                const index_type & indexes )
  {
    for ( std::size_t n = 0 , i = 0 ; n < vsize ; ++i , n += L() )
      take ( i , hn::GatherIndex ( D() , p_src ,
             indexes.yield ( i ) ) ) ;
  }

  void scatter ( value_type * const & p_trg ,
                 const index_type & indexes ) const
  {
    for ( std::size_t n = 0 , i = 0 ; n < vsize ; ++i , n += L() )
      hn::ScatterIndex ( yield ( i ) , D() , p_trg ,
                         indexes.yield ( i ) ) ;
  }

#endif

  // goading implementation with arbitrary index type. This will be used
  // if the indexes aren't precisely index_type, but some other entity
  // providing operator[]. The loop construct may well be autovectorized,
  // but this can't be guaranteed. It's recommended to use the proper
  // index_type wherever possible.

  template < typename index_t >
  void gather ( const value_type * const & p_src ,
                const index_t & indexes )
  {
    for ( std::size_t i = 0 ; i < vsize ; i++ )
      inner [ i ] = p_src [ indexes [ i ] ] ;
  }

  template < typename index_t >
  void scatter ( value_type * const & p_trg ,
                 const index_t & indexes ) const
  {
    for ( std::size_t i = 0 ; i < vsize ; i++ )
      p_trg [ indexes [ i ] ] = inner [ i ] ;
  }

  // c'tor from pointer and indexes, uses gather

  template < typename index_t >
  simd_t ( const value_type * const & p_src ,
              const index_t & indexes )
  {
    gather ( p_src , indexes ) ;
  }

  // 'regular' gather and scatter, accessing strided memory so that the
  // first address visited is p_src/p_trg, and successive addresses are
  // 'step' apart - in units of T. IndexesFrom generates the 'proper'
  // index_type for 'real' g/s, so if the ISA supports it the operation
  // should be reasonably fast. It's used to implement de/interleaving
  // for scenarios which are not covered by the Load/StoreInterleaved
  // family of functions.

  void rgather ( const value_type * const & p_src ,
                 const std::size_t & step )
  {
    if ( step == 1 )
    {
      load ( p_src ) ;
    }
    else
    {
      auto indexes = IndexesFrom ( 0 , step ) ;
      gather ( p_src , indexes ) ;
    }
  }

  void rscatter ( value_type * const & p_trg ,
                  const std::size_t & step ) const
  {
    if ( step == 1 )
    {
      store ( p_trg ) ;
    }
    else
    {
      auto indexes = IndexesFrom ( 0 , step ) ;
      scatter ( p_trg , indexes ) ;
    }
  }

//   // use 'indexes' to perform a gather from the data held in 'inner'
//   // and return the result of the gather operation.
// 
//   template < typename index_type >
//   simd_t shuffle ( index_type indexes )
//   {
//     simd_t result ;
//     result.rgather ( inner , indexes ) ;
//     return result ;
//   }

  // apply functions from namespace hn to each vector in the simd_t
  // or to each corresponding set of vectors going up to three for fma.
  // This method works well on some platforms with some functions, but
  // altogether the results aren't satisfactory - it's more of an
  // emergency measure if proper vector code can't be had. But it's
  // a quick way to get code up and running which requires certain
  // functions, which allows for rapid prototyping and leaving the
  // proper implementation for later.

  #define BROADCAST_HWY_FUNC(FUNC,HFUNC) \
    friend simd_t FUNC ( const simd_t & arg ) \
    { \
      simd_t result ; \
      for ( std::size_t n = 0 , i = 0 ; n < vsize ; ++i , n += arg.L() ) \
        result.take ( i , hn::HFUNC ( D() , arg.yield ( i ) ) ) ; \
      return result ; \
    }

  BROADCAST_HWY_FUNC(log,Log)
  BROADCAST_HWY_FUNC(exp,Exp)
  BROADCAST_HWY_FUNC(sin,Sin)
  BROADCAST_HWY_FUNC(cos,Cos)

  // hn function not (yet) available for tan, rolling out to sin/cos

//   BROADCAST_HWY_FUNC(tan,Tan)
  
  friend simd_t tan ( const simd_t & arg )
  {
    simd_t result ;
    for ( std::size_t n = 0 , i = 0 ; n < vsize ; ++i , n += arg.L() )
      result.take ( i ,
                    hn::Div ( hn::Sin ( D() , arg.yield ( i ) ) ,
                              hn::Cos ( D() , arg.yield ( i ) ) ) ) ;
    return result ;
  }
  
  // goading version

//   friend simd_t tan ( simd_t arg )
//   {
//     static const mod_f f = [](const T & x)
//       { return T ( std::tan ( x ) ) ; } ;
//     return arg.broadcast ( f ) ;
//   }

  BROADCAST_HWY_FUNC(asin,Asin)
  BROADCAST_HWY_FUNC(acos,Acos)
  BROADCAST_HWY_FUNC(atan,Atan)

  #undef BROADCAST_HWY_FUNC

  #define BROADCAST_HWY_FUNC(FUNC,HFUNC) \
    friend simd_t FUNC ( const simd_t & arg ) \
    { \
      simd_t result ; \
      for ( std::size_t n = 0 , i = 0 ; n < vsize ; ++i , n += arg.L() ) \
        result.take ( i , hn::HFUNC ( arg.yield ( i ) ) ) ; \
      return result ; \
    }

  BROADCAST_HWY_FUNC(abs,Abs)
  BROADCAST_HWY_FUNC(trunc,Trunc)
  BROADCAST_HWY_FUNC(round,Round)
  BROADCAST_HWY_FUNC(floor,Floor)
  BROADCAST_HWY_FUNC(ceil,Ceil)
  BROADCAST_HWY_FUNC(sqrt,Sqrt)

  #undef BROADCAST_HWY_FUNC

  #define BROADCAST_HWY_FUNC2(FUNC,HFUNC) \
    friend simd_t FUNC ( const simd_t & arg1 , \
                            const simd_t & arg2 ) \
    { \
      simd_t result ; \
      for ( std::size_t n = 0 , i = 0 ; n < vsize ; ++i , n += arg1.L() ) \
        result.take ( i , hn::HFUNC ( arg1.yield ( i ) , \
                                      arg2.yield ( i ) ) ) ; \
      return result ; \
    }

  // hn function not available for atan2, pow

//   BROADCAST_HWY_FUNC2(pow,Pow)

  // implementation of atan2 relying on a highway function Atan2.
  // The implementation of that is a port from Vc, in hwy_atan2.h

  friend simd_t atan2 ( const simd_t & y , const simd_t & x )
  {
    simd_t result ;
    for ( std::size_t n = 0 , i = 0 ; n < vsize ; ++i , n += x.L() )
      result.take ( i , hn::Atan2 ( D() , y.yield(i) , x.yield(i) ) ) ;
    return result ;
  }

  // goading version

//   friend simd_t atan2 ( simd_t y , const simd_t & x )
//   {
//     static const bin_f f = [](const T & y, const T & x)
//       { return T ( std::atan2 ( y , x ) ) ; } ;
//     return y.broadcast ( f , x ) ;
//   }
  
  friend simd_t pow ( const simd_t & base , const simd_t & exponent )
  {
    simd_t result ;
    for ( std::size_t n = 0 , i = 0 ; n < vsize ; ++i , n += base.L() )
      result.take ( i ,
                    hn::Exp ( D() ,
                              hn::Mul ( exponent.yield ( i ) ,
                                        hn::Log ( D() , base.yield ( i ) ) )
                            )
                  ) ;
    return result ;
  }

#undef BROADCAST_HWY_FUNC2

  // three-argument-functions are not currently used.

  // macro used for the parameter 'CONSTRAINT' in the definitions
  // further down. Some operations are only allowed for integral types
  // or boolans. This might be enforced by enable_if, here we use a
  // static_assert with a clear error message. I found that highway
  // is quite relaxed about enforcing integer data - things like
  // Xor and Not are allowed even for floating point data and
  // presumably work on the bare bits.
  // TODO: might relax constraints by using 'std::is_convertible'

  #define INTEGRAL_ONLY \
    static_assert ( std::is_integral < value_type > :: value , \
                    "this operation is only allowed for integral types" ) ;

  // augmented assignment with a chunk as rhs and with a T as rhs.
  // The first one is actually defined wider - the rhs can be any
  // type which can yield an object suitable as rhs to OPFN, and
  // the second variant fills a vector with the scalar and uses that
  // as the rhs to OPFN. This may not be optimal in every case, one
  // might consider using the scalar as rhs if OPFN has such an
  // overload.

  #define OPEQ_FUNC(OP,OPFN,CONSTRAINT) \
    simd_t & OP ( const simd_t & rhs ) \
    { CONSTRAINT \
      for ( std::size_t n = 0 , i = 0 ; n < vsize ; ++i , n += L() ) \
        take ( i , OPFN ( yield ( i ) , rhs.yield ( i ) ) ) ; \
      return *this ; \
    } \
    simd_t & OP ( const T & rhs ) \
    { CONSTRAINT \
      for ( std::size_t n = 0 , i = 0 ; n < vsize ; ++i , n += L() ) \
        take ( i , OPFN ( yield ( i ) , hn::Set ( D() , rhs ) ) ) ; \
      return *this ; \
    }

    OPEQ_FUNC(operator+=,hn::Add,)
    OPEQ_FUNC(operator-=,hn::Sub,)
    OPEQ_FUNC(operator*=,hn::Mul,)
    
    // Div and Mod are not available for all types, see further down
    
    OPEQ_FUNC(operator&=,hn::And,)
    OPEQ_FUNC(operator|=,hn::Or,)
    OPEQ_FUNC(operator^=,hn::Xor,)

    // these definitions of left and right shift may not be
    // optimal for scalar rhs, which is provided by creating
    // a vec_t from the scalar and broadcasting that.

    OPEQ_FUNC(operator<<=,hn::Shl,INTEGRAL_ONLY)
    OPEQ_FUNC(operator>>=,hn::Shr,INTEGRAL_ONLY)

  #undef OPEQ_FUNC

  // integer division is rolled out to scalar, but float data
  // will use hn::Div

  template < typename rhs_t >
  simd_t & div ( const rhs_t & rhs , std::false_type )
  {
    auto * p_r = rhs.data() ;
    for ( std::size_t i = 0 ; i < size() ; i++ )
      inner [ i ] /= p_r [ i ] ;
    return *this ;
  }

  simd_t & div ( const T & rhs , std::false_type )
  {
    for ( std::size_t i = 0 ; i < size() ; i++ )
      inner [ i ] /= rhs ;
    return *this ;
  }

  // float division is handled with vector code:

  template < typename rhs_t >
  simd_t & div ( const rhs_t & rhs , std::true_type )
  {
    for ( std::size_t n = 0 , i = 0 ; n < vsize ; ++i , n += L() )
      take ( i , hn::Div ( yield ( i ) , rhs.yield ( i ) ) ) ;
    return *this ;
  }

  simd_t & div ( const T & rhs , std::true_type )
  {
    for ( std::size_t n = 0 , i = 0 ; n < vsize ; ++i , n += L() )
      take ( i , hn::Div ( yield ( i ) , hn::Set ( D() , rhs ) ) ) ;
    return *this ;
  }

  template < typename rhs_t >
  simd_t & operator/= ( const rhs_t & rhs )
  {
    typedef typename std::is_floating_point < T > :: type is_float_t ;
    return div ( rhs , is_float_t() ) ;
  }

  // modulo is rolled out unconditionally, because there is no hwy::Mod

  template < typename rhs_t >
  simd_t & operator%= ( const rhs_t & rhs )
  {
    auto * p_r = rhs.data() ;
    for ( std::size_t i = 0 ; i < size() ; i++ )
      inner [ i ] %= p_r [ i ] ;
    return *this ;
  }

  simd_t & operator%= ( const T & rhs )
  {
    auto * p_r = rhs.data() ;
    for ( std::size_t i = 0 ; i < size() ; i++ )
      inner [ i ] %= rhs ;
    return *this ;
  }

  // binary operators and left and right scalar operations with
  // value_type, unary operators -, ! and ~

#define OP_FUNC(OPFUNC,OPEQ,CONSTRAINT) \
  template < typename RHST , \
             typename = typename std::enable_if \
                       < std::is_fundamental < RHST > :: value \
                       > :: type \
           > \
  simd_t < PROMOTE ( T , RHST ) , vsize > \
  OPFUNC ( simd_t < RHST , vsize > rhs ) const \
  { \
    CONSTRAINT \
    simd_t < PROMOTE ( T , RHST ) , vsize > help ( *this ) ; \
    help OPEQ rhs ; \
    return help ; \
  } \
  template < typename RHST , \
             typename = typename std::enable_if \
                       < std::is_fundamental < RHST > :: value \
                       > :: type \
           > \
  simd_t < PROMOTE ( T , RHST ) , vsize > \
  OPFUNC ( RHST rhs ) const \
  { \
    CONSTRAINT \
    simd_t < PROMOTE ( T , RHST ) , vsize > help ( *this ) ; \
    help OPEQ rhs ; \
    return help ; \
  } \
  template < typename LHST , \
             typename = typename std::enable_if \
                       < std::is_fundamental < LHST > :: value \
                       > :: type \
           > \
  friend simd_t < PROMOTE ( LHST , T ) , vsize > \
  OPFUNC ( LHST lhs , simd_t rhs ) \
  { \
    CONSTRAINT \
    simd_t < PROMOTE ( LHST , T ) , vsize > help ( lhs ) ; \
    help OPEQ rhs ; \
    return help ; \
  }

  OP_FUNC(operator+,+=,)
  OP_FUNC(operator-,-=,)
  OP_FUNC(operator*,*=,)
  OP_FUNC(operator/,/=,)

  OP_FUNC(operator%,%=,INTEGRAL_ONLY)
  OP_FUNC(operator&,&=,INTEGRAL_ONLY)
  OP_FUNC(operator|,|=,INTEGRAL_ONLY)
  OP_FUNC(operator^,^=,)
  OP_FUNC(operator<<,<<=,INTEGRAL_ONLY)
  OP_FUNC(operator>>,>>=,INTEGRAL_ONLY)

  #undef OP_FUNC

  // for unary operators, relying on the operators to be defined
  // for now - should use hwy functions instead

  #define OP_FUNC(OPFUNC,OP,CONSTRAINT) \
    simd_t OPFUNC() const \
    { \
      simd_t help ; \
      for ( std::size_t n = 0 , i = 0 ; n < vsize ; ++i , n += L() ) \
        help.take ( i , OP ( yield ( i ) ) ) ; \
      return help ; \
    }

  OP_FUNC(operator-,hn::Neg,)
  OP_FUNC(operator~,hn::Not,)

  #undef OP_FUNC

  // provide methods to produce a mask on comparing a vector
  // with another vector or a value_type.

  #define COMPARE_FUNC(OP,OPFUNC) \
  friend mask_type OPFUNC ( const simd_t & lhs , \
                            const simd_t & rhs ) \
  { \
    mask_type m ; \
    for ( std::size_t n = 0 , i = 0 ; n < vsize ; ++i , n += lhs.L() ) \
      m.take ( i , OP ( lhs.yield ( i ) , rhs.yield ( i ) ) ) ; \
    return m ; \
  } \
  friend mask_type OPFUNC ( const simd_t & lhs , \
                            const value_type & rhs ) \
  { \
    mask_type m ; \
    for ( std::size_t n = 0 , i = 0 ; n < vsize ; ++i , n += lhs.L() ) \
      m.take ( i , OP ( lhs.yield ( i ) , hn::Set ( D() , rhs ) ) ) ; \
    return m ; \
  } \
  friend mask_type OPFUNC ( const value_type & lhs , \
                            const simd_t & rhs ) \
  { \
    mask_type m ; \
    for ( std::size_t n = 0 , i = 0 ; n < vsize ; ++i , n += rhs.L() ) \
      m.take ( i , OP ( hn::Set ( D() , lhs ) , rhs.yield ( i ) ) ) ; \
    return m ; \
  }

  COMPARE_FUNC(Lt,operator<) ;
  COMPARE_FUNC(Le,operator<=) ;
  COMPARE_FUNC(Gt,operator>) ;
  COMPARE_FUNC(Ge,operator>=) ;
  COMPARE_FUNC(Eq,operator==) ;
  COMPARE_FUNC(Ne,operator!=) ;

  #undef COMPARE_FUNC

  // next we define a masked vector as an object holding two members:
  // one mask type, determining which of the vector's elements will
  // be 'open' to an effect, and one reference to a vector, which will
  // be affected by the operation.
  // The resulting object will only be viable as long as the referred-to
  // vector is alive - it's meant as a construct to be processed
  // in the same scope, as the lhs of an assignment, typically using
  // notation introduced by Vc: a vector's operator() is overloaded
  // to produce a masked_type when called with a mask_type object, and
  // the resulting masked_type object is then assigned to.
  // Note that this does not have any effect on those values in 'whither'
  // for which the mask is false. They remain unchanged.

  struct masked_type
  {
    const mask_type whether ; // if the mask is true at whether[i]
    simd_t & whither ; // whither[i] will be assigned to

    std::size_t L() const
    {
      return whither.L() ;
    }

    masked_type ( const mask_type & _whether ,
                  simd_t & _whither )
    : whether ( _whether ) ,
      whither ( _whither )
      { }

    template < typename D2 , std::size_t N2 >
    masked_type ( const mchunk_t < D2 , N2 > & _whether ,
                  simd_t & _whither )
    : whether ( _whether ) ,
      whither ( _whither )
      {
//         whether = _whether ;
      }

    // for the masked vector, we define the complete set of assignments:

    simd_t & operator= ( const value_type & rhs ) \
    { \
      for ( std::size_t n = 0 , i = 0 ; n < vsize ; ++i , n += L() ) \
      { \
        auto m = whether.yield ( i ) ; \
        auto v = whither.yield ( i ) ; \
        auto vr = hn::Set ( D() , rhs ) ; \
        whither.take ( i , hn::IfThenElse ( m , vr , v ) ) ; \
      } \
      return whither ; \
    } \
    simd_t & operator= ( const simd_t & rhs ) \
    { \
      for ( std::size_t n = 0 , i = 0 ; n < vsize ; ++i , n += L() ) \
      { \
        auto m = whether.yield ( i ) ; \
        auto v = whither.yield ( i ) ; \
        auto vr = rhs.yield ( i ) ; \
        whither.take ( i , hn::IfThenElse ( m , vr , v ) ) ; \
      } \
      return whither ; \
    }

    // most operators can be rolled out over vec_t

    #define OPEQ_FUNC(OPFUNC,OP,CONSTRAINT) \
      simd_t & OPFUNC ( const value_type & rhs ) \
      { \
        CONSTRAINT \
        for ( std::size_t n = 0 , i = 0 ; n < vsize ; ++i , n += L() ) \
        { \
          auto m = whether.yield ( i ) ; \
          auto v = whither.yield ( i ) ; \
          auto vr = hn::OP ( v , hn::Set ( D() , rhs ) ) ; \
          whither.take ( i , hn::IfThenElse ( m , vr , v ) ) ; \
        } \
        return whither ; \
      } \
      simd_t & OPFUNC ( const simd_t & rhs ) \
      { \
        CONSTRAINT \
        for ( std::size_t n = 0 , i = 0 ; n < vsize ; ++i , n += L() ) \
        { \
          auto m = whether.yield ( i ) ; \
          auto v = whither.yield ( i ) ; \
          auto vr = hn::OP ( v , rhs.yield ( i ) ) ; \
          whither.take ( i , hn::IfThenElse ( m , vr , v ) ) ; \
        } \
        return whither ; \
      }

    OPEQ_FUNC(operator+=,Add,)
    OPEQ_FUNC(operator-=,Sub,)
    OPEQ_FUNC(operator*=,Mul,)
    OPEQ_FUNC(operator&=,And,INTEGRAL_ONLY)
    OPEQ_FUNC(operator|=,Or,INTEGRAL_ONLY)
    OPEQ_FUNC(operator^=,Xor,INTEGRAL_ONLY)
    OPEQ_FUNC(operator<<=,Shl,INTEGRAL_ONLY)
    OPEQ_FUNC(operator>>=,Shr,INTEGRAL_ONLY)

    #undef OPEQ_FUNC

    // some operators are implemented on the simd_t level:
    // first, obtain a simd_t by applying OPEQ rhs to whither. This
    // intermediate result would be appropriate for an all-true mask.
    // Then apply this by calling operator= on *this: this picks the
    // masked assignment above (aka operator=), which only copies from
    // the all-true intermediate result where the mask is true.

    #define OPEQ_FUNC(OPFUNC,OPEQ,CONSTRAINT) \
      simd_t & OPFUNC ( const value_type & rhs ) \
      { \
        CONSTRAINT \
        simd_t mrhs ( whither ) ; \
        mrhs OPEQ rhs ; \
        return ( *this = mrhs ) ; \
      } \
      simd_t & OPFUNC ( const simd_t & rhs ) \
      { \
        CONSTRAINT \
        simd_t mrhs ( whither ) ; \
        mrhs OPEQ rhs ; \
        return ( *this = mrhs ) ; \
      }

    OPEQ_FUNC(operator%=,%=,INTEGRAL_ONLY)
    OPEQ_FUNC(operator/=,/=,)
    
  #undef OPEQ_FUNC

  } ;

  #undef INTEGRAL_ONLY

  // mimicking Vc, we define operator() with a mask_type argument
  // to produce a masked_type object, which can be used later on to
  // masked-assign to the referred-to vector. With this definition
  // we can use the same syntax Vc uses, e.g. v1 ( v1 > v2 ) = v3
  // This helps write code which compiles with Vc and without,
  // because this idiom is 'very Vc'.

  masked_type operator() ( const mask_type & mask )
  {
    return masked_type ( mask , *this ) ;
  }

  // member functions at_least and at_most. These functions provide the
  // same functionality as max, or min, respectively. Given simd_t X
  // and some threshold Y, X.at_least ( Y ) == max ( X , Y )
  // Having the functionality as a member function makes it easy to
  // implement, e.g., min as: min ( X , Y ) { return X.at_most ( Y ) ; }

  #define CLAMP(FNAME,REL) \
    simd_t FNAME ( const T & threshold ) const \
    { \
      return (*this) ( *this REL threshold ) = threshold ; \
    } \
    simd_t FNAME ( const simd_t & threshold ) const \
    { \
      return (*this) ( *this REL threshold ) = threshold ; \
    }

  CLAMP(at_least,<)
  CLAMP(at_most,>)

  #undef CLAMP

  // sum of vector elements. Note that there is no type promotion; the
  // summation is done to value_type. Caller must make sure that overflow
  // is not a problem.

  value_type sum() const
  {
    vec_t s ( hn::Zero ( D() ) ) ;
    for ( std::size_t n = 0 , i = 0 ; n < vsize ; ++i , n += L() )
      s += yield ( i ) ;
    return hn::GetLane ( hn::SumOfLanes ( D() , s ) ) ;
  }
} ;

} ; // namespace HWY_NAMESPACE

HWY_AFTER_NAMESPACE();  // at file scope

namespace simd
{
  template < typename T , std::size_t N >
  using hwy_simd_type = HWY_NAMESPACE::simd_t < T , N > ;
} ;

#ifndef HWY_SIMD_ALLOCATOR
#define HWY_SIMD_ALLOCATOR

namespace simd
{
  // for highway data, vspline needs an allocator, which is in turn
  // required by vigra::MultiArray to allocate vector-aligned storage.
  // Initially I coded the allocation using aligned_alloc, but this
  // function is not available on msys2, so I'm now using highway
  // functions.

  template < typename T >
  struct simd_allocator
  : public std::allocator < T >
  {
    typedef std::allocator < T > base_t ;
    using typename base_t::pointer ;
    pointer allocate ( std::size_t n )
    {
      return (pointer) hwy::AllocateAlignedBytes
        ( n * sizeof(T) , nullptr , nullptr ) ;
    }
    void deallocate ( T* p , std::size_t n )
    {
      hwy::FreeAlignedBytes ( p , nullptr , nullptr ) ;
    }
    using base_t::base_t ;
  } ;

  // // Im fixing this allocator via std::allocator_traits, but it
  // // does not seem to be picked for all allocations - I had, e.g.
  // // a std::vector of hwy_simd_type which contained unaligned
  // // memory and caused a crash (only with c++11, 17 is okay)
  // // - I worked around it in lux, but the problem is not solved.
  //
  // template < typename T , std::size_t N >
  // struct allocator_traits < hwy_simd_type < T , N > >
  // {
  //   typedef simd_allocator < hwy_simd_type < T , N > > type ;
  // } ;
} ;

namespace std
{
  template < typename T , std::size_t N >
  struct allocator_traits < simd::hwy_simd_type < T , N > >
  {
    typedef simd::simd_allocator
              < simd::hwy_simd_type < T , N > > allocator_type ;
  } ;
} ;

#endif // HWY_SIMD_ALLOCATOR

#endif // #define HWY_SIMD_TYPE_H
