/************************************************************************/
/*                                                                      */
/*    zimt - abstraction layer for SIMD programming                     */
/*                                                                      */
/*            Copyright 2024 by Kay F. Jahnke                           */
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


/*! \file block.h

    \brief tightly packed nD container types
*/

#ifndef ZIMT_BLOCK_H
#define ZIMT_BLOCK_H

#include <vector>
#include "xel.h"
#include "array.h"

namespace zimt
{

// block_t is like a small cousin to array_t: it holds a rectangular
// block of nD memory and can be indexed with a xel_t, but the strides
// are fixed to what an array_t would use by default. This container
// inherits iterator properties from std::vector, so it can be
// traversed efficiently. Access to individual data is possible via
// operator[], passing a xel_t as index, and also sequential in
// memory order, passing std::size_t as indexes - this simply maps
// to std::vector's operator[].
// Since block_t is only a small wrapper around std::vector and the
// size is fixed at construction, this type should optimize well.

template < std::size_t D , typename T >
struct vblock_t
: protected std::vector < T >
{
  typedef xel_t < std::size_t , D > index_t ;

  const index_t shape ;
  const index_t strides ;

  static index_t make_strides ( const index_t & shape )
  {
    index_t strides ;
    std::size_t stride = 1 ;
    strides [ 0 ] = stride ;

    for ( std::size_t d = 1 ; d < D ; d++ )
    {
      stride *= shape [ d - 1 ] ;
      strides [ d ] = stride ;
    }
    return strides ;
  }

  vblock_t ( const index_t & _shape )
  : std::vector < T > ( _shape.prod() ) ,
    shape ( _shape ) ,
    strides ( make_strides ( _shape ) )
  { }

  // we map to std:vector's iterator interface

  using std::vector<T>::begin ;
  using std::vector<T>::cbegin ;
  using std::vector<T>::end ;
  using std::vector<T>::cend ;

  // we allow sequential access with scalar indices. For 1D blocks,
  // access indices can be xel_t with one element or plain size_t.

  using std::vector < T > :: operator[] ;

  // indexing with nD indexes is also possible and uses the same
  // semantics as array indexing.

  const T & operator[] ( const index_t & i ) const
  {
    return operator[] ( ( i * strides ) . sum() ) ;
  }

  T & operator[] ( const index_t & i )
  {
    return operator[] ( ( i * strides ) . sum() ) ;
  }

  T * data()
  {
    return & ( (*this)[0] ) ;
  }
} ;

// block_t is a view_t with memory attached. It behaves like array_t,
// but doesn't use the shared_ptr for the memory.

template < std::size_t D , typename T >
struct block_t
: public view_t < D , T >
{
  typedef view_t < D , T > base_t ;
  using base_t::origin ;
  using typename base_t::index_type ;

  static index_type make_strides ( const index_type & shape )
  {
    index_type strides ;
    std::size_t stride = 1 ;
    strides [ 0 ] = stride ;

    for ( std::size_t d = 1 ; d < D ; d++ )
    {
      stride *= shape [ d - 1 ] ;
      strides [ d ] = stride ;
    }
    return strides ;
  }

  block_t ( const index_type & _shape )
  : base_t ( new T [ _shape.prod() ] ,
             make_strides ( _shape ) ,
             _shape )
  { }

  block_t ( const block_t & other )
  : base_t ( other )
  {
    origin = new T [ other.shape.prod() ] ;
    base_t::copy_data ( other ) ;
  }

  block_t operator= ( const block_t & other )
  {
    base_t::strides = other.strides ;
    base_t::shape = other.shape ;
    origin = new T [ other.shape.prod() ] ;
    base_t::copy_data ( other ) ;
  }

  ~block_t()
  {
    delete[] origin ;
  }
} ;

} ; // namespace zimt

#endif // #define ZIMT_BLOCK_H

