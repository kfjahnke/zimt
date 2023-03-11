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

/*! \file array.h

    \brief nD array and view classes

    classes view_t and array_t provide code to handle n-dimensional
    arrays with arbitrary striding. The strides are given in units
    of the value_type held in the array. This is vigra's convention
    and it differs from other schemes - e.g. strides in units of the
    fundamental type involved, or even bytes.

    Apart from that, the types are similar to NumPy's ndarray and
    vigra's Multiarray(View). The dimensionality and value_type are
    template arguments, shape and strides are run-time arguments,
    but const and fixed at construction time.

    While the implementation is very close to vigra's MultiArray and
    MultiArrayView, the types here are - for now - stripped down to
    the bare essentials. There are also subtle differences in the
    copying and assignment logic: in zimt, all copy c'tors only ever
    copy the member variables of the array/view object. To copy data
    between memory referred to by different views, the explicit
    copy_data function has to be used.

    view_t is lightweight and only holds the 'origin' pointer, shape
    and strides. array_t adds data ownership via a std::shared_ptr.
    This allows array_t objects to be copied and passed around freely,
    and when the last copy goes out of scope, the owned data are
    destructed.

    This header also has two simple iterators, one random-access
    iterator producing an nD index from a sequence number, and one
    sequential operator, producing nD indexes rising from {n*0}

    Note here that I don't care much about keeping class mebers
    hidden by making them private. I do so now and then when there is
    no obvious use for class member code to be used outside, but apart
    from that, you're free to 'look into' the objects and to mess with
    them at your own risk. Coding this way saves writing a lot of
    getters, setters, and parameter checkers, but at times it's
    walking a thin line.
*/

#include "xel.h"
#include <memory>
#include <cstring>

#ifndef ZIMT_ARRAY_H

namespace zimt
{
class view_flag { } ;

// view_t is a view to an array. Like vigra, zimt uses view_t
// as the base class and derives array_t from it, adding ownership
// of the data. class view_t is lightweight and only refers to data
// held elsewhere via a pointer to value_type.

template < std::size_t D , typename T >
struct view_t
: public view_flag
{
  typedef T value_type ;
  static const std::size_t dimension = D ;
  typedef xel_t < std::size_t , dimension > shape_type ;
  typedef xel_t < long , dimension > index_type ;

  // we keep the members const. assignment between views and arrays
  // is not allowed, only copy construction can initialize a view
  // or array with the same members.

  value_type * const origin ;
  const index_type strides ;
  const shape_type shape ;

  view_t()
  : origin ( nullptr ) ,
    strides ( 0 ) ,
    shape ( 0 )
  { }

  // helper type six_t is used to construct slice_t. It is a xel_t
  // of one dimension less than 'dimension' - or a xel_t with one
  // channel if dimension == 1. six_t stands for 'slice index type'.
  // The second variant is for the shape and uses dtf::size_t
  // instead of long

  typedef typename
    std::conditional < dimension == 1 ,
                       xel_t < long , dimension > ,
                       xel_t < long , dimension - 1 >
                      > :: type six_t ;

  typedef typename
    std::conditional < dimension == 1 ,
                       xel_t < std::size_t , dimension > ,
                       xel_t < std::size_t , dimension - 1 >
                      > :: type sixsz_t ;

  // slice_t, the type for a slice, is a view_t with one dimension
  // less - unless 'this' view already is 1D, in which case it's
  // another 1D view_t. The logic is so that you can 'slice' a 1D
  // view, but you'll only get another 1D view as a result. To
  // 'pick out' an individual value, use operator[], to 'pick out'
  // a section, use 'window'.

  typedef view_t < six_t::nch , value_type > slice_t ;

  // make_strides creates 'canonical' strides for an array of
  // the given shape. The strides are in ascending order.

  static index_type make_strides ( const shape_type & shape )
  {
    index_type strides ;
    long stride = 1 ;
    strides [ 0 ] = stride ;

    for ( std::size_t d = 1 ; d < D ; d++ )
    {
      stride *= shape [ d - 1 ] ;
      strides [ d ] = stride ;
    }
    return strides ;
  }

  // compact checks whether the view refers to a contiguous chunk
  // of memory in 'canonical' order, like a view which refers to
  // an array created with a given shape.

  bool compact() const
  {
    return ( strides == make_strides ( shape ) ) ;
  }

  // This c'tor creates a new view from the given arguments.

  view_t ( value_type * _origin ,
           const index_type & _strides ,
           const shape_type & _shape )
  : origin ( _origin ) ,
    strides ( _strides ) ,
    shape ( _shape )
    { }

  // view_t's copy c'tor creates a new view with the same properties,
  // referring to the same data.

  view_t ( const view_t & rhs )
  : origin ( rhs.origin ) ,
    strides ( rhs.strides ) ,
    shape ( rhs.shape )
  { }

  view_t ( const view_t & rhs ,
           const index_type & _strides ,
           const shape_type & _shape )
  : view_t ( rhs.origin , _strides , _shape )
  { }

  // copy assignment is forbidden.

  view_t & operator= ( const view_t & rhs ) = delete ;

  // get the number of value_type the view refers to.

  long size() const
  {
    return shape.prod() ;
  }

  // convert an index to an offset from origin.

  long offset ( const index_type & crd ) const
  {
    return ( crd * strides ) . sum() ;
  }

  // non-const and const versions of operator[].

  T & operator[] ( const index_type & crd )
  {
    return origin [ offset ( crd ) ] ;
  }

  const T & operator[] ( const index_type & crd ) const
  {
    return origin [ offset ( crd ) ] ;
  }

  // 'peek' function giving access to the view's origin

  value_type * data() const
  {
    return origin ;
  }

  // window is used to create a view to a 'box-shaped' part
  // of the data.

  view_t window ( const index_type & start ,
                  const index_type & end )
  {
    return view_t ( origin + offset ( start ) ,
                    strides ,
                    end - start ) ;
  }

private:

  // now the slicing operation. it creates a subdimensional view
  // coinciding with a window with dimension d having extent 1.
  // If 'this' view already is 1D, the only dimension left is
  // dimension 0, and the only meaningful argument 'k' is zero.
  // This is not enforced - it's assumed that no sane code will
  // attempt to slice 1D views - but the 1D case is handled
  // saparately and returns the same view.
  // TODO: might consider a 1D view a 2D view with extent 1
  // in the second axis. Then, d could be 0 or 1, and k would
  // be meaningful.

  slice_t _slice ( std::size_t d , long k , std::false_type ) const
  {
    six_t sl_strides ;
    sixsz_t sl_shape ;

    for ( int i = 0 , j = 0 ; i < dimension ; i++ )
    {
      if ( i != d )
      {
        sl_shape [ j ] = shape [ i ] ;
        sl_strides [ j ] = strides [ i ] ;
        ++j ;
      }
    }
    return slice_t ( origin + k * strides [ d ] , sl_strides , sl_shape ) ;
  }

  slice_t _slice ( std::size_t d , long k , std::true_type ) const
  {
    return slice_t ( origin , strides , 1 ) ; // shape ) ;
  }

public:

  slice_t slice ( std::size_t d , long i ) const
  {
    static const bool is_1d = ( dimension == 1 ) ;
    return _slice ( d , i , std::integral_constant < bool , is_1d >() ) ;
  }

private:

  // for assignments from another view, we need a bit of collateral
  // code. Here we actually copy the data from one view to the
  // other, and depending on the strides and dimensionality this
  // is done differently, so we use a dispatch (bottom of private
  // section) and special case overloads. We keep the types and
  // member functions used for the operation private.

  // The first two overloads of 'copy_data' handle a 'straight' copy from
  // one 1D view to another. The second two overloads deal with nD views
  // and use slicing and recursion until (or unless) the data are
  // contiguous im memory.

  void _copy_data ( const view_t & rhs ,
                    std::false_type , // may not use memcpy
                    std::true_type )  // data are 1D
  {
    // Always uses a loop with individual assignments.

    for ( long i = 0 ; i < shape[0] ; i++ )
      origin [ i * strides[0] ] = rhs.origin [ i * rhs.strides [ 0 ] ] ;
  }

  void _copy_data ( const view_t & rhs ,
                    std::true_type , // may use memcpy
                    std::true_type ) // data are 1D
  {
    // If the data are unstrided, delegate to memcpy. Otherwise,
    // use the loop version (above).

    if ( strides[0] == 1 && rhs.strides[0] == 1 )
    {
      memcpy ( origin , rhs.origin , sizeof ( value_type ) * shape[0] ) ;
    }
    else
    {
      _copy_data ( rhs , std::false_type() , std::true_type() ) ;
    }
  }

  // the nD overload of copy_data looks at the strides of the
  // arrays, and if they indicate that both views are 'compact'
  // it uses a straight memcpy if the data allow it.

  void _copy_data ( const view_t & rhs ,
                    std::false_type , // may not use memcpy
                    std::false_type ) // dta are > 1D
  {
    // for all slices 'along' the last dimension, invoke copy_data

    for ( long i = 0 ; i < shape [ dimension - 1 ] ; i++ )
    {
      auto slhs = slice ( dimension - 1 , i ) ;
      auto srhs = rhs.slice ( dimension - 1 , i ) ;
      slhs.copy_data ( srhs ) ;
    }
  }

  void _copy_data ( const view_t & rhs ,
                    std::true_type ,  // may use memcpy
                    std::false_type ) // data are > 1D
  {
    // to check whether we can use a straight memcpy, we need to
    // make sure both views are 'compact'. Their size must agree,
    // because their shape is the same - this is checked in the
    // public overload.

    bool compatible = ( compact() && rhs.compact() ) ;

    if ( compatible )
    {
      // std::cout << "compatible for memcpy" << std::endl ;

      memcpy ( origin , rhs.origin ,
               sizeof ( value_type ) * shape.prod() ) ;
    }
    else
    {
      _copy_data ( rhs , std::false_type() , std::false_type() ) ;
    }
  }

  // set_data is simpler, we only need to handle 1D and nD cases.

  void _set_data ( const value_type & rhs , std::true_type ) // 1D
  {
    // If the data are 1D, use a loop with individual assignments.

    if ( strides [ 0 ] == 1 )
    {
      for ( long i = 0 ; i < shape[0] ; i++ )
        origin [ i ] = rhs ;
    }
    else
    {
      long end = shape [ 0 ] * strides [ 0 ] ;
      for ( long i = 0 ; i < end ; i += strides [ 0 ] )
        origin [ i ] = rhs ;
    }
  }

  void _set_data ( const value_type & rhs , std::false_type ) // > 1D
  {
    if ( compact() )
    {
      for ( long i = 0 ; i < size() ; i++ )
        origin [ i ] = rhs ;
    }
    else
    {
      for ( long i = 0 ; i < shape [ dimension - 1 ] ; i++ )
      {
        auto sl = slice ( dimension - 1 , i ) ;
        sl.set_data ( rhs ) ;
      }
    }
  }

public:

  // public overloads of copy_data and set_data, deciding whether
  // the operation is 1D or nD, and whether memcpy may be used.

  void copy_data ( const view_t & rhs )
  {
    static const bool is_bland
      = std::is_trivially_copyable < value_type > :: value ;

    static const bool is_1d ( dimension == 1 ) ;

    assert ( shape == rhs.shape ) ;

    _copy_data ( rhs ,
                 std::integral_constant < bool , is_bland >() ,
                 std::integral_constant < bool , is_1d >() ) ;
  }

  void set_data ( const value_type & rhs )
  {
    // std::cout << "base set_data" << std::endl ;
    static const bool is_1d ( dimension == 1 ) ;
    _set_data ( rhs , std::integral_constant < bool , is_1d >() ) ;
  }

} ;

// array_t 'holds' memory holding the array's data. This done via a
// std::shared_ptr, so that arrays can be passed around without having
// to worry about when the data are destroyed. 'base' does not necessarily
// point to the same address as origin does - when forming subarrays or
// slicing, base will be passed unchanged, but origin may differ. It's
// also possible to create the array with a shared_ptr passed in, which
// allows for code where, e.g., one thread allocates the memory and then
// passes it to some other thread for a specific task. If the allocating
// thread 'lets go' of the shared_ptr, and the processing thread is done
// with the data and also 'lets go' of the shared_ptr, the memory is
// freed automatically. The shared_ptr used to keep track of data
// ownership may or may not hold the same value as 'origin'; user code
// should not make any assumptions about this.
// passing around array_t objects is rarely needed - the typical scenario
// is passing transient arrays to separate threads, when it's impractical
// to wait around and test for the thread to signal that it's done with
// the data. Most use cases will instead use RAII and initially create
// an array_t object but then pass around views to it, letting the array_t
// object destruct when it goes out of scope. If you want to avoid using
// array_t altogether that's fine, you just have to deal with memory
// 'manually'.

template < std::size_t D , typename T >
class array_t
: public view_t < D , T >
{
  array_t() = default ;

public:

  typedef view_t < D , T > base_t ;

  // add a shared_ptr to the data the view is 'based on', meaning that
  // the chunk of memory the shared_ptr points to envelopes the data
  // the view refers to.

  std::shared_ptr < T > base ;

  using typename base_t::value_type ;
  using typename base_t::shape_type ;
  using typename base_t::index_type ;
  using base_t::dimension ;
  using base_t::origin ;
  using base_t::strides ;
  using base_t::shape ;
  using base_t::make_strides ;
  using base_t::operator[] ;

  // array based on memory held elsewhere. The shared_ptr is copied
  // in, and if the array is destroyed, the memory is only released
  // if this was the last copy of the shared_ptr in circulation

  array_t ( std::shared_ptr < T > _base ,
            const shape_type & _shape )
  : base_t ( _base.get() ,
             make_strides ( _shape ) ,
             _shape ) ,
    base ( _base )
  { }

  // array based on a shared_ptr and a view

  array_t ( std::shared_ptr < T > _base ,
            const base_t & view )
  : base_t ( view ) ,
    base ( _base )
  { }

  // copy c'tor. This does create a view to the same data, sharing the
  // same shared_ptr to the data. If you want a new array holding a
  // copy of the data, create a new array and use copy_data.

  array_t ( const array_t & rhs )
  : base_t ( rhs ) ,
    base ( rhs.base )
  { }

  // copy c'tor with different shape and strides

  array_t ( const array_t & rhs ,
            const index_type & _strides ,
            const shape_type & _shape )
  : base ( rhs.base ) ,
    base_t ( rhs.origin , _strides , _shape )
  { }

  // copy assignment is forbidden.

  array_t & operator= ( const array_t & rhs ) = delete ;

  // array allocating fresh memory. The array is now in sole possesion
  // of the memory, and unless it's copied the memory is released when
  // the array is destructed.

  array_t ( const shape_type & _shape )
  : base_t ( new T [ _shape.prod() ] ,
             make_strides ( _shape ) ,
             _shape )
  {
    base.reset ( origin ) ;
  }

  // array's window function also copies the shared_ptr, base. This makes
  // sure that the returned subarray will hold on to the memory. The down
  // side of this is of course that even a tiny subarry may hold on to
  // a large chunk of memory.

  array_t window ( const index_type & start ,
                   const index_type & end )
  {
    return array_t ( base , base_t::window ( start , end ) ) ;
  }

  // slicing of an array_t 'holds on' to the shared_ptr, base, just as
  // windowing does.

  typedef typename std::conditional
                     < dimension == 1 ,
                       array_t ,
                       array_t < dimension - 1 , value_type >
                     > :: type slice_t ;

  slice_t slice ( std::size_t d , long i ) const
  {
    base_t const & v ( *this ) ;
    auto slc = v.slice ( d , i ) ;
    return slice_t ( base , slc ) ;
  }
} ;

// coordinate iterators. For now, we don't implement 'proper'
// c++ standard operators, but just two stripped-down ones.
// zimt views are suitable for random access iteration, but it
// would require extra coding effort to provide that.

// stripped-down version of vigra::MultiCoordinateIterator which only
// produces the nth nD index into an array of given shape. This class
// is used in wielding.h. use of the modulo and division make this
// quite slow, so it's best not used for iterations over arrays.
// in wielding.h it's only used to produce the start address for
// each line, so it's not time-critical there.
// Note that if the data are contiguous in memory, it can be much
// more efficient to iterate over the data directly - or over a 1D
// view of the data.
// Why do we need this iterator? For multithreading. There, the worker
// threads obtain 'joblet numbers' from an atomic and decode these
// numbers to segments of data which need to be processed. And the
// decoding is where this iterator is used.

template  < std::size_t D >
struct mci_t
: public xel_t < long , D >
{
  static const std::size_t dimension = D ;

  typedef xel_t < long , D > index_type ;
  const index_type & shape ;

  template < typename T >
  mci_t ( xel_t < T , D >  _shape )
  : index_type ( _shape ) ,
    shape ( *this )
  { }

  // accept any argument as long as can be converted to index_type
  // TODO: this is very permissive - might be better to restrict to
  // fixed-size containers of some integral type

  template < typename T >
  mci_t ( T _shape )
  : index_type ( _shape ) ,
    shape ( *this )
  { }

  template < typename T >
  mci_t ( const std::initializer_list < T > & rhs )
  : index_type ( rhs ) ,
    shape ( *this )
  { }

  index_type operator[] ( long i )
  {
    index_type result ;

    for ( std::size_t d = 0 ; d < dimension ; d++ )
    {
      result [ d ] = i % shape [ d ] ;
      i /= shape [ d ] ;
    }
    return result ;
  }
} ;

// similar to mci_t, but this class is for walking through a sequence
// of nD coordinates. It's not for random access. For now, forward
// only. Here there are no division or modulo operations.

template  < std::size_t D >
struct mcs_t
: public mci_t < D >
{
  typedef mci_t < D > base_t ;
  using base_t::base_t ;
  using base_t::dimension ;
  using typename base_t::index_type ;
  using base_t::shape ;

  index_type current = 0L ;

  // operator() yields the current value and updates 'current'.

  index_type operator() ()
  {
    // a bit verbose, let the optimizer figure it out

    auto result = current ;
    for ( std::size_t d = 0 ; d < dimension ; d++ )
    {
      if ( ++ current [ d ] == shape [ d ] )
        current [ d ] = 0 ;
      else
        break ;
    }
    return result ;
  }
} ;

} ;

#define ZIMT_ARRAY_H
#endif

