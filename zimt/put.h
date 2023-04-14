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

/*! \file put.h

    \brief classes to provide input to an 'act' functor

    zimt::process feeds the 'act' functor, working along
    the aggregation axis, while the coordinate's component along the
    other axes remain constant. This effectively 1D operation can
    be coded efficiently, whereas the 'act' functor would typically
    look at it's input as an nD entity and not be aware of the fact
    that only one of the input's components is 'moving'. Once the
    'act' functor is done, it produces a vectorized result which
    is in turn disposed of by a put_t object.
    This header provides a set of classes which are meant to fit into
    this slot. They follow a specific pattern which results from the
    logic in zimt_process, namely the two init and two save overloads.

*/

#ifndef ZIMT_PUT_H

namespace zimt
{

// class storer disposes of the results of calling the 'act'
// functor by storing the data to a view/array.

template < typename T ,    // elementary/fundamental type
           std::size_t N , // number of channels
           std::size_t D , // dimension of the view/array
           std::size_t L = zimt::vector_traits < T > :: vsize >
struct storer
{
  typedef zimt::xel_t < T , N > value_t ;
  typedef zimt::simdized_type < value_t , L > value_v ;
  typedef zimt::xel_t < long , D > crd_t ;

  const std::size_t & d ;
  zimt::view_t < D , value_t > & trg ;
  const std::size_t stride ;

  value_t * p_trg ;

  // get_t's c'tor receives the target array and the processing
  // axis. Similar classes might use other c'tor arguments. This
  // get_t holds no variable state apart from p_trg.

  storer ( zimt::view_t < D , value_t > & _trg ,
           const std::size_t & _d = 0 )
  : trg ( _trg ) , d ( _d ) , stride ( _trg.strides [ _d ] )
  { }

  // init is used to initialize the target pointer

  void init ( const crd_t & crd )
  {
    p_trg = & ( trg [ crd ] ) ;
  }

  // save writes to the current taget pointer and increases the
  // pointer by the amount of value_t written

  void save ( const value_v & v )
  {
    v.fluff ( p_trg , stride ) ;
    p_trg += L * stride ;
  }

  // capped save, used for the final batch of data which did not
  // fill out an entire value_v

  void save ( const value_v & v , const std::size_t & cap )
  {
    v.fluff ( p_trg , stride , cap ) ;
  }
} ;

// split_t encodes the reverse operation of join_t: The content of
// simdized data is distributed to N arrays of T which will contain
// per-channel data, e.g. a red, a green and a blue array to store
// RGB data.

template < typename T ,    // elementary/fundamental type
           std::size_t N , // number of channels
           std::size_t D , // dimension of the view/array
           std::size_t L = zimt::vector_traits < T > :: vsize >
struct split_t
{
  typedef zimt::xel_t < T , N > value_t ;
  typedef zimt::simdized_type < value_t , L > value_v ;
  typedef zimt::xel_t < long , D > crd_t ;

  typedef std::array < zimt::view_t < D , T > , N > trg_t ;
  trg_t & trg ;

  zimt::xel_t < T* , N > p_trg ;   // target pointers
  zimt::xel_t < long , N > stride ; // strides of target arrays
  const std::size_t d ;             // 'hot' axis

  // the c'tor receives the set of target arrays and the 'hot' axis

  split_t ( trg_t & _trg , const std::size_t & _d = 0 )
  : trg ( _trg ) , d ( _d )
  {
    // copy out the strides of the target arrays

    for ( int ch = 0 ; ch < N ; ch++ )
    {
      stride [ ch ] = trg [ ch ] . strides [ d ] ;
    }
  }

  // init is used to initialize the target pointers

  void init ( const crd_t & crd )
  {
    for ( int ch = 0 ; ch < N ; ch++ )
    {
      p_trg [ ch ] = & ( trg [ ch ] [ crd ] ) ;
    }
  }

  // save writes to the current taget pointers and increases the
  // pointers by the amount of value_t written

  void save ( const value_v & v )
  {
    for ( int ch = 0 ; ch < N ; ch++ )
    {
      v [ ch ] . rscatter ( p_trg [ ch ] , stride [ ch ] ) ;
      p_trg [ ch ] += L * stride [ ch ] ;
    }
  }

  // capped save, used for the final batch of data which did not
  // fill out an entire value_v

  void save ( const value_v & v , const std::size_t & cap )
  {
    for ( int ch = 0 ; ch < N ; ch++ )
    {
      for ( std::size_t e = 0 ; e < cap ; e++ )
        p_trg [ ch ] [ e * stride [ ch ] ] = v [ ch ] [ e ] ;
    }
  }
} ;

// vstorer stores vectorized data to a zimt::view_t of fundamental
// values (T). This is a good option for storing intermediate results,
// because it can use efficient SIMD store operations rather than
// having to interleave the data to store them as xel of T. To
// retrieve the data from such a vectorized storage array, use
// class vloader (see get.h). The target view should refer to an
// array obtained via zimt::get_vector_buffer.

template < typename T ,    // elementary/fundamental type
           std::size_t N , // number of channels
           std::size_t D , // dimension of the view/array
           std::size_t L = zimt::vector_traits < T > :: vsize >
struct vstorer
{
  typedef zimt::xel_t < T , N > value_t ;
  typedef zimt::simdized_type < value_t , L > value_v ;

  // type of coordinate passed by the caller (zimt::process)

  typedef zimt::xel_t < long , D > crd_t ;

  // axis of the storage array which corresponds to the 'hot' axis
  // of the 'notional' array (so, d + 1 ), we'll refer to this axis
  // as the 'hot' axis of the storage array as well.

  const std::size_t d ;

  // the array serving as storage space for the vectorized data,
  // with the additional dimension zero with extent N * L

  zimt::view_t < D + 1 , T > & trg ;

  // stride along the 'hot' axis of the storage array

  const std::size_t stride ;

  // current target of the store operation

  T * p_trg ;

  // vstorer's c'tor receives the target array and the processing
  // axis. Here, the target array is an array of T with an added
  // dimension zero with extent N * L. The template argument 'D'
  // is the dimensionality of the 'notional' array. The argument _d
  // refers to the 'hot' axis of the 'notional' array

  vstorer ( zimt::view_t < D + 1 , T > & _trg ,
            const std::size_t & _d = 0 )
  : trg ( _trg ) ,
    d ( _d + 1 ) ,
    stride ( _trg.strides [ _d + 1 ] )
  { }

  // init is used to initialize the target pointer. We receive a
  // coordinate referring to a D-dimensional array of xel<T,N>

  void init ( const crd_t & _crd )
  {
    // calculate the D+1-dimensional coordinate into 'trg'.
    // This coordinate's first component is zero.

    xel_t < std::size_t , D + 1 > crd ;
    crd [ 0 ] = 0 ;

    for ( std::size_t i = 0 ; i < D ; i++ )
    {
      // the coordinate's component along the 'hot' axis is
      // divided by the lane count, the other components remain
      // the same, but all components are 'one axis further up'.

      if ( i == ( d - 1 ) )
      {
        crd [ i + 1 ] = _crd [ i ] / L ;
      }
      else
      {
        crd [ i + 1 ] = _crd [ i ] ;
      }
    }

    // with the coordinate into 'trg' we can figure out the
    // initial target address

    p_trg = & ( trg [ crd ] ) ;
  }

  // save writes to the current taget pointer and increases the
  // pointer by the amount of value_t written

  void save ( const value_v & v , const std::size_t & cap = 0 )
  {
    v.store ( p_trg ) ;
    p_trg += stride ;
  }
} ;

// discard_result is a put_t which doesn't do anything, it's only
// there to fill the put_t 'slot'. It's useful for situations where
// the results of the 'act' functor aren't needed, e.g. in reductions
// where the 'act' functor performs statistics. Another use case is
// performance tests, where the act functor is invoked repeatedly,
// but only it's performance is of interest and the cost of saving
// it's output should be avoided.

template < typename T ,     // fundamental type
           std::size_t N ,  // channel count
           std::size_t D ,  // dimensions
           std::size_t L >  // lane count
struct discard_result
{
  template < typename A >
  void init ( const A & crd )
  { }

  template < typename ... A >
  void save ( A ... args )
  { }
} ;

// to avoid having to deal with the concrete type of a put_t, we can
// use type erasure, a technique I call 'grokking'. Basically, it
// captures an object's set of member functions as std::functions
// and provides a new object which delegates to these std::functions.
// In result the new object is decoupled from the 'grokkee' type, but
// provides it's functionality just the same. The internal workings
// do still use the 'grokked' object, but it's type is hidden from
// view - it's been 'erased'.
// For a put_t, we need an object providing one init and two save
// overloads. Here's the class definition - it's quite a mouthful,
// but further down there's a factory function to perform the 'grok'
// which uses ATD, making the process simple.

template < typename T ,     // fundamental type
           std::size_t N ,  // channel count
           std::size_t D ,  // dimensions
           std::size_t L >  // lane count
struct grok_put_t
{
private:
  // we need some of the grokkee's types

  typedef zimt::xel_t < T , N > value_t ;
  typedef zimt::simdized_type < value_t , L > value_v ;
  typedef zimt::xel_t < long , D > crd_t ;

  // grok_put_t holds five std::functions of these five types:

  typedef std::function < void ( void * & ,
                                 const crd_t & ) > vinit_f ;

  typedef std::function < void ( void * & ,
                                 value_v & ) > vsave_f ;

  typedef std::function < void ( void * & ,
                                 value_v & ,
                                 const std::size_t & ) > csave_f ;

  // these last two types are 'infrastructure code' - it's the
  // same mechanism that's used in grok_type.

  typedef std::function < void* ( void* ) > replicate_type ;
  typedef std::function < void ( void* ) > terminate_type ;

  // here they are

  vinit_f vinit ;
  vsave_f vsave ;
  csave_f csave ;

  replicate_type rep ;
  terminate_type trm ;

  // and here we have the pointer to the copy of the grokkee cast
  // to void*.

  void * p_context ;

public:

  template < typename grokkee_type >
  grok_put_t ( const grokkee_type & grokkee )
  {
    typedef grokkee_type g_t ;

    // p_context is initialized with a copy of 'grokkee'

    p_context = new g_t ( grokkee ) ;

    // the std::functions are initialized with wrappers taking
    // p_context and a set of arguments which are passed on to
    // the grokkee's member functions.

    vinit = [] ( void * & p_ctx ,
                 const crd_t & crd )
          {
            auto p_gk = static_cast<g_t*> ( p_ctx ) ;
            p_gk->init ( crd ) ;
          } ;

    vsave = [] ( void * & p_ctx ,
                 value_v & v )
          {
            auto p_gk = static_cast<g_t*> ( p_ctx ) ;
            p_gk->save ( v ) ;
          } ;

    csave = [] ( void * & p_ctx ,
                 value_v & v ,
                 const std::size_t & cap )
          {
            auto p_gk = static_cast<g_t*> ( p_ctx ) ;
            p_gk->save ( v , cap ) ;
          } ;

    rep = [] ( void * p_ctx ) -> void*
          {
            auto p_gk = static_cast<g_t*> ( p_ctx ) ;
            return new g_t ( *p_gk ) ;
          } ;

    trm = [] ( void * p_ctx )
          {
            auto p_gk = static_cast<g_t*> ( p_ctx ) ;
            delete p_gk ;
          } ;
  }

  // grok_put_t itself offers the typical member functions of
  // a put_t object, which in turn delegate to the stored
  // std::functions.

  void init ( const crd_t & crd )
  {
    vinit ( p_context , crd ) ;
  }

  void save ( value_v & trg )
  {
    vsave ( p_context , trg ) ;
  }

  void save ( value_v & trg ,
              const std::size_t & cap )
  {
    csave ( p_context , trg , cap ) ;
  }

  grok_put_t & operator= ( const grok_put_t & rhs )
  {
    // first copy the std::functions

    vinit = rhs.vinit ;
    vsave = rhs.vsave ;
    csave = rhs.csave ;
    rep = rhs.rep ;
    trm = rhs.trm ;

    // now use 'rep' to copy the 'hidden' grokkee

    p_context = rep ( rhs.p_context ) ;
    return *this ;
  }

  // copy construction delegates to copy assignment

  grok_put_t ( const grok_put_t & rhs )
  {
    *this = rhs ;
  }

  // finally, the d'tor destroys the context object in a type-safe
  // manner by passing it to 'trm', which knows how to cast it to
  // it's 'true' type and then calls delete on that.

  ~grok_put_t()
  {
    if ( p_context )
      trm ( p_context ) ;
  }
} ;

// grok_put is a factory function to 'grok' a put_t object.
// using ATD, the invocation to 'grok' some put_t x is simply
// auto gk = grok_put ( x ) ;
// gk can then be used wherever a put_t is required.

template < typename T ,     // fundamental type
           std::size_t N ,  // channel count
           std::size_t D ,  // dimensions
           std::size_t L ,  // lane count
           template < typename ,
                      std::size_t ,
                      std::size_t ,
                      std::size_t >
             class G >
grok_put_t < T , N , D , L > grok_put
  ( G < T , N , D , L > grokkee )
{
  return grok_put_t < T , N , D , L > ( grokkee ) ;
}

} ; // namespace zimt

#define ZIMT_PUT_H
#endif
