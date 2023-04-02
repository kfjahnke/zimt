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

/*! \file get_t.h

    \brief classes to provide input to an 'act' functor

    zimt::process generates input to the 'act' functor, working along
    the aggregation axis, while the coordinate's component along the
    other axes remain constant. This effectively 1D operation can
    be coded efficiently, whereas the 'act' functor would typically
    look at it's input as an nD entity and not be aware of the fact
    that only one of the input's components is 'moving'.
    This header provides a set of classes which are meant to fit into
    this slot. They follow a specific pattern which results from the
    logic in zimt_process, namely the two init and two increase
    overloads.
*/

#ifndef ZIMT_GET_T_H

namespace zimt
{

/// class get_crd is an implementation of a 'get_t' class which the
/// rolling-out code uses to produce input values to the functor 'act'.
/// This specific class provides discrete nD coordinates. The c'tor
/// receives the 'hot' axis along which the coordinate will vary,
/// the other components remain constant.

template < typename T ,    // elementary type
           std::size_t N , // number of channels
           std::size_t D , // dimension of the view/array
           std::size_t vsize = zimt::vector_traits < T > :: vsize >
struct get_crd
{
  typedef zimt::xel_t < T , N > value_t ;
  typedef zimt::simdized_type < value_t , vsize > value_v ;
  typedef typename value_v::value_type value_ele_v ;
  typedef zimt::xel_t < long , D > crd_t ;

  const std::size_t d ;

  // get_crd's c'tor receives the processing axis

  get_crd ( const std::size_t & _d = 0 )
  : d ( _d )
  { }

  // init is used to initialize the vectorized value to the value
  // it should hold at the beginning of the run. The discrete
  // coordinate 'crd' gives the location of the first value, and
  // the get_crd infers the start value from it. Note that we can't
  // simply start with a value of zero for the 'hot' axis, because
  // zimt::process may 'cut up' the 1D subarrays further into
  // 'segments'. Since this is an archetypal implementation of a
  // get_t class, I'll explain the signature some more: The first
  // argument is a reference to the object residing in zimt:process
  // which will be passed to the act functor. This is the datum we
  // need to generate here. The second argument is a const reference
  // to the initial nD coordinate of the 'run'. This is a scalar
  // value, whereas the target is a simdized value. All we need to
  // do here is to add iota() to the 'hot' component, while the
  // other components are simply assigned, resulting in components
  // which are SIMD data, but share the same value.

  void init ( value_v & trg , const crd_t & crd ) const
  {
    trg = crd ;
    trg [ d ] += value_ele_v::iota() ;
  }

  // 'capped' variant. This is only needed if the current segment is
  // so short that no vectors can be formed at all. We fill up the
  // target value with the last valid datum, using the 'stuff'
  // member function provided for xel of simdized data.
  // If zimt::process has at least one full vector in it's current
  // 'run', the target datum (the intended input to the 'act' functor)
  // already has data which are deemed to be valid, so this (capped)
  // variant won't be called, avoiding the overhead needed for
  // 'stuffing' the unused lanes with the last 'genuine' value.
  // Why not leave the 'stuffing' to the caller (zimt::process)?
  // Because some get_t classes may, e.g., access memory to obtain
  // input, and they must be aware of the cap, to avoid 'overshooting'
  // which might result in a memory fault.

  void init ( value_v & trg ,
              const crd_t & crd ,
              std::size_t cap ) const
  {
    init ( trg , crd ) ;
    trg.stuff ( cap ) ;
  }

  // increase modifies it's argument to contain the next value.
  // The argument is, again, the 'target' datum, the intended input
  // to the 'act' functor. It's a simdized value residing in
  // zimt::process. Here, we are aware of the 'hot' axis (we received
  // it in the c'tor), so all we need to do is to increase the
  // 'hot' component by the lane count. Note how in this get_t
  // class, we don't hold any mutable state at all: the init and
  // increase member functions directly modify the target datum.
  // This is the reason for handling all arguments to these member
  // functions by reference. You may also have noticed that all
  // member functions are declared const, showing that they don't
  // change the get_t object itself.

  void increase ( value_v & trg ) const
  {
    trg [ d ] += vsize ;
  }

  // 'capped' variant. This is called after all vectors in the current
  // segment have been processed, so the lanes in trg beyond the cap
  // should hold valid data, and 'stuffing' them with the last datum
  // before the cap is optional. Note how we don't call 'plain'
  // increase: we want to avoid touching the values at and beyond
  // the cap, unless _stuff is true: then we fill them with the last
  // 'genuine' value.

  void increase ( value_v & trg ,
                  std::size_t cap ,
                  bool _stuff = true ) const
  {
    auto mask = ( value_ele_v::IndexesFromZero() < int(cap) ) ;
    trg [ d ] ( mask ) += T(vsize) ;
    if ( _stuff )
    {
      trg [ d ] ( ! mask ) = trg [ d ] [ cap - 1 ] ;
    }
  }
} ;

// class loader is a get_t which loads data from a zimt::view.
// With this get_t, 'process' can be used to implement 'coupled_f'
// We have two variants which differ only in the use of the argument
// 'stride' when calling 'bunch'. TODO: may not make a performance
// difference to call bunch with stride==1 vs. call without stride,
// in which case the unstrided variant would be superfluous.

template < typename T ,
           std::size_t N ,
           std::size_t M = N ,
           std::size_t L = zimt::vector_traits < T > :: vsize >
struct loader
{
  typedef zimt::xel_t < T , N > value_t ;
  typedef zimt::simdized_type < value_t , L > value_v ;
  typedef typename value_v::value_type value_ele_v ;
  typedef zimt::xel_t < long , M > crd_t ;
  typedef zimt::simdized_type < long , L > crd_v ;

  const std::size_t d ;
  const zimt::view_t < M , value_t > src ;
  const value_t * p_src ;
  const std::size_t stride ;

  // get_t's c'tor receives the zimt::view providing data and the
  // 'hot' axis. It extracts the strides from the source view.

  loader ( const zimt::view_t < M , value_t > & _src ,
           const std::size_t & _d = 0 )
  : src ( _src ) , d ( _d ) , stride ( _src.strides [ _d ] )
  { }

  // init is used to initialize the 'target'' value to the value
  // it should hold at the beginning of the run. The discrete
  // coordinate 'crd' gives the location of the first value, and the
  // 'loader' fills it's target (the intended input for the 'act'
  // functor) with the first set of values from the source view.
  // Note how 'loader' has mutable state: the pointer p_src points
  // to data in the source view and it varies with the progess along
  // the current 'run'.

  void init ( value_v & trg , const crd_t & crd )
  {
    p_src = & ( src [ crd ] ) ;
    trg.bunch ( p_src , stride ) ;
    p_src += L * stride ;
  }

  // the 'capped' variant of 'init' only fills in the values below
  // the cap. For class loader, this is very important: there must
  // not be any memory access beyond the limit given by 'cap' to
  // avoid a memory fault.

  void init ( value_v & trg ,
              const crd_t & crd ,
              std::size_t cap )
  {
    p_src = & ( src [ crd ] ) ;
    trg.bunch ( p_src , stride , cap , true ) ;
  }

  // 'increase' fetches a full vector of data from the source view
  // and increments the pointer to the data in the view to the next
  // position.

  void increase ( value_v & trg )
  {
    trg.bunch ( p_src , stride ) ;
    p_src += L * stride ;
  }

  // The 'capped' variant only fetches a part of a full vector
  // and doesn't increase the pointer: the capped increase is
  // always the last call in a run.

  void increase ( value_v & trg ,
                  std::size_t cap ,
                  bool stuff = true )
  {
    trg.bunch ( p_src , stride , cap , stuff ) ;
  }
} ;

// unstrided_loader is a variant of class loader where the stride
// of the source view along the 'hot' axis is one.
// TODO: it may not be necessary to have this variant.
// test performance against using plain loader.

template < typename T ,
           std::size_t N ,
           std::size_t M = N ,
           std::size_t L = zimt::vector_traits < T > :: vsize >
struct unstrided_loader
: public loader < T , N , M , L >
{
  typedef loader < T , N , M , L > base_t ;

  using typename base_t::crd_t ;
  using typename base_t::value_t ;
  using typename base_t::value_v ;
  using base_t::src ;
  using base_t::p_src ;

  unstrided_loader ( const zimt::view_t < M , value_t > & _src ,
                     const std::size_t & _d = 0 )
  : base_t ( _src , _d )
  {
    assert ( _src.strides [ _d ] == 1 ) ;
  }

  void init ( value_v & trg , const crd_t & crd )
  {
    p_src = & ( src [ crd ] ) ;
    trg.bunch ( p_src ) ;
    p_src += L ;
  }

  void init ( value_v & trg ,
              const crd_t & crd ,
              std::size_t cap )
  {
    p_src = & ( src [ crd ] ) ;
    trg.bunch ( p_src , 1 , cap , true ) ;
  }

  void increase ( value_v & trg )
  {
    trg.bunch ( p_src ) ;
    p_src += L ;
  }

  void increase ( value_v & trg ,
                  std::size_t cap ,
                  bool stuff = true )
  {
    trg.bunch ( p_src , 1 , cap , stuff ) ;
  }
} ;

// vloader loads vectorized data from a zimt::view_t of vectorized
// values. This is a good option for storing intermediate results,
// because it can use efficient SIMD load operations rather than
// having to deinterleave the data from memory holding xel of T.
// To store data so they can be loaded from vectorized storage,
// use class vstorer (see put_t.h)
// The design is so that vloader can operate within the logic used
// by zimt::process: the discrete coordinate taken by 'init' refers
// to 'where the datum would be in an array of value_t'. The array
// of value_v which the data are stored to has to be suitably sized:
// along the 'hot' axis, the size should be so that, multiplied with
// the lane count, it will be greater or equal the 'notional' size
// of the unvectorized array - for efficiency reasons, class vstorer
// will only load 'full' simdized values from the array of simdized
// data. Along the other axes, the size must be the same as the
// 'notional' size of the equivalent unvectorized array.

template < typename T ,
           std::size_t N ,
           std::size_t M = N ,
           std::size_t L = zimt::vector_traits < T > :: vsize >
struct vloader
{
  typedef zimt::xel_t < T , N > value_t ;
  typedef zimt::simdized_type < value_t , L > value_v ;
  typedef typename value_v::value_type value_ele_v ;
  typedef zimt::xel_t < long , M > crd_t ;
  typedef zimt::simdized_type < long , L > crd_v ;

  const std::size_t d ;
  const zimt::view_t < M , value_v > src ;
  const value_v * p_src ;
  const std::size_t stride ;
  const std::size_t shift ;

  // get_t's c'tor receives the zimt::view providing data and the
  // 'hot' axis. It extracts the strides from the source view.

  vloader ( const zimt::view_t < M , value_v > & _src ,
            const std::size_t & _d = 0 )
  : src ( _src ) , d ( _d ) , stride ( _src.strides [ _d ] ) ,
    shift ( std::log2 ( L ) )
  {
    assert ( ( 1 << shift ) == L ) ;
  }

  // init is used to initialize the 'target'' value to the value
  // it should hold at the beginning of the run. The discrete
  // coordinate 'crd' gives the location of the first value, and the
  // 'loader' fills it's target (the intended input for the 'act'
  // functor) with the first set of values from the source view.
  // Note how 'loader' has mutable state: the pointer p_src points
  // to data in the source view and it varies with the progess along
  // the current 'run'.

  void init ( value_v & trg , const crd_t & crd , std::size_t cap = 0 )
  {
    auto crdv = crd ;
    crdv [ d ] <<= shift ;
    p_src = & ( src [ crdv ] ) ;
    trg = *p_src ;
    p_src += stride ;
  }

  // 'increase' fetches a full vector of data from the source view
  // and increments the pointer to the data in the view to the next
  // position.

  void increase ( value_v & trg , std::size_t cap = 0 ,
                  bool stuff = false )
  {
    trg = *p_src ;
    p_src += stride ;
  }
} ;

// class permute is a get_t which puts together the components of
// the values it produces from separate 1D arrays, one per axis.
// This can be used to set up grids which don't have regular steps
// from one slice to the next, like for the grid_eval function
// that uses permute.

template < typename T ,
           std::size_t N ,
           std::size_t L = zimt::vector_traits < T > :: vsize >
struct permute
{
  typedef zimt::xel_t < T , N > value_t ;
  typedef zimt::simdized_type < value_t , L > value_v ;
  typedef typename value_v::value_type value_ele_v ;
  typedef zimt::xel_t < long , N > crd_t ;
  typedef zimt::simdized_type < long , L > crd_v ;

  const std::size_t d ;
  const std::array < zimt::view_t < 1 , T > , N > src ;
  const T * p_src ;
  const std::size_t stride ;

  // permute's c'tor receives a std::vector of 1D zimt::views providing
  // per-component data and the 'hot' axis.

  permute ( const std::array < zimt::view_t < 1 , T > , N > & _src ,
            const std::size_t & _d = 0 )
  : src ( _src ) ,
    d ( _d ) ,
    stride ( _src [ _d ] . strides [ 0 ] )
  { }

  // init is used to initialize the 'target' value to the value
  // it should hold at the beginning of the run. The discrete
  // coordinate 'crd' gives the location of the first value, and the
  // 'permute' fills it's target (the intended input for the 'act'
  // functor) with the first set of values from the source view.
  // Note how 'permute' has mutable state: the pointer p_src points
  // to data in the source view and it varies with the progess along
  // the current 'run'.

  void init ( value_v & trg , const crd_t & crd )
  {
    p_src = src [ d ] . data() + crd [ d ] ;
    for ( std::size_t ch = 0 ; ch < N ; ch++ )
    {
      if ( ch == d )
      {
        trg [ ch ] . rgather ( p_src , stride ) ;
      }
      else
      {
        trg [ ch ] = src [ ch ] [ crd [ ch ] ] ;
      }
    }
    p_src += L * stride ;
  }

  // the 'capped' variant of 'init' only fills in the values below
  // the cap. For class permute, this is very important: there must
  // not be any memory access beyond the limit given by 'cap' to
  // avoid a memory fault.

  void init ( value_v & trg ,
              const crd_t & crd ,
              std::size_t cap )
  {
    p_src = src [ d ] . data() + crd [ d ] ;
    for ( std::size_t ch = 0 ; ch < N ; ch++ )
    {
      if ( ch == d )
      {
        for ( std::size_t e = 0 ; e < cap ; e++ )
          trg [ ch ] [ e ] = p_src [ e * stride ] ;
        for ( std::size_t e = cap ; e < L ; e++ )
          trg [ ch ] [ e ] = trg [ ch ] [ cap - 1 ] ;
      }
      else
      {
        trg [ ch ] = src [ ch ] [ crd [ ch ] ] ;
      }
    }
  }

  // 'increase' fetches a full vector of data from the source view
  // and increments the pointer to the data in the view to the next
  // position.

  void increase ( value_v & trg )
  {
    trg [ d ] . rgather ( p_src , stride ) ;
    p_src += L * stride ;
  }

  // The 'capped' variant only fetches a part of a full vector
  // and doesn't increase the pointer: the capped increase is
  // always the last call in a run.

  void increase ( value_v & trg ,
                  std::size_t cap ,
                  bool stuff = true )
  {
    for ( std::size_t e = 0 ; e < cap ; e++ )
      trg [ d ] [ e ] = p_src [ e * stride ] ;
    for ( std::size_t e = cap ; e < L ; e++ )
      trg [ d ] [ e ] = trg [ d ] [ cap - 1 ] ;
  }

} ;

// join_t is a get_t which loads data from N arrays of T
// (fundamentals) into simdized packets of data suitable for
// processing with the 'act' functor. This is a common scenario
// when the channels of multi-channel raster data are stored as
// separate arrays. join_t's c'tor receives a std::vector of
// N views to per-channel arrays, which should all agree in
// shape. It loads vectors of data from all arrays in turn,
// into components of the simdized datum it provides as input
// to the act functor.
// TODO: write the corresponding put_t class

template < typename T ,     // fundamental type
           std::size_t N ,  // channel count
           std::size_t D ,  // dimensions
           std::size_t L >  // lane count
struct join_t
{
  typedef zimt::xel_t < T , N > value_t ;
  typedef zimt::simdized_type < value_t , L > value_v ;
  typedef typename value_v::value_type value_ele_v ;
  typedef zimt::xel_t < long , D > crd_t ;

  // const std::size_t d ; // processing axis - 0 or 1 for a 2D array

  // source of data: N D-dimensional views to the fundamental type

  typedef std::array < zimt::view_t < D , T > , N > src_t ;
  src_t src ;

  zimt::xel_t < const T * , N > pickup ; // source pointers
  zimt::xel_t < long , N > stride ;      // strides of source arrays

  join_t ( const src_t & _src )
  : src ( _src )
  {
    // copy out the strides of the source arrays

    for ( int ch = 0 ; ch < N ; ch++ )
    {
      stride [ ch ] = src [ ch ] . strides [ 0 ] ;
    }
  }

  // init is used to initialize the vectorized value to the value
  // it should hold at the beginning of the peeling run. The discrete
  // coordinate 'crd' gives the location of the first value, and this
  // function infers the start value from it. Here we set the pickup
  // pointers and perform a 'regular gather' to get the first batch
  // of values. Note how we could specialize the code to use load
  // instructions instead of rgather if the stride is 1.

  void init ( value_v & v , const crd_t & crd )
  {
    for ( int ch = 0 ; ch < N ; ch++ )
    {
      pickup [ ch ] = & ( src [ ch ] [ crd ] ) ;
      v [ ch ] . rgather ( pickup [ ch ] , stride [ ch ] ) ;
    }
  }

  // 'capped' variant. This is only needed if the current segment is
  // so short that no vectors can be formed at all. We fill up the
  // target value with the last valid datum.

  void init ( value_v & trg ,
              const crd_t & crd ,
              std::size_t cap )
  {
    for ( int ch = 0 ; ch < N ; ch++ )
    {
      pickup [ ch ] = & ( src [ ch ] [ crd ] ) ;
      for ( std::size_t e = 0 ; e < cap ; e++ )
        trg [ ch ] [ e ] = pickup [ ch ] [ e * stride [ ch ] ] ;
      trg.stuff ( cap ) ;
    }
  }


  // increase modifies it's argument to contain the next value, or
  // next vectorized value, respectively - first we increase the
  // pickup pointers, then we get the data from that location.

  void increase ( value_v & trg )
  {
    for ( int ch = 0 ; ch < N ; ch++ )
    {
      pickup [ ch ] += L * stride [ ch ] ;
      trg [ ch ] . rgather ( pickup [ ch ] , stride [ ch ] ) ;
    }
  }

  // 'capped' variant. This is called after all vectors in the current
  // segment have been processed, so the lanes in trg beyond the cap
  // should hold valid data, and 'stuffing' them with the last datum
  // before the cap is optional.

  void increase ( value_v & trg ,
                  std::size_t cap ,
                  bool _stuff = true )
  {
    for ( int ch = 0 ; ch < N ; ch++ )
    {
      pickup [ ch ] += L * stride [ ch ] ;
      for ( std::size_t e = 0 ; e < cap ; e++ )
        trg [ ch ] [ e ] = pickup [ ch ] [ e * stride [ ch ] ] ;
      if ( _stuff )
        trg.stuff ( cap ) ;
    }
  }
} ;

// linspace_t produces input to the 'act' functor which is
// like a NumPy ngrid: The data start out at a given initial
// value and increase (or decrease) by a fixed amount along
// each axis. It's also possible to pass discrete coordinates
// on to the 'act' functor and do the range transformation
// there, but the 'act' functor should not have a notion of
// which axis is 'hot', but rather take the entire simdized
// datum it receives as input 'at face value', so doing the
// transformation there would involve all channels, whereas
// linspace_t is aware of the 'hot' axis and can produce the
// simdized data more efficiently, by only modifying the
// component of the simdized datum pertaining to the hot axis.

template < typename T ,     // elementary type
           std::size_t N ,  // number of channels
           std::size_t L >  // lane count
struct linspace_t
{
  typedef zimt::xel_t < T , N > value_t ;
  typedef zimt::simdized_type < value_t , L > value_v ;
  typedef typename value_v::value_type value_ele_v ;
  typedef zimt::xel_t < long , N > crd_t ;
  typedef zimt::simdized_type < crd_t , L > crd_v ;
  typedef typename crd_v::value_type crd_ele_v ;

  const std::size_t d ;
  const value_t start ;
  const value_t step ;

  // linspace_t's c'tor receives start, step and axis. Note how
  // start and step are N-dimensional; each component gives the
  // intended value for the corresponding axis.

  linspace_t ( const value_t & _start ,
               const value_t & _step ,
               const std::size_t & _d )
  : start ( _start ) ,
    step ( _step ) ,
    d ( _d )
  { }

  // init is used to initialize the vectorized value to the value
  // it should hold at the beginning of the peeling run. The discrete
  // coordinate 'crd' gives the location of the first value, and this
  // function infers the start value from it. The scalar value will
  // not be used until peeling is done, so it isn't initialized here.

  void init ( value_v & trg , const crd_t & crd )
  {
    trg = step * crd + start ;
    trg [ d ] += value_ele_v::iota() * step [ d ] ;
  }

  // 'capped' variant. This is only needed if the current segment is
  // so short that no vectors can be formed at all. We fill up the
  // target value with the last valid datum.

  void init ( value_v & trg ,
              const crd_t & crd ,
              std::size_t cap )
  {
    trg = step * crd + start ;
    for ( std::size_t e = 0 ; e < cap ; e++ )
      trg [ d ] [ e ] += T ( e ) * step [ d ] ;
    trg.stuff ( cap ) ;
  }

  // increase modifies it's argument to contain the next value

  void increase ( value_v & trg )
  {
    trg [ d ] += ( step [ d ] * L ) ;
  }

  // 'capped' variant. This is called after all vectors in the current
  // segment have been processed, so the lanes in trg beyond the cap
  // should hold valid data, and 'stuffing' them with the last datum
  // before the cap is optional.

  void increase ( value_v & trg ,
                  std::size_t cap ,
                  bool _stuff = true )
  {
    for ( std::size_t e = 0 ; e < cap ; e++ )
      trg [ d ] [ e ] += ( step [ d ] * L ) ;
    if ( _stuff )
      trg.stuff ( cap ) ;
  }
} ;

// For brevity, here's a using declaration deriving the default
// get_t from the actor's type and the array's/view's dimension.

template < typename act_t , std::size_t dimension >
using norm_get_crd
  = get_crd
  < typename act_t::in_ele_type ,
    act_t::dim_in ,
    dimension ,
    act_t::vsize > ;

} ; // namespace zimt

#define ZIMT_GET_T_H
#endif