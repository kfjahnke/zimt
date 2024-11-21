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

/// \file tiles.h
///
/// \brief simple tile storage for large arrays
///
/// This is new code, and in parts it's 'naive' in so far as
/// that it accepts metrics without plausibility tests and
/// assumes file operations will succeed. As the design
/// 'solidifies', appropriate checks can be added.
///
/// If the 'notional' shape becomes very large, holding the
/// entire corpus of data in memory may become impossible. The
/// data have to be held in mass storage and processed in
/// increments, reading them before and writing them back
/// after processing.
///
/// This header codes a notion of 'tiles': small-ish arrays of
/// memory which provide a part of the data corresponding to
/// a part of the 'notional' shape. These tiles are linked to
/// files which hold the data when the tiles are not active.
/// When a tile is opened for reading and the file is present
/// and sufficiently large, the data are loaded from the file.
/// If there is no file or it doesn't have enough data, that's
/// not an error: The tile, when active, holds a chunk of memory
/// which is left as it is if no data can be had from the file.
/// When a tile is closed, it's data may be written back to the
/// file (provided that 'write_to_disk' was set). Currently
/// the code silently assumes opening the file for writing will
/// succeed. Files for tiles are simply memory copied to disk,
/// there are no metadata nor any provisions to make the data
/// portable. There is no compression, either. Such improvements
/// can be added later by diversifying the access to mass storage;
/// class tile_store is a base class with virtual members which
/// user code can override if the base class' members aren't
/// suitable.
///
/// Next is a class bundling a set of tiles which, together,
/// represent the storage for an entire (possibly very large)
/// 'notional' shape. This is coded to cooperate with specific
/// get_t and put_t objects to be used with zimt::process; the
/// main intent is currently to get a smooth zimt::process run
/// over a large notional shape to read from, process and store
/// to tile stores with minimal overhead. This is a narrower
/// scope than access to arbitrary tiles as they are needed,
/// and can be coded so that - even with concurrent access -
/// overhead for coordinating data access can be kept low,
/// because the access pattern is known beforehand: it follows
/// the pattern given by zimt::process. Nevertheless, accessing
/// mass storage is intrinsically slow, so don't expect miracles.
///
/// Finally, we have templates of get_t and a put_t objects
/// interacting with the tile store, in order to make tiled
/// storage available as input or output (or both) of
/// zimt::process. With this integration into
/// zimt::process, we gain access to code which, for example,
/// can run a reduction over an entire tile store (or parts
/// of it) without having to load the entire store to memory.
/// We can also process entire tile stores (or parts) with SIMD
/// code like any other data source/sink, using the same act
/// functors as we'd use for non-tiled storage.
///
/// Apart from accessing single large data sets, another use
/// case which I intend to exploit is work with several data
/// sets at once. Even if one of these sets may fit into memory,
/// several of them may not, which is where tiled storage comes
/// in. As long as processing is limited to parts of the data
/// in a given time window, such collections of data sets can
/// be processed together, i.e. to form a synopsis or correlation.
/// a typical use case would be a tile store of N*N tiles. The
/// code in this header will, on average, limit the amount of
/// tiles which are held 'open' (so, with a memory footprint)
/// to N, which, with rising N, is substantially lower than the
/// N*N tiles of the entire store. If this still exceeds the
/// capacity of the system, access to the data can be cut up
/// to partial workloads by setting appropriate boundaries in
/// the 'loading bill' - with a window which makes the 'hot'
/// axis shorter, while the other axes won't make a difference.
///
/// A special type of 'tile' is as 'wide' as the notional shape
/// and the other dimensions have size one. So this
/// type of tile represents an entire line of the notional shape.
/// With an appropriate override of the tile loading and storing
/// functions this can be exploited to communicate with many
/// codecs, which often provide access to entire lines of, e.g.,
/// images. Set up this way, the resulting tile_loader/storer
/// objects can be used to directly read from/write to (possibly
/// very large) files on mass storage and only expose a few lines
/// at a time to the 'act' functor. Note that such a scheme would
/// not store the 'tiles' as individual files - this is not required
/// by the design, the overrides can obtain the tile data in any way
/// they like. Using this method should make processing large images
/// and similar data sets simple and efficient.
///
/// The tile_t and tile_store_t classes plus the code to access
/// them are designed to work with zimt::process, so if you 'bend'
/// them to other uses, you may at times find they are not perfectly
/// general or make certain assumptions. I try and point this out in
/// the comments, so to use the code outside the zimt::process
/// context will require careful planning. With zimt::process,
/// everything should 'snap into place'.

#include <stdio.h>
#include <utility>
#include <vector>
#include <deque>

#include "xel.h"
#include "array.h"

#if defined(ZIMT_TILES_H) == defined(HWY_TARGET_TOGGLE)
  #ifdef ZIMT_TILES_H
    #undef ZIMT_TILES_H
  #else
    #define ZIMT_TILES_H
  #endif

HWY_BEFORE_NAMESPACE() ;
BEGIN_ZIMT_SIMD_NAMESPACE(zimt)

#ifndef ZIMT_SINGLETHREAD
std::mutex stdout_mutex ;
#endif
std::atomic < long > load_count ( 0 ) ;
std::atomic < long > store_count ( 0 ) ;

// tile_t acts as conduit to a tile's data and encodes it's
// interaction with files. The struct itself is lightweight,
// and memory for storage of the tile's actual data is allocated
// only if necessary.

template < typename T ,    // fundamental type
           std::size_t N , // number of channels
           std::size_t D > // dimensionality
struct tile_t
{
  typedef xel_t < T , N > value_t ;
  typedef array_t < D , value_t > storage_t ;
  typedef xel_t < std::size_t , D > shape_type ;
  typedef xel_t < long , D > index_type ;

  const shape_type shape ;
  storage_t * p_data = nullptr ;

  // allocate allocates the storage: an array of value_t with
  // shape 'shape'. Note that the array's 'owned' memory and
  // the view which it holds to it are not touched in any way;
  // the code will assume that the array holds a single
  // contiguous, compact block of memory with as many value_t
  // as the shape's inner product.

  // TODO: introduce tiles with additional support around the
  // edges, e.g. for b-spline coeffcients, to allow evaluation
  // without having to assemble a coefficient window from
  // several tiles

  // TODO: abstraction of allocation. I'd like to have allocation
  // obtain memory from a pool of tiles held in one large-ish
  // contiguous block of memory. Then each tile manifest in this
  // pool (rather than latent in the form of a file on disk) can
  // be characterized uniquely by a fixed offset from the pool's
  // base address. The scheme would work with power-of-two tile
  // extents. Notional coordinates coming in can be split into
  // a tile coordinate and an in-tile coordinate (this can be
  // done efficiently with mask and shift operations due to the
  // power-of-two-based extent), and the tile coordinate can be
  // translated into an tile index from the pool base address.
  // Using the pool's shape and strides (stride one in dimension
  // zero can be prescribed), the combination of the tile's index
  // from the pool base address and the in-tile coordinate yields
  // an offset from the pool base address where the corresponding
  // data can be found. If an access attempts to acces a tile which
  // is not manifest, an exception is thrown and all the tiles which
  // are needed to satisfy the access are loaded into the pool.
  // This access scheme can easily be simdized: a set of incoming
  // notional coordinates yields a set of tile coordinates and a
  // set of in-tile coordinates. The tile coordinates are converted
  // to offsets (by multiplication with the tile store's strides
  // amd summing up) and the offsets used in a gather operation to
  // retrieve the tile indexes into the pool. If any of these come
  // out invalid (null, negative, any easily detectable property)
  // the exception is thrown, resultig in the - likely slow -
  // process of making all needed tiles manifest. 'Regular'
  // processing (without the exception) is fast, though, and if
  // especially detrimental access patterns (like, truly random
  // access) are avoided, exceptions should be relatively rare.

  // void allocate()
  // {
  //   assert ( p_data == nullptr ) ;
  //   p_data = new storage_t ( shape ) ;
  // }

  // tile_t's c'tor only sets the shape; allocating storage or
  // access to files is done later with specific functions as
  // the need arises. A tile store is - at the tile level -
  // a sparse storage medium: tiles which aren't ever 'touched'
  // by processing have neither memory nor mass storage footprint.
  // So very large notional shapes can be used without necessarily
  // producing any load at all, and only when processing touches
  // a specific tile, it becomes manifest.

  tile_t ( const shape_type & _shape )
  : shape ( _shape )
  { }

  // 'provide' obtains a pointer into the tile at location 'crd'
  // and writes it to 'p_memory'. 'tail' receives the number of
  // values left along axis d. This function is called by the
  // tile-based get_t/put_t objects to obtain the appropriate
  // data pointer and number of values which the current tile
  // can still provide until the 'right edge' of the tile is
  // reached. Every call to 'provide' also marks the entry into
  // a new 'transit', and when you look at the invocations of
  // 'provide', you can see that each of them is followed by
  // code counting up the 'done' values which record the number
  // of transits the code performs.

  void provide ( const index_type & crd , // in-tile coordinate
                 const std::size_t & d ,  // hot axis
                 value_t * & p_memory ,   // returns pointer
                 std::size_t & tail )     // and number of values
               const
  {
    tail = shape [ d ] - crd [ d ] ;
    p_memory = & ( (*p_data) [ crd ] ) ;
  }

  // overload setting a const pointer to memory; this overload
  // is used by tile_loader which won't modify the tile's memory

  void provide ( const index_type & crd , // in-tile coordinate
                 const std::size_t & d ,  // hot axis
                 const value_t * & p_memory , // returns pointer
                 std::size_t & tail )     // and number of values
               const
  {
    tail = shape [ d ] - crd [ d ] ;
    p_memory = & ( (*p_data) [ crd ] ) ;
  }
} ;

// The tile_store class (below) accesses tiles via a 'tether'
// structure, a small struct holding a std::mutex to mediate
// thread-safe access, the number of users currently accessing
// the tile pointer, and the tile pointer itself.
// user code gaining access to a tile pointer is allowed to
// interface with the tile's memory as it sees fit, but it
// is assumed that several threads which access the same tile
// concurrently will *not access the same data* inside this
// memory. With this access model, user code can simply hold
// the tile pointer and access it without mutex protection.
// This is an important optimization, reducing the use of
// mutex-protected access dramatically, but it's scope is
// limited to use cases where it's guaranteed that threads
// won't 'step on each other's toes', like in zimt::process:
// there, each value is accessed precisely once, it's a
// plain traversal of the data. With the multithreaded access,
// the precise sequence in which data are processed and the
// concrete thread which processes them is knot known and
// 'sorts itself out', but it's guaranteed that no datum is
// accessed more than once, so there doesn't have to be
// a protection against conflicting data access by several
// concurrent threads.

template < typename tile_t >
struct tether_t
{
  // for single-threaded processing, we can do without the mutex

#ifndef ZIMT_SINGLETHREAD
  std::mutex tile_mutex ;
#endif

  // at the beginning of a 'run', this value is set to the
  // number of 'fragments' the tile will 'provide' during the
  // run. tile_loader and tile_storer objects cumulate the number
  // of fragments they provide in their own buffers and only
  // interact with the tile's (so, this) datum when they move
  // on to a new row of tiles, minimizing mutex-protected
  // access. When a new row is encountered, the cumulated
  // values are subtracted from this here datum. The result
  // of the subtraction is monitored, and when it sinks to
  // zero, this is the signal that the tile has played out
  // it's role in the zimt::process invocation and can safely
  // be 'dropped', because it won't be accessed any more
  // during the run.

  std::size_t due = 0 ;

  // this is the tile pointer itself

  tile_t * p_tile = nullptr ;

  tether_t()
  : p_tile ( nullptr ) ,
    tile_mutex() ,
    due ( 0 )
  { }
    
} ;

// tile_store_t holds an array of tether_t to mediate access
// to the tiles. The amount of tiles needed is calculated from
// the 'notional' shape passed to the c'tor and the intended
// shape of individual tiles. This object provides member
// functions to coordinate access to the tiles; code which is
// meant to be called by the individual processing threads
// to gain access to tiles and to end their access are
// appropriately mutex-protected, making sure that only one
// thread at a time can access shared resources like the
// tether structures and the pointers they hold.
// Note that this is a lightweight object; setting it up will
// allocate and initialize some memory, but there won't be any
// access to mass storage or other lengthy operations. It's
// coded to fit well into an RAII scheme, so that it's set up
// right before it's used with zimt::process and destructed
// afterwards. Note that, if you intend to re-use a tile_store_t
// object after the zimt::process run, you should call 'close'
// afterwards - if the tile_store_t object is destructed, this
// is done automatically be the d'tor.

template < typename T , std::size_t N , std::size_t D >
struct tile_store_t
{
  typedef tile_t < T , N , D > tile_type ;
  typedef xel_t < T , N > value_t ;
  typedef xel_t < std::size_t , D > shape_type ;
  typedef xel_t < long , D > index_type ;
  typedef tether_t < tile_type > tether_type ;

  const shape_type array_shape ; // 'notional' shape
  const shape_type tile_shape ;  // shape of individual tiles
  const shape_type store_shape ; // shape of the array of tether_t

  // if this flag is set, tiles will be written to disk when they
  // are 'dropped'.

  bool write_to_disk = false ;

  // if this flag is set, data will be read from disk if they
  // are available

  bool read_from_disk = false ;

  // array holding tether_type, controlling access to the tiles

  array_t < D , tether_type > store ;

  // base name of the files the store will access

  std::string basename ;

  // creating and destroying the in-memory representation of tiles
  // is done with virtual functions, so that derived classes can
  // override the de/allocation process, e.g. to use a tile pool.
  // Initially I coded these as pure virtual member functions, but
  // I see no harm in having them here in the base class, because
  // they are very simple and general and will be adequate for
  // most purposes.

protected:

  virtual tile_type * allocate_tile()
  {
    auto p_tile = new tile_type ( tile_shape ) ;
    p_tile->p_data = new typename tile_type::storage_t ( tile_shape ) ;
    return p_tile ;
  }

  virtual void deallocate_tile ( tile_type * p_tile )
  {
    delete p_tile->p_data ;
    delete p_tile ;
  }

private:

  // helper function to provide store_shape in the c'tor
  // for the sake of efficiency, we only use precisely one size
  // of tile, even though tiles on the edges of the notional
  // shape may be only partially filled with data. We assume
  // that this will not produce a significant amount of 'dead'
  // memory. Of course this assumption is false if the tile
  // size is large compared to the notionaly shape, and the
  // waste gets worse with higher dimensionality of the construct.

  static shape_type get_store_shape
    ( const shape_type & array_shape ,
      const shape_type & tile_shape )
  {
    shape_type store_shape ;
    for ( std::size_t i = 0 ; i < D ; i++ )
    {
      store_shape[i] = array_shape[i] / tile_shape[i] ;
      if ( array_shape[i] % tile_shape[i] )
        store_shape[i]++ ;
    }
    return store_shape ;
  }

  // 'dropping' a tile optionally flushes it's content to
  // a file, then frees the memory for data. this should only
  // be called by mutex-protected code or when it's assured
  // that only one thread is interacting with the tile store,
  // so this member function is also private.

  void drop ( const index_type & tile_index )
  {
    auto & tether ( store [ tile_index ] ) ; // shorthand

    if ( tether.p_tile != nullptr )
    {
      if ( write_to_disk )
      {
        store_tile ( tether.p_tile , tile_index ) ;
      }
      // delete tether.p_tile->p_data ;
      // delete tether.p_tile ;
      deallocate_tile ( tether.p_tile ) ;
      tether.p_tile = nullptr ;
    }
  }

  // calculate_due sets the 'due' member in the tether_t objects
  // in the store. This is done separately for read and write
  // access - if the tile store is opened for read/write
  // operation, this function will be called twice, once with
  // 'read_not_write' true and once false. this is due to the
  // offsets in the bill, which can be different for the read
  // and write part. Apart from that, the code is the same.
  // Using two distinct calls allows us all three use cases:
  // reading only, writing only, and read/write.

  void calculate_due ( const zimt::bill_t & bill ,
                       bool read_not_write )
  {
    // what is the range of coordinates we'll see in this run?

    index_type offset ;

    // decode the limits and offsets in the bill

    if ( read_not_write )
      offset = zimt::decode_bill_vector < D > ( bill.get_offset ) ;
    else
      offset = zimt::decode_bill_vector < D > ( bill.put_offset ) ;

    auto low = zimt::decode_bill_vector < D > ( bill.lower_limit ) ;

    auto high = array_shape ;

    if ( bill.upper_limit.size() )
      high = zimt::decode_bill_vector < D > ( bill.upper_limit ) ;

    // add the offset to the limits, to obtain the range of
    // coordinates in the reference frame of the tile user, so
    // the coordinates the tile_loader and tile_storer objects
    // receive in their 'init' calls.

    low += offset ;
    high += offset ;

    auto & sgsz ( bill.segment_size ) ; // shorthand

    // now we'll iterate over all tiles and find out their relation
    // to segment boundaries

    zimt::mcs_t < D > mcs ( store_shape ) ;
    for ( std::size_t i = 0 ; i < store_shape.prod() ; i++ )
    {
      // this is the tile we're looking at:

      auto tile_index = mcs() ;
      auto & tether ( store [ tile_index ] ) ; // shorthand

      // find the range of coordinates which this tile covers

      auto tile_lower = tile_index * tile_shape ;
      auto tile_upper = tile_lower + tile_shape ;

      // now we intersect this range with the range of coordinates
      // which the entire run covers

      auto intersect_lower = tile_lower.at_least ( low ) ;
      auto intersect_upper = tile_upper.at_most ( high ) ;

      // test whether there actually is an intersection

      bool filled = true ;

      for ( std::size_t d = 0 ; d < D && filled == true ; d++ )
      {
        if ( intersect_lower [ d ] >= intersect_upper [ d ] )
          filled = false ;
      }

      if ( filled )
      {
        // to calculate the number of intersections with segments,
        // - the 'transit count' - we need to re-base the range to
        // the beginning of the very first *segment*

        intersect_lower -= low ;
        intersect_upper -= low ;

        auto head = intersect_lower [ bill.axis ] / sgsz ;
        head *= sgsz ;

        auto nfragments
          = ( intersect_upper [ bill.axis ] - head ) / sgsz ;

        if ( ( intersect_upper [ bill.axis ] - head ) % sgsz )
          nfragments++ ;

        auto frag_shape = ( intersect_upper - intersect_lower ) ;
        frag_shape [ bill.axis ] = nfragments ;

        // finally we *add* this value to what may already be in
        // the due member - read/write access needs the sum of
        // both the values for read and write access.

        tether.due += frag_shape.prod() ;
      }
    }
  }

protected:

  // moving the data drom the external representation - e.g.
  // a file - and back to it, are coded as pure virtual member
  // functions in this (base) class:

  // load data from a file to a tile's storage array. If the
  // file can't be opened, this is accepted and the data off
  // p_data are left as they are. When this function returns,
  // the tile is deemed operational.

  virtual bool load_tile ( tile_type * p_tile ,
                           const index_type & tile_index ) const = 0 ;

  // Store a tile's storage array to the external representation.
  // When this function returns, it's expected that the external
  // representation is complete and usable.

  virtual bool store_tile ( tile_type * p_tile ,
                            const index_type & tile_index ) const = 0 ;

public:

  // open() is called when the tile store is actually put to
  // use. The call is done by the tile_loader/tile_storer objects.
  // The call serves to set the 'due' member in the tether objects
  // in the store - these values will be counted down as processing
  // proceeds, finally reaching zero for each tile, which is when
  // the tile's contribution to the 'run' ends and it's stored to
  // mass storage. After zimt::process terminates, the 'due'
  // values should all be zero again - currently, we test this
  // in the 'close' function, but this is only a precaution while
  // development isn't finished and can go later.
  // There are three scenarios for the call to 'open': with only
  // one of the flags set, resulting in preparation of the store
  // for either read or write access, and with both flags set,
  // when the store is used in read/write mode. The latter will
  // result in two calls to calculate_due.

  void open ( const zimt::bill_t & bill ,
              bool _read_from_disk ,
              bool _write_to_disk )
  {
    if ( _read_from_disk )
    {
      calculate_due ( bill , true ) ;
      read_from_disk = true ;
    }

    if ( _write_to_disk )
    {
      calculate_due ( bill , false ) ;
      write_to_disk = true ;
    }
  }

  // tile_store_t's c'tor receives the 'notional' shape of the
  // entire workload, the intended - or given - shape of an
  // individual tile, and the base name of files associated
  // with tiles.
  // attention: immediately after construction, the member 'store'
  // is not initialized (it's a zimt::array) - the tether_t objects
  // are only initialized when open() is called.

  tile_store_t ( shape_type _array_shape ,
                 shape_type _tile_shape ,
                 std::string _basename )
  : array_shape ( _array_shape ) ,
    tile_shape ( _tile_shape ) ,
    store_shape ( get_store_shape ( _array_shape , _tile_shape ) ) ,
    store ( get_store_shape ( _array_shape , _tile_shape ) ) ,
    basename ( _basename )
  { }

  // 'get' provides a pointer to the tile at 'tile_index'. If the
  // pointer in the 'tether' is initially nullptr, a new tile_t
  // object is created and the pointer in 'store' is set to it.
  // Memory for the tile is allocated, and, if a file is available,
  // it's content is read into the tile's memory. Acccess is guarded
  // via the mutex in the 'tether', so that only one thread at a
  // time can manipulate the members of the tile object, other
  // threads are blocked until the access is over. Note that access
  // to the tile's data array is allowed concurrently - it's assumed
  // that all threads will only access 'their' share and that the
  // shares don't overlap. In the context of zimt::process this is
  // guaranteed, because each 'joblet index' stands for a distinct,
  // unique part of the total workload.
  // Note also that - since this member function is called by the
  // individual processing threads - the blocking does not block
  // processing altogether: only the thread which 'happens upon'
  // the tile first will spend (considerable) time 'breaking the
  // ground', and only if other threads need access to the same
  // tile at the same time, they are made to wait until the
  // tile is ready. With this granular approach and a suitable
  // number of threads, the CPU cores can still be kept busy
  // even if some threads are busy loading data from files.
  // The design goes for the process to self-organize, rather
  // than controlling every aspect rigidly.

  tile_type * get ( const index_type & tile_index )
  {
    auto & tether ( store [ tile_index ] ) ; // shorthand

#ifndef ZIMT_SINGLETHREAD

    std::lock_guard < std::mutex > lk ( tether.tile_mutex ) ;

#endif // ZIMT_SINGLETHREAD

    // if p_tile is already set, pass it back straight away.
    // The caller is then free to access the data without further
    // access control, because it's assumed that each thread
    // only ever interacts with a specific part of the tile's
    // memory. The only interaction requiring access control
    // is code which manipulates the tile or the pointer to it.

    if ( tether.p_tile == nullptr )
    {
      // the tile isn't yet manifest. create the tile_t object,
      // allocate it's memory and, optionally, read data from
      // a file.

      // tether.p_tile = new tile_type ( tile_shape ) ;
      // tether.p_tile->allocate() ;
      tether.p_tile = allocate_tile() ;

      if ( read_from_disk )
        load_tile ( tether.p_tile , tile_index ) ;
    }

    // the tile is (now) available. Pass back the pointer.

    return tether.p_tile ;
  }

  // 'releasing' a tile counts down the tile's 'due' count
  // and drops the tile when the 'due' count reaches zero,
  // indicating that the tile has contributed all it's due
  // to the current run.

  void release ( const index_type & tile_index ,
                 const long & done )
  {
    {
      auto & tether ( store [ tile_index ] ) ; // shorthand

#ifndef ZIMT_SINGLETHREAD

      std::lock_guard < std::mutex > lk ( tether.tile_mutex ) ;

#endif // ZIMT_SINGLETHREAD

      tether.due -= done ;

      if ( tether.due == 0 )
      {
        // std::cout << "dropping tile " << tile_index << std::endl ;
        drop ( tile_index ) ;
      }
    }
  }

} ;

/// reference implementation of a simple tile store. A tile store
/// implementation must override two pure virtual functions in the
/// base class, namely load_tile and store_tile.

template < typename T , std::size_t N , std::size_t D >
struct basic_tile_store_t
: public zimt::tile_store_t < T , N , D >
{
  typedef zimt::tile_store_t < T , N , D > base_t ;
  using typename base_t::index_type ;
  using typename base_t::tile_type ;
  using typename base_t::value_t ;
  using base_t::tile_shape ;
  using base_t::basename ;
  using base_t::base_t ;

protected:

  // Here we have a set of virtual member functions which can be
  // overridden in derived classes to 'bend' I/O and filename
  // generation to the user's needs. The code in this header
  // doing I/O will access these routines via a base class
  // reference. The member functions in this base class are
  // simple but usable, what's missing is error handling.
  // If something is amiss, they will rather terminate.

  // helper function to construct a filename for a file associated
  // with a specific tile index. For now, this function attaches
  // an identifier and a suffix to the basename; other schemes
  // might produce tree-like naming schemes to avoid overcrowding
  // a single folder when the tile count gets very large. Since
  // this is a virtual function, user code can easily provide
  // their own scheme. Note also that this function is only called
  // by load_tile and store_tile (below) and it's up to potential
  // overrides of these two functions to use it as-is, override it,
  // or ignore it.

  std::string get_filename ( const index_type & tile_index ) const
  {
    auto filename = basename ;
    for ( std::size_t i = 1 ; i <= D ; i++ )
    {
      filename += "_" ;
      filename += std::to_string ( tile_index [ D - i ] ) ;
    }
    filename += std::string ( ".ztl" ) ;
    return filename ;
  }

  // load data from a file to a tile's storage array. If the
  // file can't be opened, this is accepted and the data off
  // p_data are left as they are. file access is coded very
  // simply in C for this class, but classes inheriting from
  // tile_store_t may use their own code. Note that holding
  // tiles as distinct files on mass storage is only one way
  // of providing data - other schemes might read parts of
  // a large file, or blobs from a database etc - the only
  // requirement is to fill the tile_type object's buffer
  // with data corresponding with the given tile index.

  virtual bool load_tile ( tile_type * p_tile ,
                           const index_type & tile_index ) const
  {
    auto filename = get_filename ( tile_index ) ;
    auto & p_data ( p_tile->p_data ) ;

    assert ( p_data != nullptr ) ; // might alocate instead
    FILE* input = fopen ( filename.c_str() , "rb" ) ;

    if ( input != NULL )
    {
      fseek ( input , 0L , SEEK_END ) ;
      auto size = ftell ( input ) ;
      std::size_t nbytes = p_tile->shape.prod() * sizeof ( value_t ) ;
      if ( size >= nbytes )
      {
        fseek ( input , 0L , SEEK_SET ) ;
        fread ( p_data->data() , 1 , nbytes , input ) ;
      }
      fclose ( input ) ;

      ++ load_count ; // for statistics, can go later

      // std::lock_guard < std::mutex > lk ( stdout_mutex ) ;
      // std::cout << std::this_thread::get_id()
      //           << ": read  " << nbytes << " bytes from "
      //           << filename << std::endl ;

      return true ;
    }
    return false ;
  }

  // Store a tile's storage array to a file. The function assumes
  // that the tile holds a single compact block of data, as it is
  // created by the allocation routine.

  virtual bool store_tile ( tile_type * p_tile ,
                            const index_type & tile_index ) const
  {
    auto filename = get_filename ( tile_index ) ;
    auto const & p_data ( p_tile->p_data ) ;

    assert ( p_data != nullptr ) ;
    FILE* output = fopen ( filename.c_str() , "wb" ) ;
    assert ( output != NULL ) ;
    std::size_t nbytes = p_tile->shape.prod() * sizeof ( value_t ) ;
    fwrite ( p_data->data() , 1 , nbytes , output ) ;
    fclose ( output ) ;

    ++ store_count ; // for statistics, can go later

    // std::lock_guard < std::mutex > lk ( stdout_mutex ) ;
    // std::cout << std::this_thread::get_id() <<
    //           ": wrote " << nbytes << " bytes to "
    //           << filename << std::endl ;

    return true ;
  }
} ;

// classes tile_loader and tile_storer share the code to interface
// with the tile store via this common base class. It implements a
// method to limit the amount of tiles which are 'active' - meaning
// that the tile pointer in the tether is not nullptr but points to
// a tile_t object with an attached array of values.
// The method used here to detect and 'close' tiles which are no
// longer needed relies on cooperation with zimt::process and the
// order in which zimt::process accesses parts of the 'notional'
// shape. So this base class is not a general-purpose access
// mediator which could be used for random access - for such uses
// we'd have to work with timestamps, limits on the number of
// open tiles, etc. - but for the dedicated purpose of working
// with zimt::process, this class should perform well with little
// overhead. To figure out when a tile which was loaded previously
// is no longer needed, the tile_store object calculates the amount
// of 'due transits' which will occur during the zimt::process
// run, and these are 'ticked off' as they occur. Once all expected
// 'transits' have occured, the tile can be sent (back) to mass
// storage. The 'ticking off' is done in two stages: first, the
// 'transits' which a thread performs while it's busy in a given
// row of tiles is collected by this thread (in the 'done' vector),
// then, when the next row of tiles is entered, the cumulated 'done'
// values are subtracted from the precalculated 'due' counts in the
// tether (this does require mutex protection) - and here the drop
// of 'due' to zero is detected, resulting in 'dropping' the tile in
// question to mass storage. The two-stage process ensures two things:
// That the count-down only happens when processing has already left
// the row of tiles, and that mutex-protected access to the tether
// is minimized to once per row of tiles rather than once per row
// of data.

template < typename T ,
           std::size_t N ,
           std::size_t D ,
           std::size_t L = zimt::vector_traits < T > :: vsize >
struct tile_user_t
{
  typedef zimt::xel_t < long , D > crd_t ;
  typedef tile_store_t < T , N , D > tile_store_type ;
  typedef typename tile_store_type::tile_type tile_type ;

  const std::size_t d ;
  const long stride ;
  const std::size_t segment_size ;

  tile_store_type & tile_store ;
  tile_type * current_tile ;
  const xel_t < std::size_t , D > tile_shape ;
  crd_t hot_tile ;
  crd_t in_tile_crd ;

  std::vector < tile_type * > working_set ;
  std::vector < long > done ;
  crd_t set_marker ;

private:

  // The tile user maintains a list of tiles it's currently
  // using. Typically, a set of tiles is 'visited' repeatedly,
  // until, finally, the next row of tiles is entered, which
  // can be detected by the tile user. The list of currently
  // used tiles - the 'working set' - holds as many tile pointers
  // as the tile store's shape along the 'hot' axis. Once the
  // new row is entered, the tiles in the store are released
  // with this function:

  void clear_working_set()
  {
    auto index = set_marker ;

    for ( std::size_t i = 0 ;
          i < tile_store.store_shape [ d ] ;
          i++ )
    {
      if ( working_set [ i ] != nullptr )
      {
        index [ d ] = i ;

        // 'release' receives the tile index plus the cumulated
        // 'done' count, which it subtracts from the 'due' value
        // to see if the tile has maybe 'done it's due' already
        // and can be 'dropped' to mass storage.

        tile_store.release ( index , done [ i ] ) ;

        // clear these entries for the next row of tiles:

        working_set [ i ] = nullptr ;
        done [ i ] = 0 ;
      }
    }
  }

public:

  // get_tile will try and provide a pointer to tile_type from
  // a previous cycle if possible, and only if it can't find
  // one to re-use, it accesses the tile store.
  // The function exploits the fact that access to tiles is
  // along a linear subset of tiles along the 'hot' axis, and
  // once the tile_user accesses the next linear subset,
  // there will not be any more accesses to the previous linear
  // subset. So the tile user accumulates tile pointers in
  // it's 'working set' until a new subset is touched, and then
  // the tiles in the working set are released. With this
  // intermediate layer, the need for mutex-protected access
  // to the tiles is greatly reduced: the mutex-protected
  // access only occurs when a slot in the working set is
  // first filled and when the working set is cleared out.
  // In the first case, response will be quick if the tile in
  // question is already used by other threads; then the call
  // to tile_store.get() will return the tile pointer almost
  // immedately. If a tile loader 'happens upon' a tile which
  // is not in use already, it 'takes responsibility' for
  // making the tile available and - while this happens - the
  // tile is locked for other threads which will block if they
  // attempt to gain access, but they will gain access as soon
  // as they can acquire the lock on the tile's mutex.

  tile_type * get_tile ( const zimt::xel_t < long , D > & index )
  {
    // access the working set

    tile_type * p_tile = working_set [ index [ d ] ] ;

    // is there a tile pointer to be had at this position?

    if ( p_tile == nullptr )
    {
      // no luck so far - access the tile store, then save the
      // tile pointer to the working set. Here we have the
      // mutex-protected access (via tile_store.get). It only
      // occurs if the tile is accessed by this tile_user
      // for the first time in this row of tiles, so in total once
      // for every thread cooperating on the current row of tiles,
      // rather than once per 'entering' the tile's domain.

      p_tile = tile_store.get ( index ) ;
      assert ( p_tile != nullptr ) ;
      working_set [ index [ d ] ] = p_tile ;
    }

    // return the tile pointer.

    return p_tile ;
  }

  // tile_user's c'tor sets everything up, including a readily
  // sized std::vector for the 'working set'. Note that this
  // vector is default-initialized to zeros - translating to
  // nullptr in this case - and because the copy c'tor is
  // executed before the working set is populated (this copy
  // happens in zimt::process when per-thread copies of get_t
  // and put_t object are made) we can get away without coding
  // a copy c'tor for tile_user_t. zero-initialization also
  // takes care of correct initial values for 'done'.

  tile_user_t ( tile_store_type & _tile_store ,
                const bill_t & bill )
  : tile_store ( _tile_store ) ,
    d ( bill.axis ) ,
    tile_shape ( _tile_store.tile_shape ) ,
    stride ( tile_type::storage_t::make_strides
      ( _tile_store.tile_shape ) [ bill.axis ] ) ,
    set_marker ( -1 ) ,
    segment_size ( bill.segment_size ) ,
    working_set ( _tile_store.store_shape [ bill.axis ] ) ,
    done ( _tile_store.store_shape [ bill.axis ] )
  { }

  tile_user_t ( const tile_user_t & other ) = default ;

  // get_t/put_t objects aren't to be copy-assigned; we make
  // sure this can't happen:

  tile_user_t & operator= ( const tile_user_t & other ) = delete ;

  // tile_user_t's d'tor releases the tiles in the working set.
  // this needs to be done because after the last row of tiles,
  // no new row is entered to trigger the release of the tiles
  // in the working set.

  ~tile_user_t()
  {
    if ( set_marker != -1 )
    {
      clear_working_set() ;
    }
  }

  // tile_user_t's init function figures out the first tile
  // index for this cycle and calls get_tile. It also figures
  // out the in-tile coordinate. This is the point where
  // zimt::process first interacts with the tile_user object
  // and sets it onto a specific segment by passing the
  // segment's start in 'crd'. This function is called by
  // the tile_loader/tile_storer's init functions to handle
  // store-keeping tasks needed by both.

  void init ( const crd_t & crd )
  {
    hot_tile = crd / tile_shape ;
    auto match_index = hot_tile ;
    match_index [ d ] = 0 ;

    // are we still in the same row of tiles?

    if ( match_index != set_marker )
    {
      // set_marker is only ever -1 right after construction.
      // So if it's *not* -1, it's from a previous row and
      // we release all tiles which are held in working_set

      if ( set_marker != -1 )
      {
        clear_working_set() ;
      }

     // set the set_marker to refer to the new row of tiles

      set_marker = match_index ;
    }

    // call get_tile to obtain the tile pointer to first tile
    // 'touched' by this segment, and calculate the in-tile
    // coordinate

    current_tile = get_tile ( hot_tile ) ;
    in_tile_crd = crd % tile_shape ;
  }

} ; // class tile_user_t

// class tile_loader provides an object which extracts data from
// a tile store, following 'normal' get_t semantics. This does
// require some 'pedestrian' code to deal with situations where
// the tile boundaries and the segment boundaries do not agree.
// If the caller avoids such mismatches, the code runs 'smoothly'
// using efficient vector code throughout, and the caller's aim
// should be to set everything up that way, but we want the code
// to cover all eventualities, hence the 'pedestrian' special
// cases. On the plus side, the implementation is perfectly
// general and can handle arbitrary shapes and boundaries;
// it only takes a little longer when it has to 'cross tile
// boundaries' in a load/process/store cycle.
// A good part of the implementation is shared with tile_storer
// via the common base class, tile_user - namely the logic to
// efficiently gain access to the tiles from the store with a
// minimum of mutex-protected access.

template < typename T ,
           std::size_t N ,
           std::size_t D ,
           std::size_t L = zimt::vector_traits < T > :: vsize >
struct tile_loader
: public tile_user_t < T , N , D , L >
{
  typedef tile_user_t < T , N , D , L > tile_user ;

  using typename tile_user::crd_t ;
  using typename tile_user::tile_store_type ;
  using typename tile_user::tile_type ;

  using tile_user::d ;
  using tile_user::current_tile ;
  using tile_user::tile_shape ;
  using tile_user::hot_tile ;
  using tile_user::in_tile_crd ;
  using tile_user::get_tile ;
  using tile_user::stride ;
  using tile_user::tile_store ;
  using tile_user::done ;

  typedef zimt::xel_t < T , N > value_t ;
  typedef zimt::simdized_type < value_t , L > value_v ;

  const value_t * p_src ;
  std::size_t tail ;

  // the tile_loader's c'tor calls the tile store's open member
  // function to set up the number of 'due transits', which is
  // neded to determine when used tiles can be released from
  // RAM.

  tile_loader ( tile_store_type & _tile_store ,
                const bill_t & bill )
  : tile_user ( _tile_store , bill )
  {
    tile_store.open ( bill , true , false ) ;
  }

  // tile_loader's init function figures out the first tile index
  // for this cycle and obtains the current tile. Then the tile's
  // function 'provide' is called to set 'p_src' and 'tail' to
  // appropriate values, and 'increase' is called to initialize
  // the first batch of vectorized data in 'trg'. Note the
  // incremetation of the 'done' value: every call to 'provide'
  // marks the entry into a new 'transit', which is the metric
  // we use to determine when a used tile is no longer needed.

  void init ( value_v & trg , const crd_t & crd )
  {
    tile_user::init ( crd ) ;
    current_tile->provide ( in_tile_crd , d , p_src , tail ) ;
    done [ hot_tile [ d ] ] ++ ;
    increase ( trg ) ;
  }

  // the 'capped' variant of 'init' only fills in the values below
  // the cap and 'stuffs' the remainder of the lanes with the last
  // 'genuine' value before the cap. The code is the same as above,
  // only the call to 'increase' uses the capped overload with
  // 'stuff' set true to affect the stuffing. Note that this is a
  // rare exception which only occurs if a segment is very short,
  // i.e. because the notional shape is very short along the hot
  // axis. Most of the time, segments are larger and stuffing is
  // not necessary because the previous cycle has already filled
  // the simdized datum with valid content.

  void init ( value_v & trg ,
              const crd_t & crd ,
              std::size_t cap )
  {
    tile_user::init ( crd ) ;
    current_tile->provide ( in_tile_crd , d , p_src , tail ) ;
    done [ hot_tile [ d ] ] ++ ;
    increase ( trg , cap , true ) ;
  }

private:

  // helper function advance enters the next chunk and sets
  // 'p_src' and 'tail' accordingly. Again, the call to
  // 'provide' is followed by the incremetation of 'done'.

  void advance()
  {
    hot_tile [ d ] ++ ;
    in_tile_crd [ d ] = 0 ;
    current_tile = get_tile ( hot_tile ) ;
    current_tile->provide ( in_tile_crd , d , p_src , tail ) ;
    done [ hot_tile [ d ] ] ++ ;
  }

public:

  // 'increase' fetches a full vector of data from the tile store
  // and increments the pointer into the tile's data to the next
  // read position.

  void increase ( value_v & trg )
  {
    if ( tail == 0 )
    {
      // we won't encounter this case initially (init results in
      // non-zero tail) but only if previous processing made 'tail'
      // sink to precisely zero. So we need to call advance.

      advance() ;
    }

    if ( tail >= L )
    {
      // this is the 'normal' case: there are enough values left
      // in the current tile to fill an entire vector. This case
      // is also the most efficient one.

      trg.bunch ( p_src , stride ) ;
      p_src += L * stride ;
      tail -= L ;
    }
    else
    {
      std::size_t lane = 0 ;

      while ( true )
      {
        // how many lanes do we still need to fill?

        std::size_t need = L - lane ;

        // fetch that number, unless tail is smaller: then only
        // fetch as many as tail

        std::size_t fetch = std::min ( need , tail ) ;

        // transfer 'fetch' values to 'trg', counting up 'lane'

        for ( std::size_t left = fetch ;
              left > 0 ;
              --left , ++lane , p_src += stride )
        {
          for ( std::size_t ch = 0 ; ch < N ; ch++ )
          {
            trg [ ch ] [ lane ] = (*p_src) [ ch ] ;
          }
        }

        if ( lane < L )
        {
          // if 'lane' hasn't yet reached it's final value, L, call
          // advance to set 'p_src' and 'tail' to refer to the next
          // tile in line

          advance() ;
        }
        else
        {
          // 'lane' has reached the final value. We subtract 'fetch'
          // from tail because we have consumed that amount of values.
          // then we break the loop

          tail -= fetch ;
          break ;
        }
      }
    }
  }

  // The 'capped' variant only fetches a part of a full vector
  // and doesn't increase the pointer: the capped increase is
  // always the last call in a cycle. Because the cycle ends with
  // this invocation, we can do without updating some member
  // variables: the'll be reset by the subsequent call to
  // 'init' anyway.

  void increase ( value_v & trg ,
                  std::size_t cap ,
                  bool stuff = true )
  {
    if ( tail == 0 )
    {
      advance() ;
    }
    if ( tail >= cap )
    {
      trg.bunch ( p_src , stride , cap , stuff ) ;
    }
    else
    {
      std::size_t nlanes = cap ;
      std::size_t lane = 0 ;

      while ( true )
      {
         // how many lanes do we still need to fill?

        std::size_t need = cap - lane ;

        // fetch that number, unless tail is smaller: then only
        // fetch as many as tail

        std::size_t fetch = std::min ( need , tail ) ;

        // transfer 'fetch' values to 'trg', counting up 'lane'

        for ( std::size_t left = fetch ;
              left > 0 ;
              --left , ++lane , p_src += stride )
        {
          for ( std::size_t ch = 0 ; ch < N ; ch++ )
          {
            trg [ ch ] [ lane ] = (*p_src) [ ch ] ;
          }
        }

        if ( lane < cap )
        {
          // if 'lane' hasn't yet reached it's final value, cap, call
          // advance to set 'p_src' and 'tail' to refer to the next
          // tile in line

          advance() ;
        }
        else
        {
          // 'lane' has reached the final value.

          break ;
        }
      }
    }
  }

} ; // class tile_loader

// tile_storer uses the same techniques as tile_loader, just the
// flow of data is reversed. One might code the processing with
// a flag indicating direction of data flow (i.e. true meaning
// "simdized datum -> memory" and false meaning the reverse), but
// for now I stick with copy-and-paste and swappig source and
// target of the relevant assignments, assuming that this code
// will remain static later on.

template < typename T ,
           std::size_t N ,
           std::size_t D ,
           std::size_t L = zimt::vector_traits < T > :: vsize >
class tile_storer
: public tile_user_t < T , N , D , L >
{
  typedef tile_user_t < T , N , D , L > tile_user ;

  using typename tile_user::crd_t ;
  using typename tile_user::tile_store_type ;
  using typename tile_user::tile_type ;

  using tile_user::d ;
  using tile_user::current_tile ;
  using tile_user::tile_store ;
  using tile_user::tile_shape ;
  using tile_user::hot_tile ;
  using tile_user::in_tile_crd ;
  using tile_user::stride ;
  using tile_user::get_tile ;
  using tile_user::done ;

  typedef zimt::xel_t < T , N > value_t ;
  typedef zimt::simdized_type < value_t , L > value_v ;

  value_t * p_trg ;
  std::size_t tail ;

public:

  tile_storer ( tile_store_type & _tile_store ,
                const bill_t & bill )
  : tile_user ( _tile_store , bill )
  {
    tile_store.open ( bill , false , true ) ;
  }

  void init ( const crd_t & crd )
  {
    tile_user::init ( crd ) ;
    current_tile->provide ( in_tile_crd , d , p_trg , tail ) ;
    done [ hot_tile [ d ] ] ++ ;
  }

private:

  // helper function advance enters the next chunk and sets
  // 'p_trg' and 'tail' accordingly

  void advance()
  {
    hot_tile [ d ] ++ ;
    in_tile_crd [ d ] = 0 ;
    current_tile = get_tile ( hot_tile ) ;
    current_tile->provide ( in_tile_crd , d , p_trg , tail ) ;
    done [ hot_tile [ d ] ] ++ ;
  }

public:

  void save ( value_v & trg )
  {
    if ( tail == 0 )
    {
      advance() ;
    }
    if ( tail >= L )
    {
      // this is the 'normal' case: there are enough values left
      // in the current tile to take an entire vector. This case
      // is also the most efficient one.

      trg.fluff ( p_trg , stride ) ;
      p_trg += L * stride ;
      tail -= L ;
    }
    else
    {
      // The current tile can't accommodate as many values as we
      // have in 'trg' (namely L values, a full vector's lane count)
      // so we have to save as many to the current tile as it can
      // take, then switch to the next tile, until all values are
      // stored.

      std::size_t lane = 0 ;

      while ( true )
      {
        // how many lanes do we still need to store?

        std::size_t pending = L - lane ;

        // store that number, unless tail is smaller: then only
        // store as many as tail

        std::size_t store = std::min ( pending , tail ) ;

        // transfer 'store' values to memory, counting up 'lane'

        for ( std::size_t left = store ;
              left > 0 ;
              --left , ++lane , p_trg += stride )
        {
          for ( std::size_t ch = 0 ; ch < N ; ch++ )
          {
            (*p_trg) [ ch ] = trg [ ch ] [ lane ] ;
          }
        }

        if ( lane < L )
        {
          // if 'lane' hasn't yet reached it's final value, L, call
          // advance to set 'p_trg' and 'tail' to refer to the next
          // tile in line

          advance() ;
        }
        else
        {
          // 'lane' has reached the final value.

          tail -= store ;
          break ;
        }
      }
    }
  }

  // The 'capped' variant only saves a part of a full vector
  // and doesn't increase the pointer: the capped increase is
  // always the last call in a run. Because the run ends with
  // this invocation, we can do without updating some member
  // variables: the'll be reset by the subsequent call to
  // 'init' anyway.

  void save ( value_v & trg ,
              std::size_t cap )
  {
    if ( tail == 0 )
    {
      advance() ;
    }
    if ( tail >= cap )
    {
      trg.fluff ( p_trg , stride , cap ) ;
    }
    else
    {
      // The current tile can't accommodate as many values as we
      // have in 'trg' (namely 'cap' values), so we have to save as
      // many to the current tile as it can take, then switch to the
      // next tile, until all values are stored.

      std::size_t lane = 0 ;

      while ( true )
      {
        // how many lanes do we still need to store?

        std::size_t pending = cap - lane ;

        // store that number, unless tail is smaller: then only
        // store as many as tail

        std::size_t store = std::min ( pending , tail ) ;

        // transfer 'store' values to memory, counting up 'lane'

        for ( std::size_t left = store ;
              left > 0 ;
              --left , ++lane , p_trg += stride )
        {
          for ( std::size_t ch = 0 ; ch < N ; ch++ )
          {
            (*p_trg) [ ch ] = trg [ ch ] [ lane ] ;
          }
        }

        if ( lane < cap )
        {
          // if 'lane' hasn't yet reached it's final value, cap, call
          // advance to set 'p_trg' and 'tail' to refer to the next
          // tile in line

          advance() ;
        }
        else
        {
          // 'lane' has reached the final value. we break the loop
          // without updating 'tail' because the run ends now.

          break ;
        }
      }
    }
  }
} ; // class tile_storer

END_ZIMT_SIMD_NAMESPACE
HWY_AFTER_NAMESPACE() ;

#endif // sentinel
