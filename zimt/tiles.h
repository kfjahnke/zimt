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
/// This header codes a notion of 'tiles': small arrays of
/// memory which provide a part of the data corresponding to
/// a part of the 'notional' shape. These tiles are linked to
/// a file which holds the data when the tile is not active.
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
/// can be added later by diversifying the access to mass storage.
///
/// Next is a class bundling a set of tiles which, together,
/// represent the storage for an entire (possibly very large)
/// notional shape. This is coded to cooperate with specific
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
/// functor as we'd use for non-tiled storage.
///
/// Apart from accessing single large data sets, another use
/// case which I intend to exploit is work with several data
/// sets at once. Even if one of these sets may fit into memory,
/// several of them may not, which is where tiled storage comes
/// in. As long as processing is limited to parts of the data
/// in a given time window, such collections of data sets can
/// be processed together, i.e. to form a synopsis or correlation.

#ifndef ZIMT_TILES_H

#include <stdio.h>
#include <utility>
#include <vector>
#include <deque>

#include "xel.h"
#include "array.h"

namespace zimt
{
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

  void allocate()
  {
    assert ( p_data == nullptr ) ;
    p_data = new storage_t ( shape ) ;
  }

  // 'load' loads data from a file to the storage array. If the
  // file can't be opened, this is accepted and the data off
  // p_data are left as they are. file access is coded very
  // simply in C - for now.

  void load ( const char * p_filename ) const
  {
    assert ( p_data != nullptr ) ; // might alocate instead

    FILE* input = fopen ( p_filename , "rb" ) ;
    if ( input != NULL )
    {
      fseek ( input , 0L , SEEK_END ) ;
      auto size = ftell ( input ) ;
      std::size_t nbytes = shape.prod() * sizeof ( value_t ) ;
      if ( size >= nbytes )
      {
        fseek ( input , 0L , SEEK_SET ) ;
        fread ( p_data->data() , 1 , nbytes , input ) ;
      }
      fclose ( input ) ;
    }
  }

  // store stores the storage array to a file

  void store ( const char * p_filename ) const
  {
    assert ( p_data != nullptr ) ;
    FILE* output = fopen ( p_filename , "wb" ) ;
    assert ( output != NULL ) ;
    std::size_t nbytes = shape.prod() * sizeof ( value_t ) ;
    fwrite ( p_data->data() , 1 , nbytes , output ) ;
    fclose ( output ) ;
  }

  // tile_t's c'tor only sets the shape; allocating storage or
  // access to files is done later with specific functions as
  // the need arises. A tile store is - at the tile level -
  // a sparse storage medium: tiles which aren't ever 'touched'
  // by processing have neither memory nor mass storage footprint.
  // So very large notional shapes can be used without necessarily
  // producing any load at all, and only when processing touches
  // a specific tile, it becomes manifest.

  tile_t ( const shape_type & _shape )
  : shape ( _shape ) { }

  // 'provide' obtains a pointer into the tile at location 'crd'
  // and writes it to 'p_memory'. 'tail' receives the number of
  // values left along axis d. This function is called by the
  // tile-based get_t/put_t objects to obtain the appropriate
  // data pointer and number of values which the current tile
  // can still provide until the 'right edge' of the tile is
  // reached.

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
// won't 'step on each other's toes', like in zimt::process.

template < typename tile_t >
struct tether_t
{
  // for single-threaded processing, we can do without the mutex

#ifndef ZIMT_SINGLETHREAD
  std::mutex tile_mutex ;
#endif

  // nusers records the number of threads which hold and may
  // access a copy of the tile pointer, and which may access
  // the tile's array of data concurrently.

  std::size_t nusers = 0 ;

  // this is the tile pointer itself

  tile_t * p_tile = nullptr ;
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

template < typename T , std::size_t N , std::size_t D >
struct tile_store_t
{
  typedef tile_t < T , N , D > tile_type ;
  typedef xel_t < T , N > value_t ;
  typedef xel_t < std::size_t , D > shape_type ;
  typedef xel_t < long , D > index_type ;

  const shape_type array_shape ; // 'notional' shape
  const shape_type tile_shape ;  // shape of individual tiles
  const shape_type store_shape ; // shape of the array of tether_t

  // if this flag is set, tiles will be written to disk when they
  // are 'dropped'.

  bool write_to_disk = false ;

  // if this flag is set, data will be read from disk if they
  // are available

  bool read_from_disk = true ;

  // this flag is used to assert that the d'tor can proceed
  // without looking for open tiles and closing them: if
  // processing closes all tiles after use, there's no need
  // to go through the entire set again. The default is to
  // check all tiles, set it to false to avoid the check.
//
  bool cleanup = true ;

  // array holding tether_type, controlling access to the tiles

  typedef tether_t < tile_type > tether_type ;
  array_t < D , tether_type > store ;

  // base name of the files the store will access

  std::string basename ;

private:

  // helper function to provide store_shape in the c'tor

  static shape_type get_store_shape ( const shape_type & array_shape ,
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

  // helper function to construct a filename for a file associated
  // with a specific tile index

  std::string get_file_name ( const index_type & tile_index ) const
  {
    auto filename = basename ;
    for ( std::size_t i = 0 ; i < D ; i++ )
    {
      filename += "_" ;
      filename += std::to_string ( tile_index[i] ) ;
    }
    filename += std::string ( ".ztl" ) ;
    return filename ;
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
        auto filename = get_file_name ( tile_index ) ;
        tether.p_tile->store ( filename.c_str() ) ;
      }
      delete tether.p_tile->p_data ;
      delete tether.p_tile ;
      tether.p_tile = nullptr ;
    }
  }

  // close the tile store altogether - i.e. after it was processed.
  // if write_to_disk is set, all tiles marked are written to
  // their associated files. Since this is not multithreaded code,
  // we can proceed without the lock_guard. This is more of a
  // precaution - if all goes according to plan, all tiles should
  // have been released already before this call happens. This is
  // also a private member function.

  void close_all()
  {
    zimt::mcs_t < D > tile_it ( store_shape ) ;
    std::size_t ntiles = store_shape.prod() ;

    for ( std::size_t i = 0 ; i < ntiles ; i++ )
    {
      auto tile_index = tile_it() ;
      auto & tether ( store [ tile_index ] ) ;
      assert ( tether.nusers == 0 ) ;
      drop ( tile_index ) ;
    }
  }

public:

  // tile_store_t's c'tor receives the 'notional' shape of the
  // entire workload, the intended - or given - shape of an
  // individual tile, and the base name of files associated
  // with tiles.

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
  // shares don't overlap. This is - in the context of zimt::process -
  // guaranteed, because each 'joblet index' stands for a distinct,
  // unique part of the total workload.
  // Note that - since this member function is called by the
  // individual processing threads - the blocking does not block
  // processing altogether: only the thread which 'happens upon'
  // the tile first will spend (considerable) time 'breaking the
  // ground', and only if other threads need access to the same
  // same tile at the same time, they are made to wait until the
  // tile is ready. With this granular approach and a suitable
  // number of threads, the CPU cores can still be kept busy
  // even if some threads are busy loading data from files.

  tile_type * get ( const index_type & tile_index )
  {
    auto & tether ( store [ tile_index ] ) ; // shorthand

#ifndef ZIMT_SINGLETHREAD
      std::lock_guard < std::mutex > lk ( tether.tile_mutex ) ;
#endif

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

      tether.p_tile = new tile_type ( tile_shape ) ;
      tether.p_tile->allocate() ;
      auto filename = get_file_name ( tile_index ) ;
      tether.p_tile->load ( filename.c_str() ) ;
      tether.nusers = 0 ;
    }

    // the tile is (now) available. Increase the user count
    // and pass back the pointer.

    ++ tether.nusers ;

    return tether.p_tile ;
  }

  // releasing a tile counts down the tile's user count and
  // 'drops' the tile if the user count reaches zero. release
  // is not called by tile_loader/tile_storer immediately when
  // processing leaves the scope of a tile, but only when
  // processing enters a new set of tiles, so the release
  // comes typically after all processing affecting the tile
  // is over: as zimt::process moves through the notional shape
  // processing lines collinear to the 'hot axis', subsequent
  // lines in the tile are accessed, until processing leaves the
  // tile's scope. Only once all processing threads have left
  // the current row of tiles (as recorded in their 'working set')
  // user count will drop to zero, triggering the call to 'drop'.
  // With this logic, the need for 'overseeing' the process is
  // minimal - it's, in a way, self-organizing, just as the
  // joblet-based code in zimt::process, which drives processing.

  void release ( const index_type & tile_index )
  {
    auto & tether ( store [ tile_index ] ) ; // shorthand

#ifndef ZIMT_SINGLETHREAD
      std::lock_guard < std::mutex > lk ( tether.tile_mutex ) ;
#endif

    // we are strict here - at least as long as the design
    // hasn't 'solidified' entirely: release should only be
    // called 'if there is something to release', and if
    // there are no more users, the tile should have been
    // dropped earlier.

    assert ( tether.nusers > 0 ) ;

    // count down the user count and 'drop' the tile if it
    // reaches zero

    -- tether.nusers ;

    if ( tether.nusers == 0 )
    {
      drop ( tile_index ) ;
    }
  }

  // calling close_all should not be necessary - if 'cleanup'
  // is false, it will be omitted, assuming all tiles were
  // closed earlier.

  ~tile_store_t()
  {
    if ( cleanup )
      close_all() ;
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
// overhead.

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

  tile_store_type & tile_store ;
  tile_type * current_tile ;
  const xel_t < std::size_t , D > chunk_shape ;
  crd_t hot_chunk ;
  crd_t in_tile_crd ;

  tile_type ** working_set ;
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
    for ( std::size_t i = 0 ;
          i < tile_store.store_shape [ d ] ;
          i++ )
    {
      if ( working_set [ i ] != nullptr )
      {
        set_marker [ d ] = i ;
        tile_store.release ( set_marker ) ;
        working_set [ i ] = nullptr ;
      }
    }
  }

public:

  // get_tile will try and provide a pointer to tile_type from
  // a previous cycle if possible, and only if it can't find
  // one to reuse, it accesses the tile store.
  // The function exploits the fact that access to tiles is
  // along a linear subset of tiles along the 'hot' axis, and
  // once the tile loader accesses the next linear subset,
  // there will not be any more accesses to the previous linear
  // subset. So the tile loader accumulates tile pointers in
  // it's 'working set' until a new subset is touched, and then
  // the tiles in the working set are released. With this
  // inermediate layer, the need for mutex-protected access
  // to the tiles is greatly reduced.

  tile_type * get_tile ( const zimt::xel_t < long , D > & index )
  {
    auto match_index = index ;
    match_index [ d ] = 0 ;

    // set_marker is only ever -1 right after construction.

    if ( set_marker == -1 )
    {
      set_marker = match_index ;
    }

    // are we still in the same linear subset?

    else if ( match_index != set_marker )
    {
      // no: this is a new linear subset. release all tiles
      // which are held in working_set

      clear_working_set() ;

      // set the set_marker to refer to the new linear subset

      set_marker = match_index ;
    }

    // now try and access the working set

    tile_type * p_tile = working_set [ index [ d ] ] ;

    // is there a pointer to be had at this position?

    if ( p_tile == nullptr )
    {
      // no luck so far - access the tile store, then save the
      // tile pointer to the working set. Here we have the
      // mutex-protected access (via tile_store.get). It only
      // occurs if the tile is accessed by this tile loader
      // for the very first time, so in total once for every
      // thread cooperating on the current row of tiles,
      // rather than once per 'entering' the tile's domain.

      p_tile = tile_store.get ( index ) ;
      assert ( p_tile != nullptr ) ;

      working_set [ index [ d ] ] = p_tile ;
    }

    // return the tile pointer.

    return p_tile ;
  }

  tile_user_t ( tile_store_type & _tile_store ,
                const bill_t & bill )
  : tile_store ( _tile_store ) ,
    d ( bill.axis ) ,
    chunk_shape ( _tile_store.tile_shape ) ,
    stride ( tile_type::storage_t::make_strides
      ( _tile_store.tile_shape ) [ d ] ) ,
    set_marker ( -1 )
  {
    working_set = new tile_type * [ tile_store.store_shape [ d ] ] ;
    std::memset ( working_set ,
                  0 ,
                    tile_store.store_shape [ d ]
                  * sizeof ( tile_type* ) ) ;
  }

  // because we have the working set in dynamic memory, we need
  // a copy c'tor which sets up a new working set for the copy.

  tile_user_t ( const tile_user_t & other )
  : tile_store ( other.tile_store ) ,
    d ( other.d ) ,
    chunk_shape ( other.chunk_shape ) ,
    stride ( other.stride ) ,
    set_marker ( -1 )
  {
    working_set = new tile_type * [ tile_store.store_shape [ d ] ] ;
    std::memset ( working_set ,
                  0 ,
                    tile_store.store_shape [ d ]
                  * sizeof ( tile_type* ) ) ;
   }

  // get_t/put_t objects aren't to be copy-assigned; we make
  // sure this can't happen:

  tile_user_t & operator= ( const tile_user_t & other ) = delete ;

  // tile_user_t's d'tor releases the tiles in the working set.
  // this needs to be done because after the last row of tiles,
  // no new row is entered to trigger the release of the tiles
  // in the working set.

  ~tile_user_t()
  {
    clear_working_set() ;
    delete[] working_set ;
  }

  // tile_user_t's init function figures out the first tile
  // index for this cycle and calls get_tile. It also figures
  // out the in-tile coordinate

  void init ( const crd_t & crd )
  {
    hot_chunk = crd / chunk_shape ;
    current_tile = get_tile ( hot_chunk ) ;
    in_tile_crd = crd % chunk_shape ;
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
// general and can handle arbitrary shapes and boundaries,
// it only takes a little longer when it has to 'cross tile
// boundaries' in a load/process/store cycle.

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
  using tile_user::chunk_shape ;
  using tile_user::hot_chunk ;
  using tile_user::in_tile_crd ;
  using tile_user::get_tile ;
  using tile_user::stride ;
  using tile_user::tile_store ;

  typedef zimt::xel_t < T , N > value_t ;
  typedef zimt::simdized_type < value_t , L > value_v ;

  const value_t * p_src ;
  std::size_t tail ;

  // TODO: think about tile stores which are used at the same
  // time for reading and writing data: they should have both
  // flags true.

  tile_loader ( tile_store_type & _tile_store ,
                const bill_t & bill )
  : tile_user ( _tile_store , bill )
  {
    tile_store.write_to_disk = false ;
    tile_store.read_from_disk = true ;
  }

  tile_loader ( const tile_loader & other )
  : tile_user ( other )
  {
    tile_store.write_to_disk = false ;
    tile_store.read_from_disk = true ;
  }

  // tile_loader's init function figures out the first tile
  // index for this cycle and calls get_tile. Then the tile's
  // function 'provide' is called to set 'p_src' and 'tail'
  // to correct values, and 'increase' is called to initialize
  // the first batch of vectorized data in 'trg'.

  void init ( value_v & trg , const crd_t & crd )
  {
    tile_user::init ( crd ) ;
    current_tile->provide ( in_tile_crd , d , p_src , tail ) ;
    increase ( trg ) ;
  }

  // the 'capped' variant of 'init' only fills in the values below
  // the cap and 'stuffs' the remainder of the lanes with the last
  // 'genuine' value before the cap. The code is the same as above,
  // only the call to 'increase' uses the capped overload.

  void init ( value_v & trg ,
              const crd_t & crd ,
              std::size_t cap )
  {
    tile_user::init ( crd ) ;
    current_tile->provide ( in_tile_crd , d , p_src , tail ) ;
    increase ( trg , cap , true ) ;
  }

  // helper function advance enters the next chunk and sets
  // 'p_src' and 'tail' accordingly

  void advance()
  {
    hot_chunk [ d ] ++ ;
    in_tile_crd [ d ] = 0 ;
    current_tile = get_tile ( hot_chunk ) ;
    current_tile->provide ( in_tile_crd , d , p_src , tail ) ;
  }

  // 'increase' fetches a full vector of data from the source view
  // and increments the pointer to the data in the view to the next
  // position.

  void increase ( value_v & trg )
  {
    if ( tail == 0 )
    {
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
  // always the last call in a run. Because the run ends with
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
// flow of data is reversed.

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
  using tile_user::chunk_shape ;
  using tile_user::hot_chunk ;
  using tile_user::in_tile_crd ;
  using tile_user::stride ;
  using tile_user::get_tile ;

  typedef zimt::xel_t < T , N > value_t ;
  typedef zimt::simdized_type < value_t , L > value_v ;

  value_t * p_trg ;
  std::size_t tail ;

public:

  tile_storer ( tile_store_type & _tile_store ,
                const bill_t & bill )
  : tile_user ( _tile_store , bill )
  {
    tile_store.write_to_disk = true ;
    tile_store.read_from_disk = false ;
  }

  tile_storer ( const tile_storer & other )
  : tile_user ( other )
  {
    tile_store.write_to_disk = true ;
    tile_store.read_from_disk = false ;
  }

  // tile_storer's init function figures out the first tile
  // index for this cycle and calls get_tile. Then the tile's
  // function 'provide' is called to set 'p_trg' and 'tail'
  // to correct values, and 'increase' is called to initialize
  // the first batch of vectorized data in 'trg'.

  void init ( const crd_t & crd )
  {
    tile_user::init ( crd ) ;
    current_tile->provide ( in_tile_crd , d , p_trg , tail ) ;
  }

  // helper function advance enters the next chunk and sets
  // 'p_trg' and 'tail' accordingly

  void advance()
  {
    hot_chunk [ d ] ++ ;
    in_tile_crd [ d ] = 0 ;
    current_tile = get_tile ( hot_chunk ) ;
    current_tile->provide ( in_tile_crd , d , p_trg , tail ) ;
  }

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


} ; // namespace zimt

#define ZIMT_TILES_H
#endif
