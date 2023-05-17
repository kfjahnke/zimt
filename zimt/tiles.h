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
/// context will require careful planning. With zimt:::process,
/// everything should 'snap into place'.

#ifndef ZIMT_TILES_H

#include <stdio.h>
#include <utility>
#include <vector>
#include <deque>

#include "xel.h"
#include "array.h"

std::mutex stdout_mutex ;
std::atomic < long > load_count ( 0 ) ;
std::atomic < long > store_count ( 0 ) ;

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

  // this flag indicates whether the current process has written
  // thie tile to disk. At times, tiles which were already stored
  // only have a partial result and need to be reloaded to add
  // (more of the) remainder. If the 'modified' flag is set, this
  // will be done unconditionally. If it is not set, the flag
  // read_from_disk determines whether tiles are loaded from
  // disk initially or not.

  bool modified = false ;

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
// Note that this is a lightweight object; setting it up will
// allocate and initialize some memory, but there won't be any
// access to mass storage or other lengthy operations. It's
// coded to fit well into an RAII scheme, so that it's set up
// right befor it's used with zimt::process and destructed
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

  const shape_type array_shape ; // 'notional' shape
  const shape_type tile_shape ;  // shape of individual tiles
  const shape_type store_shape ; // shape of the array of tether_t

  // if this flag is set, tiles will be written to disk when they
  // are 'dropped'.

  bool write_to_disk = false ;

  // if this flag is set, data will be read from disk if they
  // are available

  bool read_from_disk = false ;

  // this flag is used to assert that the d'tor can proceed
  // without looking for open tiles and closing them: if
  // processing closes all tiles after use, there's no need
  // to go through the entire set again. The default is to
  // check all tiles, set it to false to avoid the check.

  bool cleanup = true ;

  // array holding tether_type, controlling access to the tiles

  typedef tether_t < tile_type > tether_type ;
  array_t < D , tether_type > store ;

  // base name of the files the store will access

  std::string basename ;

private:

  std::deque < index_type > limbo ;
  std::mutex limbo_mutex ;
  std::size_t limbo_threshold ;

  // helper function to provide store_shape in the c'tor

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
      delete tether.p_tile->p_data ;
      delete tether.p_tile ;
      tether.p_tile = nullptr ;
      tether.nusers = 0 ;
    }
  }

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

  virtual std::string get_filename
            ( const index_type & tile_index ) const
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
  // created by the allocation routine. Note that store-then-reload
  // situations can happen, so if your override writes the data,
  // make sure they are flushed to mass storage so that a subsequent
  // read can reload them. Here, we close the file, which flushes
  // it to mass storage.

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

public:

  // tile_store_t's c'tor receives the 'notional' shape of the
  // entire workload, the intended - or given - shape of an
  // individual tile, and the base name of files associated
  // with tiles. The last parameter gives the size of the
  // 'limbo', in bytes. This is the maximal amount of storage
  // tiles 'in limbo' will occupy, and when there are too many
  // tiles in limbo, so that this limit is exceeded, they are
  // flushed to disk. I haven't yet settled on a good heuristic
  // for this value; I assume it should be in the order of
  // magnitude of two rows of tiles in the store. The code will
  // work even with small limbo, but then there will be many
  // store-then-reload operations and performance will suffer.

  tile_store_t ( shape_type _array_shape ,
                 shape_type _tile_shape ,
                 std::string _basename ,
                 const std::size_t & _limbo_size = 10000000 )
  : array_shape ( _array_shape ) ,
    tile_shape ( _tile_shape ) ,
    store_shape ( get_store_shape ( _array_shape , _tile_shape ) ) ,
    store ( get_store_shape ( _array_shape , _tile_shape ) ) ,
    basename ( _basename )
  {
    // the _limbo_size parameter gives the size in bytes, but
    // internally we use the corresponding number of tiles:

    std::size_t tile_size = tile_shape.prod() * sizeof ( value_t ) ;
    limbo_threshold = 1 + _limbo_size / tile_size ;

    // std::cout << "limbo will hold " << limbo_threshold
    //           << " tiles" << std::endl ;
  }

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
  // Note also that - since this member function is called by the
  // individual processing threads - the blocking does not block
  // processing altogether: only the thread which 'happens upon'
  // the tile first will spend (considerable) time 'breaking the
  // ground', and only if other threads need access to the same
  // tile at the same time, they are made to wait until the
  // tile is ready. With this granular approach and a suitable
  // number of threads, the CPU cores can still be kept busy
  // even if some threads are busy loading data from files.

  // threads can claim a tile right when they run 'init',
  // because then the set of tiles they'll access is already known.
  // The 'claim' only increases the user count, so that the tile
  // won't be dropped, but the time-consuming read-from-disk
  // operation is deferred until the access to the tile actually
  // requires it. Disadvantage: an additional mutex-protected
  // access. To activate this feature, #define PREEMPTIVE_CLAIM,
  // I make this default behaviour for now:

#define PREEMPTIVE_CLAIM

#ifdef PREEMPTIVE_CLAIM

  void claim ( const index_type & tile_index )
  {
    auto & tether ( store [ tile_index ] ) ; // shorthand

#ifndef ZIMT_SINGLETHREAD

      std::lock_guard < std::mutex > lk ( tether.tile_mutex ) ;

#endif // ZIMT_SINGLETHREAD

    ++ tether.nusers ;
  }

#endif // PREEMPTIVE_CLAIM

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
      // a file. If 'read_from_disk' is false, some tiles may
      // still be read from disk, if they were previously written
      // to disk and therefore may only contain a partial result.
      // Such tiles are marked 'modified' in their tether. Such
      // write-then-reload situations are annoying, but they
      // can only be made to disapper by running the code with
      // a single thread or by using a large 'limbo'. How often
      // they occur depends on the notional shape, the tile shape,
      // the number of threads, the scheduling... it's hard to
      // tell beforehand. It may be wise to have an external
      // monitoring process which can modify the limbo size of
      // several processes, so that when many processes compete
      // for resources (i.e. when co-processing many tile stores)
      // limbo sizes are lowered to avoid unduely high RAM need
      // at the cost of suffering more write-then-reload cycles,
      // and if there is less resource demand, rising the limbo
      // size, which will make the individual processes use more
      // RAM but avoid more write-then-reload cycles.
      // write-then-reload cycles may be mitigated by I/O caching.
      // Other strategies to deal with the problem would be to
      // process the store tile by tile (assigning entire tiles
      // to each participating thread), which reduces granularity
      // but may come out on top, especially with large tile stores.
      // Workload subdivision is also an option, running the partial
      // workloads with a single thread. This works well when the
      // divisions are along tile boundaries, if not, the processes
      // may clash on tiles they share (TODO test, I'm not sure
      // whether this is a problem or not). If it can be avoided that
      // threads are halted by the scheduler, this should also make
      // the problem go away - i.e. by running one thread less than
      // the number of CPU cores, which reduces the likelyhood for
      // worker threads being scheduled out. Reducing the number
      // of threads may be less of an issue than it seems: unless
      // the act functor is very CPU-intensive, the process is
      // I/O-bound anyway.

      tether.p_tile = new tile_type ( tile_shape ) ;
      tether.p_tile->allocate() ;

      if ( read_from_disk || tether.modified )
        load_tile ( tether.p_tile , tile_index ) ;
    }

#ifndef PREEMPTIVE_CLAIM

    // if we don't use preemptive claiming (to save one mutex-protected
    // access to the tile's tether) we need to increase the user count
    // here.

    ++tether.nusers ;

#endif // PREEMPTIVE_CLAIM

    // the tile is (now) available. Pass back the pointer.

    return tether.p_tile ;
  }

  // 'releasing' a tile counts down the tile's user count and
  // pushes the tile to the 'limbo' queue if the user count
  // reaches zero. It is not called by tile_loader/tile_storer
  // immediately when processing leaves the scope of a tile, but
  // only when processing enters a new set of tiles; the release
  // comes typically after all processing affecting the tile
  // is over: as zimt::process moves through the notional shape
  // processing lines collinear to the 'hot axis', subsequent
  // lines in the tile are accessed, until processing leaves the
  // tile's scope. Only once all processing threads have left
  // the current row of tiles (as recorded in their 'working set')
  // user count will drop to zero, triggering the pushing of the
  // tile to limbo, from where it will eventually be dropped
  // if the limbo queue's size exceeds the threshold.
  // With this logic, the need for 'overseeing' the process is
  // minimal - it's, in a way, self-organizing, just as the
  // joblet-based code in zimt::process, which drives processing.
  // But if the OS's scheduler halts a thread in mid-operation
  // before it can 'claim' a specific tile, and other threads all
  // 'pass through' this tile, the halted thread may need to
  // reload the tile when it had already passed beyond the limbo
  // and was written to disk.

  void release ( const index_type & tile_index )
  {
    {
      auto & tether ( store [ tile_index ] ) ; // shorthand

#ifndef ZIMT_SINGLETHREAD

        std::lock_guard < std::mutex > lk ( tether.tile_mutex ) ;

#endif // ZIMT_SINGLETHREAD

      // we are strict here - at least as long as the design
      // hasn't 'solidified' entirely: release should only be
      // called 'if there is something to release', and if
      // there are no more users, the tile should have been
      // dropped earlier.

      assert ( tether.nusers > 0 ) ;

      // count down the user count and 'drop' the tile if it
      // reaches zero

      -- tether.nusers ;

      // initially I coded to drop the tile now, but then there
      // are a lot of store-then-reload situations. So now I defer
      // dropping the tile to 'later' when it's more likely that
      // another thread won't need to reload a tile we've already
      // stored.

      if ( tether.nusers == 0 )
      {
        // push the tile to the limbo queue. Note how the entry
        // in the tile's tether remains untouched by this - the
        // entry simply persists, albeit with zero user count.
        // But if the tile is accessed again, user count will
        // rise above zero again. When entries are removed from
        // the limbo queue (when it is too full, or at the end
        // when the tile_store is destructed) and found to have
        // zero users *then*, they are finally dropped. This
        // leaves room for store-then-load-again situations,
        // which can be handled with very little ado: the user
        // count simply rises again, and if the tile is still
        // in use when it's pushed out of the queue, it's
        // simply not dropped, assuming that it will 'come back'
        // again when the current use ends and user count drops
        // to zero again, triggering the tile to be pushed to
        // limbo again *then*.

        std::lock_guard < std::mutex > lk ( limbo_mutex ) ;
        limbo.push_back ( tile_index ) ;
      }
    }

    // if we find now that the limbo queue is too full, we
    // consume so many tile indices from the tip of the queue
    // that the size is below threshold again. Tiles which
    // have no users when popped from the queue are finally
    // dropped, and if they should be needed again they have
    // to be re-loaded from disk - but this should be rare.

    while ( limbo.size() > limbo_threshold )
    {
      index_type tip ;
      {
        std::lock_guard < std::mutex > lk ( limbo_mutex ) ;
        tip = limbo.front() ;
        limbo.pop_front() ;
      }

      auto & tether ( store [ tip ] ) ; // shorthand

#ifndef ZIMT_SINGLETHREAD

      std::lock_guard < std::mutex > lk ( tether.tile_mutex ) ;

#endif // ZIMT_SINGLETHREAD

      // if the tile index from limbo refers to a zero-user
      // tile, we drop that tile. If not, the tile has been
      // taken into use again, in which case it will 'come back'
      // some time later when the last current user releases it,
      // so we needn't actually do anything with it here.

      if ( tether.nusers == 0 )
      {
        // we set a flag to indicate that this tile was written
        // to disk by the current process. Only such tiles should
        // be read again from disk (they contain a partial result)
        // whereas - if the process really generates data rather
        // than modify them - they should not be read even if they
        // are available initially, because that would be futile:
        // the data are simply overwritten.

        drop ( tip ) ;
        tether.modified = true ;
      }
    }
  }

  // close the tile store altogether - i.e. after it was processed.
  // if write_to_disk is set, all tiles marked are written to
  // their associated files. Since this is not multithreaded code,
  // we can proceed without the lock_guard. All tiles still in
  // limbo are dropped now, and the 'read_from_disk' and
  // 'write_to_disk' flags are cleared, in case this object is
  // to be reused.

  void close()
  {
    // zimt::process should not have left any 'active' tiles
    // 'tethered', only the 'limbo' may still hold active tiles.
    // Note that if you use tile_store_t outside the context of
    // zimt::process, this may not be the case and you may need
    // to go through the entire store and drop active frames.

    for ( index_type inx : limbo )
    {
      drop ( inx ) ;
    }
    read_from_disk = false ;
    write_to_disk = false ;
  }

  // calling close drops all tiles still in limbo

  ~tile_store_t()
  {
    close() ;
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
  const std::size_t segment_size ;

  tile_store_type & tile_store ;
  tile_type * current_tile ;
  const xel_t < std::size_t , D > tile_shape ;
  crd_t hot_tile ;
  crd_t in_tile_crd ;

  std::vector < tile_type * > working_set ;
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
        tile_store.release ( index ) ;
        working_set [ i ] = nullptr ;
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
  // subset. So the tile loader accumulates tile pointers in
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
  // The second place where mutex-protected access happens is
  // when user count sinks to zero and the tile is released.
  // Then, the tile 'spends some time in limbo', and when it
  // is released from limbo and still has zero users, it's
  // finally 'dropped' to disk.

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
  // a copy c'tor for tile_user_t.

  tile_user_t ( tile_store_type & _tile_store ,
                const bill_t & bill )
  : tile_store ( _tile_store ) ,
    d ( bill.axis ) ,
    tile_shape ( _tile_store.tile_shape ) ,
    stride ( tile_type::storage_t::make_strides
      ( _tile_store.tile_shape ) [ bill.axis ] ) ,
    set_marker ( -1 ) ,
    segment_size ( bill.segment_size ) ,
    working_set ( _tile_store.store_shape [ bill.axis ] )
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

#ifdef PREEMPTIVE_CLAIM

  // starting at notional coordinate start_crd, claim all tiles
  // which will be needed until the end of the segment.
  // This implements the 'preemptive claiming' mechanism, which
  // increases the user count of all tiles which will be used
  // during the processing of the current segment by one
  // 'preemptively': this would happen later anyway when these
  // tiles are actually accessed, but in the meantime this
  // thread my be scheduled out, other threads may collectively
  // 'pass through' the row of tiles, then detect a zero user
  // count and send the tiles to limbo. When this thread is
  // scheduled to run again, it would have to reactivate the
  // tile from limbo, or, much worse, reload it from disk. If
  // the user count was preemptively raised right at 'init' time,
  // the tile will not reach zero user count and stay available
  // in the tiles store.

  void claim ( const xel_t < long , D > & start_crd ,
               const long segment_size ,
               std::size_t d )
  {
    std::size_t first_pos = start_crd [ d ] ;
    std::size_t last_pos = ( first_pos + segment_size - 1 ) ;

    auto last_crd = start_crd ;
    last_crd [ d ] = last_pos ;

    auto first_index = start_crd / tile_shape ;
    auto last_index = last_crd / tile_shape ;

    while ( true )
    {
      if ( working_set [ first_index [ d ] ] == nullptr )
        tile_store.claim ( first_index ) ;

      if ( first_index [ d ] == last_index [ d ] )
        break ;

      first_index [ d ] ++ ;
    }
  }

#endif // PREEMPTIVE_CLAIM

  // tile_user_t's init function figures out the first tile
  // index for this cycle and calls get_tile. It also figures
  // out the in-tile coordinate

  void init ( const crd_t & crd )
  {
    hot_tile = crd / tile_shape ;
    auto match_index = hot_tile ;
    match_index [ d ] = 0 ;

    // are we still in the same linear subset?

    if ( match_index != set_marker )
    {
      // set_marker is only ever -1 right after construction.
      // So if it's not -1, it's from a previous subset and
      // we release all tiles which are held in working_set

      if ( set_marker != -1 )
      {
        clear_working_set() ;
      }

     // set the set_marker to refer to the new linear subset

      set_marker = match_index ;
    }

#ifdef PREEMPTIVE_CLAIM

    // now we 'claim' all tiles which will be used for this segment.
    // most of the time, the tiles are already in the working set, so
    // this is fast, because then claiming does nothing - the tile has
    // already been claimed earlier. Only if it wasn't claimed before
    // while processing this row of tiles, the user count will be raised
    // by one - preemptively; with the conviction that the actual access
    // to the tile must surely come (all tiles used by this segmemnt must
    // be visited). The preemptive raise of the user count is to avoid
    // write-then-reload situation as best as possible: If the raise of
    // the user count is only done once the tile in question has actually
    // been loaded from disk, much time passes (including I/O) and the
    // thread may be scheduled out by the OS. Meanwhile, other threads
    // can work through an entire row of tiles and finally all let go
    // of 'their' access, and the the user count becoming zero produces
    // a 'spurious drop' because it does not take into account the tiles
    // which the interrupted thread still has 'in the pipeline'. Even with
    // the code as it is, this is possible, but the time window for it to
    // happen is minimized: 'claiming' happens befor the lengthy I/O and
    // only takes a few cycles. A disadvantage of this scheme is the need
    // for an additional mutex-protected access to the new tile in the
    // working set and then another one when it's finally obtained via
    // get_tile. The preemptive claiming should reduce write-then-reload
    // situations so that the limbo size can be reduced.

    claim ( crd , segment_size , d ) ;

#endif

    // finally, call get_tile to obtain the first tile for this segment

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

  typedef zimt::xel_t < T , N > value_t ;
  typedef zimt::simdized_type < value_t , L > value_v ;

  const value_t * p_src ;
  std::size_t tail ;

  tile_loader ( tile_store_type & _tile_store ,
                const bill_t & bill )
  : tile_user ( _tile_store , bill )
  {
    // we set read_from_disk true unconditionally, but user code
    // can change the settings later on. Since we only set the
    // single flag, write_to_disk is unaffected and may or may not
    // be set: if it's also set, the tile store is in read/write
    // mode, which is an expected and accepted modus operandi.
    // With this mode, tiles will be read from disk before they
    // are processed and written back afterwards. If the tile
    // store is only used as data drain, the tiles will be read
    // but not written back.
    // TODO: these flags persist in the tile store after the
    // zimt::process run ends, which may be problematic.

    tile_store.read_from_disk = true ;
  }

  tile_loader ( const tile_loader & other )
  : tile_user ( other )
  {
    tile_store.read_from_disk = true ;
  }

  // tile_loader's init function figures out the first tile index
  // for this cycle obtains the current tile. Then the tile's
  // function 'provide' is called to set 'p_src' and 'tail' to
  // appropriate values, and 'increase' is called to initialize
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
  // only the call to 'increase' uses the capped overload with
  // 'stuff' set true to affect the stuffing. Note that this is a
  // rare exception which only occurs if a segment is very short,
  // i.e. because the notional shape is very short along the hot
  // axis. Most of the time, segments are larger ans stuffing is
  // not necessary because the previous cycle has already filled
  // the simdized datum with valid content.

  void init ( value_v & trg ,
              const crd_t & crd ,
              std::size_t cap )
  {
    tile_user::init ( crd ) ;
    current_tile->provide ( in_tile_crd , d , p_src , tail ) ;
    increase ( trg , cap , true ) ;
  }

private:

  // helper function advance enters the next chunk and sets
  // 'p_src' and 'tail' accordingly

  void advance()
  {
    hot_tile [ d ] ++ ;
    in_tile_crd [ d ] = 0 ;
    current_tile = get_tile ( hot_tile ) ;
    current_tile->provide ( in_tile_crd , d , p_src , tail ) ;
  }

public:

  // 'increase' fetches a full vector of data from the tile store
  // and increments the pointer to the data to the next read position.

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
// flow of data is reversed. One might code the processing with
// a flag indicating direction of data flow (i.e. true meaning
// simsized datum -> memory and false meaning the reverse, but
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
  }

  tile_storer ( const tile_storer & other )
  : tile_user ( other )
  {
    tile_store.write_to_disk = true ;
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
    hot_tile [ d ] ++ ;
    in_tile_crd [ d ] = 0 ;
    current_tile = get_tile ( hot_tile ) ;
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
