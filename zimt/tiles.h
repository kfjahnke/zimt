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
/// file (provided that it's 'modified' flag was set). Currently
/// the code silently assumes opening the file for writing will
/// succeed. Files for tiles are simply memory copied to disk,
/// there are no metadata nor any provisions to make the data
/// portable.
///
/// Next is a class bundling a set of tiles which, together,
/// represent the storage for an entire (possibly very large)
/// notional shape. This is coded to cooperate with specific
/// get_t and put_t objects to be used with zimt::process; the
/// main intent is currently to get a smooth zimt::process run
/// over a large notional shape to read from, process and store
/// to tile stores with minimal overhead. This is realized
/// with the 'subdivision' feature of zimt::process, which can
/// subdivide a large workload into smaller parcels. These
/// parcels can be chosen so that they fit into the processing
/// system's memory. Once a parcel has been dealt with, the
/// tiles it has accessed during the 'partial' processing can
/// be closed again and the memory is free to be used for the
/// next parcel. zimt::process has the capabilities needed to
/// do this 'subdivided' processing, and it can be triggered
/// by passing appropriate parameters in the 'loading bill'.
///
/// Finally, we have templates of get_t and a put_t objects
/// interacting with the tile store, in order to make tiled
/// storage available as input or output (or both) of
/// zimt::process. For very large workloads, this relies on
/// appropriate 'subdivision' specified in the loading bill
/// and passing of a suitable 'conclude' callback to flush
/// tiles after a partial workload. These additional measures
/// affect processing, but they are compatible with other
/// get_t and put_t objects. With this integration into
/// zimt::process, we gain access to code which, for example,
/// can run a reduction over an entire tile store (or parts
/// of it) without having to load the entire store to memory.
/// We can also process entire tile stores (or parts) with SIMD
/// code like any other data source/sink, using the same act
/// functor as we'd use for non-tiled storage.

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
// memory for storage is allocated only if necessary.

template < typename T , std::size_t N , std::size_t D >
struct tile_t
{
  typedef xel_t < T , N > value_t ;
  typedef array_t < D , value_t > storage_t ;
  typedef xel_t < std::size_t , D > shape_type ;
  typedef xel_t < long , D > index_type ;

  const shape_type shape ;
  storage_t * p_data = nullptr ;
  bool modified = false ;

  // allocate allocates the storage: an array of value_t with
  // shape 'shape'.

  void allocate()
  {
    assert ( p_data == nullptr ) ;
    p_data = new storage_t ( shape ) ;
  }

  // load loads data from a file to the storage array. If the
  // file can't be opened, this is accepted and the data off
  // p_data are left as they are.

  void load ( const char * p_filename )
  {
    assert ( p_data != nullptr ) ;
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

  void store ( const char * p_filename )
  {
    assert ( p_data != nullptr ) ;
    FILE* output = fopen ( p_filename , "wb" ) ;
    assert ( output != NULL ) ;
    std::size_t nbytes = shape.prod() * sizeof ( value_t ) ;
    fwrite ( p_data->data() , 1 , nbytes , output ) ;
    fclose ( output ) ;
  }

  // tile_t's c'tor only sets the shape; allocating storage or
  // access to files is done later with specific functions

  tile_t ( const shape_type & _shape )
  : shape ( _shape ) { }

  // 'provide' provides a pointer into the tile at location 'crd'
  // and writes it to 'p_memory'. 'tail' receives the number of
  // values left along axis d. This function is called by the
  // tile-based get_t/put_t objects to obtain the appropriate
  // data pointer and number of values which the current tile
  // can provide.

  void provide ( const index_type & crd , // in-tile coordinate
                 const std::size_t & d ,  // hot axis
                 value_t * & p_memory ,   // returns pointer
                 std::size_t & tail )     // and number of values
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
// is assumed that several thread which access the same tile
// concurrently will *not access the same data* inside this
// memory. With this access model, user code can simply hold
// the tile pointer and access it without mutex protection.
// This is an important optimization.
// TODO: the tile pointer might be coded as a std::atomic,
// to avoid use of the mutex for most accesses: the calling
// code could get the atomic, if it's not null, use it, and
// if it's null it could get a lock on the mutex, recheck
// and if it's still null allocate the tile.
// Another strategy would be to open all tiles in the current
// (sub-) workload beforehand, then allow mutex-free access.

template < typename tile_t >
struct tether_t
{
#ifndef ZIMT_SINGLETHREAD
  std::mutex tile_mutex ;
#endif

  std::size_t nusers = 0 ;
  tile_t * p_tile = nullptr ;
} ;

// tile_store_t holds an array of tether_t to mediate access
// to the tiles. The amount of tiles needed is calculated from
// the 'notional' shape passed to the c'tor and the intended
// shape of individual tiles.

template < typename T , std::size_t N , std::size_t D >
struct tile_store_t
{
  typedef tile_t < T , N , D > tile_type ;
  typedef xel_t < T , N > value_t ;
  typedef xel_t < std::size_t , D > shape_type ;
  typedef xel_t < long , D > index_type ;

  const shape_type array_shape ; // 'notional' shape
  const shape_type tile_shape ;  // shape of tiles
  const shape_type store_shape ; // shape of the array of tether_t

  // if this flag is set, tiles with modified==true will be
  // written to disk when the tiles are closed.

  bool write_to_disk = false ;

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

  std::string get_file_name ( const index_type & tile_index )
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

public:

  // tile_store_t's c'tor receives the 'notional' shape of the
  // entire workload, the intended shape of an individual tile,
  // and the base name of files associated with tiles.

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
  // object is created and the pointer in store is set to it.
  // Memory for the tile is allocated, and, if p_filename is not
  // nullptr, it's taken as a filename and the file's content
  // is read into the tile's memory. Acccess is guarded via the
  // mutex in the 'tether', so that only one thread at a time
  // can manipulate the members of the tile object, other threads
  // are blocked until the access is over. Note that access to
  // the tile's data array is allowed concurrently - it's
  // assumed that all threads will only access 'their' share
  // and that the shared don't overlap.

  tile_type * get ( const index_type & tile_index )
  {
    auto & tether ( store [ tile_index ] ) ; // shorthand

#ifndef ZIMT_SINGLETHREAD
      std::lock_guard < std::mutex > lk ( tether.tile_mutex ) ;
#endif

    // if p_tile is already set, pass it back straight away.
    // The caller is free to access the data without further
    // access control, because it's assumed that each thread
    // only ever interacts with a specific part of the tile's
    // memory. The only interaction requiring access control
    // is code which manipulates the tile or the pointer to it.

    if ( tether.p_tile == nullptr )
    {
      // the tile isn't yet manifest. create the tile_t object,
      // allocate it's memory and, optionally, read date from
      // a file.

      tether.p_tile = new tile_type ( tile_shape ) ;
      tether.p_tile->allocate() ;

      auto filename = get_file_name ( tile_index ) ;
      tether.p_tile->load ( filename.c_str() ) ;
    }

    ++ tether.nusers ;
    return tether.p_tile ;
  }

private:

  // 'dropping' a tile optionally flushes it's content to
  // a file, then frees the memory for data. this should only
  // be called by mutex-protected code or when it's assured
  // that only one thread is interacting with the tile store.

  void drop ( const index_type & tile_index ,
              tether_type & tether )
  {
    if ( tether.p_tile != nullptr )
    {
      if ( write_to_disk && tether.p_tile->modified )
      {
        auto filename = get_file_name ( tile_index ) ;
        tether.p_tile->store ( filename.c_str() ) ;
      }
      delete tether.p_tile->p_data ;
      delete tether.p_tile ;
      tether.p_tile = nullptr ;
    }
  }

public:

  // closing a tile optionally writes it's data to a file and
  // then deallocates the tile's memory.

  void close ( const index_type & tile_index )
  {
    auto & tether ( store [ tile_index ] ) ; // shorthand

#ifndef ZIMT_SINGLETHREAD
      std::lock_guard < std::mutex > lk ( tether.tile_mutex ) ;
#endif

    // there mustn't be any more users of this tile!

    assert ( tether.nusers == 0 ) ;
    drop ( tile_index , tether ) ;
  }

  // releasing a tile counts down the tile's user count if the
  // tile is manifest.

  void release ( const index_type & tile_index )
  {
    auto & tether ( store [ tile_index ] ) ; // shorthand

#ifndef ZIMT_SINGLETHREAD
      std::lock_guard < std::mutex > lk ( tether.tile_mutex ) ;
#endif

    -- tether.nusers ;
  }

  // close the tile store altogether - i.e. after it was processed.
  // if write_to_disk is set, all tiles marked 'modified' are written
  // to their associated files. Since this is not multithreaded code,
  // we can proceed without the lock_guard.

  void close_all()
  {
    zimt::mcs_t < D > tile_it ( store_shape ) ;
    std::size_t ntiles = store_shape.prod() ;

    for ( std::size_t i = 0 ; i < ntiles ; i++ )
    {
      auto tile_index = tile_it() ;
      auto & tether ( store [ tile_index ] ) ;
      assert ( tether.nusers == 0 ) ;
      drop ( tile_index , tether ) ;
    }
  }

  // close only tiles in a given range of tile indices. This function
  // and the next are used when the workload is very large and
  // subdivided into partial workloads. After each partial workload,
  // zimt::process calls the 'conclude' callback, passing the bill
  // for the partial workload. This bill has the information needed
  // to figure out the range of tile indices which were affected by
  // the partial processing, and hence the range of tiles that have
  // to be checked is narrowed, saving processing time. For large
  // total worlkloads, this may make a considerable difference
  // compared to the 'sloppy' approach of calling close_all after
  // each partial workload.

  void close_some ( const index_type & start ,
                    const index_type & end )
  {
    auto range = end - start ;

    zimt::mcs_t < D > tile_it ( range ) ;
    std::size_t ntiles = range.prod() ;

    for ( std::size_t i = 0 ; i < ntiles ; i++ )
    {
      auto tile_index = tile_it() + start ;
      auto & tether ( store [ tile_index ] ) ;
      assert ( tether.nusers == 0 ) ;
      drop ( tile_index , tether ) ;
    }
  }

  // overload taking the loading bill of a partial workload and a
  // flag indicating whether the get_offset or put_offset is
  // relevant for the operation - this is only known to the caller.

  void close_some ( const bill_t & bill ,
                    bool is_get_t )
  {
    auto lower_limit = decode_bill_vector<D> ( bill.lower_limit ) ;
    auto upper_limit = decode_bill_vector<D> ( bill.upper_limit ) ;
    auto offset = decode_bill_vector<D>
      ( is_get_t ? bill.get_offset : bill.put_offset ) ;
    lower_limit += offset ;
    upper_limit += offset ;
    auto lowest = lower_limit / tile_shape ;
    auto highest = ( upper_limit - 1 ) / tile_shape ;
    close_some ( lowest , highest + 1 ) ;
  }

  // if the tile store is not 'too big', all tiles can be opened
  // initially

  void open_all()
  {
    zimt::mcs_t < D > tile_it ( store_shape ) ;
    std::size_t ntiles = store_shape.prod() ;

    for ( std::size_t i = 0 ; i < ntiles ; i++ )
    {
      auto tile_index = tile_it() ;
      auto & tether ( store [ tile_index ] ) ;
      tether.p_tile = new tile_type ( tile_shape ) ;
      tether.p_tile->allocate() ;
      auto filename = get_file_name ( tile_index ) ;
      tether.p_tile->load ( filename.c_str() ) ;
    }
  }

  ~tile_store_t()
  {
    if ( cleanup )
      close_all() ;
  }
} ;

// class tile_loader provides an object which extracts data from
// a tile store, following 'normal' get_t semantics. This does
// require some 'pedestrian' code to deal with situations where
// the tile boundaries and the segment boundaries do not agree.
// If the caller avoids such mismatches, the code runs 'smoothly'
// using efficient vector code throughout, and the caller's aim
// should be to set everything up that way, but we want the code
// to cover all eventualities, hence the 'pedestrian' special
// cases.

template < typename T ,
           std::size_t N ,
           std::size_t D ,
           std::size_t L = zimt::vector_traits < T > :: vsize >
struct tile_loader
{
  typedef zimt::xel_t < T , N > value_t ;
  typedef zimt::simdized_type < value_t , L > value_v ;
  typedef typename value_v::value_type value_ele_v ;
  typedef zimt::xel_t < long , D > crd_t ;
  typedef zimt::simdized_type < long , L > crd_v ;

  const std::size_t d ;
  typedef tile_store_t < T , N , D > tile_store_type ;
  typedef typename tile_store_type::tile_type tile_type ;
  tile_store_type & tile_store ;
  tile_type * current_tile ;
  value_t * p_src ;
  const std::size_t stride ;
  const xel_t < std::size_t , D > chunk_shape ;
  crd_t hot_chunk ;
  crd_t in_chunk_crd ;
  std::size_t tail ;

  // to use tiled storage effectively, the loading bill should
  // set line_first to false. Then, successive cycles of the
  // loader will use the same set of tiles several times before
  // a new set of tiles is 'entered'. We can exploit this fact
  // and avoid accessing the mutex-protected data in the tile
  // store for such 'repeat perfomances' with a bit of logic:

  struct tile_record
  {
    zimt::xel_t < long , D > index ;
    tile_type * p_tile ;
  } ;

  std::deque < tile_record > repeat_queue ;

  // get_tile will try and provide a pointer to tile_type from
  // the previous cycle if possible, and only if it can't find
  // one to reuse, it accesses the tile store. This function
  // exploits the fact that - if a 'repeat performance' is in fact
  // happening - the tiles are requested in the same sequence as
  // they were requested in the previous cycle, so if the first
  // tile of the 'run' does not match the first tile in the
  // queue, we can safely assume all other tiles in the queue
  // won't match either and flush the queue.

  tile_type * get_tile ( const zimt::xel_t < long , D > & index ,
                         bool flush = false )
  {
    tile_type * p_tile = nullptr ;

    if ( repeat_queue.size() )
    {
      // look at the tip of the repeat queue

      auto const & candidate ( repeat_queue.front() ) ;

      // is this the tile the user has asked for?

      if ( candidate.index == index )
      {
        // yes! we save the tile pointer to p_tile and pop the
        // tip of the queue.

        p_tile = candidate.p_tile ;
        repeat_queue.pop_front() ;
      }
      else if ( flush )
      {
        // the tip of the queue is not the requested one, and
        // 'flush' is set, indicating this is the first call to
        // get_tile in the current cycle. If the first tile is
        // 'wrong', we can forget about the rest as well, so we
        // release all tiles in the queue and clear the queue.

        for ( const auto & it : repeat_queue )
        {
          tile_store.release ( it.index ) ;
        }
        repeat_queue.clear() ;
      }
    }

    // has the search failed?

    if ( p_tile == nullptr )
    {
      // no luck so far - access the tile store

      p_tile = tile_store.get ( index ) ;
      assert ( p_tile != nullptr ) ;
    }

    // by now, we certainly have a tile and push it to the back
    // of the queue. Then we return the tile pointer.

    repeat_queue.push_back ( { index , p_tile } ) ;
    return p_tile ;
  }

  // get_t's c'tor receives the zimt::view providing data and the
  // 'hot' axis. It extracts the strides from the source view.

  tile_loader ( tile_store_type & _tile_store ,
                 const std::size_t & _stride ,
                 const bill_t & bill )
  : tile_store ( _tile_store ) ,
    d ( bill.axis ) ,
    stride ( _stride ) ,
    chunk_shape ( _tile_store.tile_shape )
  { }

  // tile_loader's d'tor releases the tiles in the repeat queue

  ~tile_loader()
  {
    for ( const auto & it : repeat_queue )
    {
      tile_store.release ( it.index ) ;
    }
  }

  // tile_loader's init function figures out the first tile
  // index for this cycle and calls get_tile with 'flush'
  // set true, so that if the first tile in the repeat queue
  // is not the same one, the queue is emptied and the tiles
  // it holds are released. Then the tile's function 'provide'
  // is called to set 'p_src' and 'tail' to correct values,
  // and 'increase' is called to initialize the first batch
  // of vectorized data in 'trg'.

  void init ( value_v & trg , const crd_t & crd )
  {
    hot_chunk = crd / chunk_shape ;
    current_tile = get_tile ( hot_chunk , true ) ;
    in_chunk_crd = crd % chunk_shape ;
    current_tile->provide ( in_chunk_crd , d , p_src , tail ) ;
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
    hot_chunk = crd / chunk_shape ;
    current_tile = get_tile ( hot_chunk , true ) ;
    in_chunk_crd = crd % chunk_shape ;
    current_tile->provide ( in_chunk_crd , d , p_src , tail ) ;
    increase ( trg , cap , true ) ;
  }

  // helper function advance enters the next chunk and sets
  // 'p_src' and 'tail' accordingly

  void advance()
  {
    hot_chunk [ d ] ++ ;
    in_chunk_crd [ d ] = 0 ;
    current_tile = get_tile ( hot_chunk ) ;
    current_tile->provide ( in_chunk_crd , d , p_src , tail ) ;
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
      std::size_t src_index ;
      std::size_t lane = 0 ;

      while ( true )
      {
        // how many lanes do we still need to fill?

        std::size_t need = L - lane ;

        // fetch that number, unless tail is smaller: then only
        // fetch as many as tail

        std::size_t fetch = std::min ( need , tail ) ;

        // transfer 'fetch' values to 'trg', counting up 'lane'

        for ( src_index = 0 ;
              fetch > 0 ;
              --fetch , ++lane , ++src_index )
        {
          for ( std::size_t ch = 0 ; ch < N ; ch++ )
          {
            trg[ch][lane] = p_src [ src_index * stride ] [ ch ] ;
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
      std::size_t src_index ;
      std::size_t lane = 0 ;

      while ( true )
      {
         // how many lanes do we still need to fill?

        std::size_t need = cap - lane ;

        // fetch that number, unless tail is smaller: then only
        // fetch as many as tail

        std::size_t fetch = std::min ( need , tail ) ;

        // transfer 'fetch' values to 'trg', counting up 'lane'

        for ( src_index = 0 ;
              fetch > 0 ;
              --fetch , ++lane , ++src_index )
        {
          for ( std::size_t ch = 0 ; ch < N ; ch++ )
          {
            trg[ch][lane] = p_src [ src_index * stride ] [ ch ] ;
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
          // 'lane' has reached the final value. We subtract 'fetch'
          // from tail because we have consumed that amount of values.
          // then we break the loop

          tail -= fetch ;
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
struct tile_storer
{
  typedef zimt::xel_t < T , N > value_t ;
  typedef zimt::simdized_type < value_t , L > value_v ;
  typedef typename value_v::value_type value_ele_v ;
  typedef zimt::xel_t < long , D > crd_t ;
  typedef zimt::simdized_type < long , L > crd_v ;

  const std::size_t d ;
  typedef tile_store_t < T , N , D > tile_store_type ;
  typedef typename tile_store_type::tile_type tile_type ;
  tile_store_type & tile_store ;
  tile_type * current_tile ;
  value_t * p_trg ;
  const std::size_t stride ;
  const xel_t < std::size_t , D > chunk_shape ;
  crd_t hot_chunk ;
  crd_t in_chunk_crd ;
  std::size_t tail ;

  // to use tiled storage effectively, the loading bill should
  // set line_first to false. Then, successive cycles of the
  // loader will use the same set of tiles several times before
  // a new set of tiles is 'entered'. We can exploit this fact
  // and avoid accessing the mutex-protected data in the tile
  // store for such 'repeat perfomances' with a bit of logic:

  struct tile_record
  {
    zimt::xel_t < long , D > index ;
    tile_type * p_tile ;
  } ;

  std::deque < tile_record > repeat_queue ;

  // get_tile will try and provide a pointer to tile_type from
  // the previous cycle if possible, and only if it can't find
  // one to reuse, it accesses the tile store. This function
  // exploits the fact that - if a 'repeat performance' is in fact
  // happening - the tiles are requested in the same sequence as
  // they were requested in the previous cycle, so if the first
  // tile of the 'run' does not match the first tile in the
  // queue, we can safely assume all other tiles in the queue
  // won't match either and flush the queue.

  tile_type * get_tile ( const zimt::xel_t < long , D > & index ,
                         bool flush = false )
  {
    tile_type * p_tile = nullptr ;

    if ( repeat_queue.size() )
    {
      // look at the tip of the repeat queue

      auto const & candidate ( repeat_queue.front() ) ;

      // is this the tile the user has asked for?

      if ( candidate.index == index )
      {
        // yes! we save the tile pointer to p_tile and pop the
        // tip of the queue.

        p_tile = candidate.p_tile ;
        repeat_queue.pop_front() ;
      }
      else if ( flush )
      {
        // the tip of the queue is not the requested one, and
        // 'flush' is set, indicating this is the first call to
        // get_tile in the current cycle. If the first tile is
        // 'wrong', we can forget about the rest as well, so we
        // release all tiles in the queue and clear the queue.

        for ( const auto & it : repeat_queue )
        {
          tile_store.release ( it.index ) ;
        }
        repeat_queue.clear() ;
      }
    }

    // has the search failed?

    if ( p_tile == nullptr )
    {
      // no luck so far - access the tile store

      p_tile = tile_store.get ( index ) ;
      assert ( p_tile != nullptr ) ;
    }

    // by now, we certainly have a tile and push it to the back
    // of the queue. Then we return the tile pointer.

    repeat_queue.push_back ( { index , p_tile } ) ;
    p_tile->modified = true ;
    return p_tile ;
  }

  // get_t's c'tor receives the zimt::view providing data and the
  // 'hot' axis. It extracts the strides from the source view.

  tile_storer ( tile_store_type & _tile_store ,
                 const std::size_t & _stride ,
                 const bill_t & bill )
  : tile_store ( _tile_store ) ,
    d ( bill.axis ) ,
    stride ( _stride ) ,
    chunk_shape ( _tile_store.tile_shape )
  {
    // for a tile storer, we need to set write_to_disk true

    tile_store.write_to_disk = true ;
  }

  // tile_storer's d'tor releases the tiles in the repeat queue

  ~tile_storer()
  {
    for ( const auto & it : repeat_queue )
    {
      tile_store.release ( it.index ) ;
    }
  }

  // tile_storer's init function figures out the first tile
  // index for this cycle and calls get_tile with 'flush'
  // set true, so that if the first tile in the repeat queue
  // is not the same one, the queue is emptied and the tiles
  // it holds are released. Then the tile's function 'provide'
  // is called to set 'p_trg' and 'tail' to correct values,
  // and 'increase' is called to initialize the first batch
  // of vectorized data in 'trg'.

  void init ( const crd_t & crd )
  {
    hot_chunk = crd / chunk_shape ;
    current_tile = get_tile ( hot_chunk , true ) ;
    in_chunk_crd = crd % chunk_shape ;
    current_tile->provide ( in_chunk_crd , d , p_trg , tail ) ;
  }

  // helper function advance enters the next chunk and sets
  // 'p_trg' and 'tail' accordingly

  void advance()
  {
    hot_chunk [ d ] ++ ;
    in_chunk_crd [ d ] = 0 ;
    current_tile = get_tile ( hot_chunk ) ;
    current_tile->provide ( in_chunk_crd , d , p_trg , tail ) ;
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

      std::size_t trg_index ;
      std::size_t lane = 0 ;

      while ( true )
      {
        // how many lanes do we still need to store?

        std::size_t pending = L - lane ;

        // store that number, unless tail is smaller: then only
        // store as many as tail

        std::size_t store = std::min ( pending , tail ) ;

        // transfer 'store' values to memory, counting up 'lane'

        for ( trg_index = 0 ;
              store > 0 ;
              --store , ++lane , ++trg_index )
        {
          for ( std::size_t ch = 0 ; ch < N ; ch++ )
          {
            p_trg [ trg_index * stride ] [ ch ] = trg[ch][lane] ;
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
          // 'lane' has reached the final value. we break the loop
          // without updating 'tail' because the run ends now.

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

      std::size_t trg_index ;
      std::size_t lane = 0 ;

      while ( true )
      {
        // how many lanes do we still need to store?

        std::size_t pending = cap - lane ;

        // store that number, unless tail is smaller: then only
        // store as many as tail

        std::size_t store = std::min ( pending , tail ) ;

        // transfer 'store' values to memory, counting up 'lane'

        for ( trg_index = 0 ;
              store > 0 ;
              --store , ++lane , ++trg_index )
        {
          for ( std::size_t ch = 0 ; ch < N ; ch++ )
          {
            p_trg [ trg_index * stride ] [ ch ] = trg[ch][lane] ;
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
