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

// simple demo of the use of tiled storage with zimt.

#include <zimt/zimt.h>
#include <zimt/tiles.h>

void first()
{
  // notional shape for zimt::process. Here we use a moderate
  // size, but this shape could become very large in 'real' code.

  zimt::xel_t < std::size_t , 2 > shape { 2048 , 2048 } ;

  // we set up two tile stores: one as data source, and one as
  // data drain. As source, we use the tile store which was made
  // by tiles2.cc, tiles with 'self-referential' coordinates

  zimt::tile_store_t < short , 2 , 2 >
    tile_source ( shape , { 256 , 256 } , "crd_tiles" ) ;

  // Here we test the extraction of a subarray to a new tile
  // store with deliberately 'odd' metrics.

  zimt::tile_store_t < short , 2 , 2 >
    tile_drain ( shape , { 95 , 113 } , "extract" ) ;

  // we set up a common 'bill'

  zimt::bill_t bill ;

  // We set upper and lower limit, and also put_offset, to place
  // the extracted data to the 'upper left' of the new tile store.
  // We expect to see the first tile in the new store to start
  // with coordinates (17, 23), the lower limit of the extracted
  // window, and we expect the 'lower right' tile to have coordinates
  // up to (1104, 1380). The put_offset (which is added to the
  // coordinates the tile_storer will receive as reference) makes
  // the chosen coordinate window appear starting at the new tile
  // store's (0, 0) coordinate.

  bill.lower_limit = { 17 , 23 } ;
  bill.upper_limit = { 1105 , 1381 } ;
  bill.put_offset = { -17 , -23 } ;

  // now we set up get_t, act and put_t for zimt::process. This is
  // just the same for tile_loader and tile_storer as it is for any
  // of the other get_t/put_t objects, making the tiled storage
  // a 'zimt standard' source/sink.

  zimt::tile_loader < short , 2 , 2 > tl ( tile_source , bill ) ;
  zimt::pass_through < short , 2 > act ;
  zimt::tile_storer < short , 2 , 2 > tp ( tile_drain , bill ) ;

  // showtime!

  zimt::process ( shape , tl , act , tp , bill ) ;

  // we can clear the 'cleanup' flag in the tile stores, all tiles
  // should already be flushed.

  tile_source.cleanup = false ;
  tile_drain.cleanup = false ;
}

void second()
{
  // now, as a second test, we re-read the newly made tile store
  // into an 'ordinary' array and check that the content is in fact
  // what we expect.

  zimt::bill_t bill ;

  // the initializers for the 'notional' shape of this run are
  // written as differences to show how they come to be: it's the
  // upper limit from the previous run (in first()) minus the
  // offset, so it's the size of the window we stored to the
  // new tile store: 1088 X 1358

  zimt::xel_t < std::size_t , 2 > shape { 1105 - 17 , 1381 - 23 } ;

  // we have to pass the same 'odd' tile size here as the tile size.
  // The tiles, taken together, actually cover more space
  // (1140 X 1469); so far the tile store is coded to use a uniform
  // tile size, and we have a bit of 'wasted space'. We need to use
  // the same base name as well to get the tiles we want.

  zimt::tile_store_t < short , 2 , 2 >
    tile_source ( shape , { 95 , 113 } , "extract" ) ;

  // this is the 'ordinary' array we'll use as target

  zimt::array_t < 2 , zimt::xel_t < short , 2 > > target ( shape ) ;

  // we set up the three functors for the call to zimt::process

  zimt::tile_loader < short , 2 , 2 > tl ( tile_source , bill ) ;
  zimt::pass_through < short , 2 > act ;
  zimt::storer < short , 2 , 2 > tp ( target , bill ) ;

  // and go!

  zimt::process ( shape , tl , act , tp , bill ) ;

  // for the doublecheck, we set up a coordinate iterator and
  // compare expected and actual value at every coordinate.

  zimt::mcs_t < 2 > mcs ( shape ) ;
  zimt::xel_t < short , 2 > offset { 17 , 23 } ;

  for ( std::size_t i = 0 ; i < shape.prod() ; i++ )
  {
    auto crd = mcs() ;
    assert ( target [ crd ] == crd + offset ) ;
  }

  // again, we don't need the cleanup.

  tile_source.cleanup = false ;
}

void third()
{
  // as a third test, we set up the tile store for read/write
  // operation, meaning that zimt::process will get data from the
  // tiles store and write 'modified' data back.

  zimt::bill_t bill ;

  // we'll just access a small notional shape, and of this shape
  // we'll only access a window.

  zimt::xel_t < std::size_t , 2 > shape { 321 , 250 } ;

  bill.lower_limit = { 19 , 35 } ;
  bill.upper_limit = { 299 , 203 } ;

  // we have to pass the shape, the tile size and the basename
  // to construct the tile store

  zimt::tile_store_t < short , 2 , 2 >
    tile_source ( shape , { 95 , 113 } , "extract" ) ;

  // this is the 'ordinary' array we'll use as target

  zimt::array_t < 2 , zimt::xel_t < short , 2 > > target ( shape ) ;

  // we set up the three functors for the call to zimt::process.
  // because we use tile_source both in a tile_loader and a
  // tile_storer, both read_from_disk and write_to_disk will
  // be set, so the tiles will be read from disk and written
  // back after the modification. We can rely on the fact that
  // no tiles will be written before they're read because of
  // the way zimt::process traverses the data: tiles will be
  // accessed first for reading, and in the same cycle also for
  // writing. The batch of data to fill a vector is extracted
  // from the tile, processed by the act functor and stored
  // back - now to the same location. Both the tile_loader
  // and the tile_storer object will hold a copy to the tile
  // pointer, user count will be at least two. The write access
  // by the tile_storer is always after the read access, so
  // the tile's user count can only drop to zero after a
  // write operation. With zero users, it's safe to store the
  // modified tile to disk. If we have several threads working
  // together, the user count will rise higher, up to twice
  // the number of threads sharing the workload. But the tile
  // is only ever flushed to disk when the user count reaches
  // zero again, which happens only when the last tile_storer
  // copy (each thread has a per-thread copy of the get_t and
  // put_t object) finally releases the tile pointers from it's
  // working set.

  zimt::tile_loader < short , 2 , 2 > tl ( tile_source , bill ) ;
  zimt::amplify_type < zimt::xel_t < short , 2 > > act ( 5 ) ;
  zimt::tile_storer < short , 2 , 2 > tp ( tile_source , bill ) ;

  // go!

  zimt::process ( shape , tl , act , tp , bill ) ;

  tile_source.cleanup = false ;
}

int main ( int argc , char * argv[] )
{
  first() ;
  second() ;
  third() ;
}
