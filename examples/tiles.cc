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

int main ( int argc , char * argv[] )
{
  // notional shape for zimt::process. Here we use a moderate
  // size, but this shape could become very large in 'real' code.

  zimt::xel_t < std::size_t , 2 > shape { 2048 , 2048 } ;

  // we set up two tile stores: one as data source, and one as
  // data drain.

  zimt::tile_store_t < float , 3 , 2 >
    tile_source ( shape , { 256 , 256 } , "tile_source" ) ;

  zimt::tile_store_t < float , 3 , 2 >
    tile_drain ( shape , { 256 , 256 } , "tile_drain" ) ;

  // we could likely fit the data into memory, but we'll do it
  // 'properly' for tiled storage processing: we set line_first
  // false and add a subdivison and appropriate 'conclude'
  // callback to the 'loading bill'.

  zimt::bill_t bill ;

  // setting line_first false will process 'first segments first',
  // meaning the first segments from all lines are processed first,
  // then the second segments of all lines etc. - this is good for
  // tiled processing because the tile_loader/tile_storer can keep
  // their context for a while (they operate over the same set of
  // 'open' tiles repeatedly). The code will work with line_first
  // true, but pobably slightly less efficient (TODO test)

  bill.line_first = false ;

  // the subdivisions should not produce too many subsets, which
  // would slow things down. The current implementation burdens the
  // caller with figuring out what a good subdivision is, it might
  // be (TODO) a good idea to have a heuristic which takes into
  // account tile size and segment size. For tiled processing, it
  // might even pay off to do the multithreading differently, having
  // individual threads process entire tiles. If there are enough
  // tiles 'in play' this may increase performance (TODO test).
  // For the time being, though, I'll stick with the 'normal' mode
  // of processing small linear segments, which is, after all, the
  // mode which all other get_t/put_t objects use, and I want to
  // have a common interface for interoperability. A tile store's
  // content may not fit into memory, but one common scenario is
  // accessing a slice or subarray which can fit, and to move the
  // data out to an array representing the slice or subarray should
  // be possible with a plain loader/storer.

  bill.subdivide = { 4 , 1 } ;

  // The 'conclude' function I use here will interrupt processing
  // and store all tiles which were opened in the current subdivision
  // to disk if they were touched and marked as candidates for
  // storage. This is the simplest way of flushing the data to disk,
  // but other schemes may be more efficient, especially such schemes
  // where processing can continue while the disk I/O takes place.
  // As the code stands, there is a possibility that a subdivision
  // will 'let go' of tiles which are 'picked up again' by the next
  // subdivision. This could be avoided by aligning subdivision
  // boundaries with tile boundaries. If it were assured that tiles
  // aren't picked up again by subsequent subdivisions, we could
  // simply push the I/O to a queue and have it processed by a
  // separate thread. As it stands, flushing to disk is mandatory,
  // because the 'reload' must contain the portion of the data
  // which were stored after processing the previous subdivision.

  bill.conclude = [&] ( const zimt::bill_t & bill )
  {
    tile_source.close_some ( bill , true ) ;
    tile_drain.close_some ( bill , false ) ;
  } ;

  // now we set up get_t, act and put_t for zimt::process. This is
  // just the same for tile_loader and tile_storer as it is for any
  // of the other get_t/put_t objects, making the tiled storage
  // a 'zimt standard' source/sink - only the additional parameters
  // in the bill have to be kept in mind to produce good performance
  // and avoid memory overload - this is not an issue in this small
  // example, but it's the reason I introduce tiled storage at all:
  // to become able to handle data sets larger than physical memory.

  zimt::tile_loader < float , 3 , 2 > tl ( tile_source , 1 , bill ) ;
  zimt::pass_through < float , 3 > act ;
  zimt::tile_storer < float , 3 , 2 > tp ( tile_drain , 1 , bill ) ;

  // showtime!

  zimt::process ( shape , tl , act , tp , bill ) ;

  // If there wetren't any tiles before, there should be now:
  // the tile_storer writes data to disk, producing one file per
  // tile with 'obvious' names. If you want to 'play' with the data,
  // next use 'tile_drain' as tile store for both the loader and
  // the storer the next run and observe how the data are read and
  // rewritten within the same tile store. Next you might set up
  // the source tile store to read the tiles (adapt the basename
  // to 'tile_drain' and set up the target tile store with a new
  // basename (like, 'tile_drain_2'). Now the data should be read
  // from the tiles from the previous run and stored to a new set
  // of tiles.

  // we can clear the 'cleanup' flag in the tile stores, all tiles
  // should already be flushed out via the 'conclude' callback. For
  // this small example, omitting 'conclude' and leaving the
  // 'cleanup' flag true would result in the tiles being stored to
  // disk as well. Even without a 'default' bill, the code will
  // work - the 'special' parameters in the bill are only for
  // performance and to hold RAM consumption low.

  tile_source.cleanup = false ;
  tile_drain.cleanup = false ;
}
