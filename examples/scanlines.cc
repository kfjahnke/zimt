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

// simple demo of the use of scanline-based storage with zimt.
// This code is to adapt the tiled storage logic to files which can
// be accessed as individual scanlines. Because zimt::tile_store_t
// can serve as a base class for all kinds of tile access methods
// (it's tile access functions are virtual) and the remainder of the
// tile-based code accesses tiles via the base class, we can simply
// 'slot in' apropriate scanline-reading and scanline-writing functions.
// Here, we simple fake the loads and stores and instead emit text
// indicating what calls the tiled process emits. A real application
// would create a line_store_t with real load/store functions, which
// access scan lines in a file. Note that there is no guarantee that
// the loads/stores will work through the scanlines in sequential
// order, unless you run a single-threaded process. Some scanline-
// based processes may fail to provide non-sequential access.

#include <zimt/zimt.h>
#include <zimt/scanlines.h>

bool fake_load ( unsigned char * p_trg ,
                 std::size_t nbytes ,
                 std::size_t line )
{
  std::cout << "fake load of " << nbytes << " to " << (void*)p_trg
            << " line " << line << std::endl ;
  return true ;
}

bool fake_store ( const unsigned char * p_trg ,
                  std::size_t nbytes ,
                  std::size_t line )
{
  std::cout << "fake store of " << nbytes << " from " << (const void*)p_trg
            << " line " << line << std::endl ;
  return true ;
}

int main ( int argc , char * argv[] )
{
  // notional shape for zimt::process. Here we use a moderate
  // size, but this shape could become very large in 'real' code.

  zimt::xel_t < std::size_t , 2 > shape { 2048 , 16 } ;

  // we set up two tile stores: one as data source, and one as
  // data drain.

  zimt::line_store_t < float , 3 , 2 >
    tile_source ( shape , { 2048 , 1 } , fake_load , fake_store ) ;

  zimt::line_store_t < float , 3 , 2 >
    tile_drain ( shape , { 2048 , 1 } , fake_load , fake_store ) ;

  // we set up a common 'bill'

  zimt::bill_t bill ;
  bill.njobs = 1 ;

  // now we set up get_t, act and put_t for zimt::process. This is
  // just the same for tile_loader and tile_storer as it is for any
  // of the other get_t/put_t objects, making the tiled storage
  // a 'zimt standard' source/sink.

  zimt::tile_loader < float , 3 , 2 > tl ( tile_source , bill ) ;
  zimt::pass_through < float , 3 > act ;
  zimt::tile_storer < float , 3 , 2 > tp ( tile_drain , bill ) ;

  // showtime!

  zimt::process ( shape , tl , act , tp , bill ) ;

  // If there weren't any tiles before, there should be now:
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
}
