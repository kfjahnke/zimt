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
  bill.line_first = false ;
  bill.subdivide = { 4 , 1 } ;

  bill.conclude = [&] ( const zimt::bill_t & bill )
  {
    tile_source.close_some ( bill , true ) ;
    tile_drain.close_some ( bill , false ) ;
  } ;

  // now we set up get_t, act and put_t for zimt::process

  zimt::tile_loader < float , 3 , 2 > tl ( tile_source , 1 , bill ) ;
  zimt::pass_through < float , 3 > act ;
  zimt::tile_storer < float , 3 , 2 > tp ( tile_drain , 1 , bill ) ;

  // showtime!

  zimt::process ( shape , tl , act , tp , bill ) ;

  // we can clear the 'cleanup' flag in the tile stores, all tiles
  // should already be flushed out via the 'conclude' callback

  tile_source.cleanup = false ;
  tile_drain.cleanup = false ;
}
