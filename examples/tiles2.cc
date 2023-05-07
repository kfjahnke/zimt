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

// set up and fill a tile store holding self-referential discrete
// coordinates. The coordinates are pairs of short, and the resulting
// tile files can be examined easily with od -x

#include <zimt/zimt.h>
#include <zimt/tiles.h>

int main ( int argc , char * argv[] )
{
  // notional shape for zimt::process. Here we use a moderate
  // size, but this shape could become very large in 'real' code.
  // Try and play with this value: like, what happens with 'odd'
  // sizes, small sizes...

  zimt::xel_t < std::size_t , 2 > shape { 2048 , 2048 } ;

  // we set up one tile store as data drain.

  typedef short dtype ;

  zimt::tile_store_t < dtype , 2 , 2 >
    tile_drain ( shape , { 256 , 256 } , "crd_tiles" ) ;

  zimt::bill_t bill ;

  // we feed discrete coordinates as input, don't modify them
  // and store to the tile store.

  zimt::get_crd < dtype , 2 , 2 > gc ( bill ) ;
  zimt::pass_through < dtype , 2 > act ;
  zimt::tile_storer < dtype , 2 , 2 > tp ( tile_drain , bill ) ;

  // showtime!

  zimt::process ( shape , gc , act , tp , bill ) ;

  // no need to tidy up the tile store

  tile_drain.cleanup = false ;
}
