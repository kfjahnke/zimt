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

// this is the 3D version of tiles2.cc, producing 3D tiles
// containing self-referential 3D coordinates. This is to verify
// that the dimensionality is a usable template argument when
// working with tile stores.

// set up and fill a tile store holding self-referential discrete
// coordinates. The coordinates are triplets of short, and the resulting
// tile files can be examined easily with the likes of od -x or od -d
// This program also allows playing with various parameters, to see
// how they affect performance. Note that if you invoke this program
// for the first time and there are no tile files yet, it may
// terminate very quickly, but the write access to mass storge may
// still go on for a good while (depending on the mass storage) because
// this is often buffered by the OS.

#include <zimt/zimt.h>
#include <zimt/tiles.h>

int main ( int argc , char * argv[] )
{
  // without arguments, use default values.

  long width = 64 ;
  long height = 64 ;
  long depth = 16 ;
  long tile_width = 16 ;
  long tile_height = 16 ;
  long tile_depth = 4 ;
  long segment_size = 32 ;
  long njobs = 8 ;
  long axis = 0 ;

  // I use this program for testing, so I like to have a way to
  // pass in arguments.

  if ( argc > 1 )
    width = std::atol ( argv[1] ) ;

  if ( argc > 2 )
    height = std::atol ( argv[2] ) ;

  if ( argc > 3 )
    depth = std::atol ( argv[3] ) ;

  if ( argc > 4 )
    tile_width = std::atol ( argv[4] ) ;

  if ( argc > 5 )
    tile_height = std::atol ( argv[5] ) ;

  if ( argc > 6 )
    tile_depth = std::atol ( argv[6] ) ;

  if ( argc > 7 )
    segment_size = std::atol ( argv[7] ) ;

  if ( argc > 8 )
    njobs = std::atol ( argv[8] ) ;

  if ( argc > 9 )
    axis = std::atol ( argv[9] ) ;

  // notional shape for zimt::process. By default we use a moderate
  // size, but this shape could become very large in 'real' code.
  // Try and play with this value: like, what happens with 'odd'
  // sizes, small sizes...

  zimt::xel_t < std::size_t , 3 > shape { width , height , depth } ;

  // we set up one tile store as data drain.

  typedef short dtype ;

  {
    zimt::basic_tile_store_t < dtype , 3 , 3 >
      tile_drain ( shape , { tile_width , tile_height , tile_depth } ,
                   "crd3d_tiles" ) ;

    zimt::bill_t bill ;
    bill.segment_size = segment_size ;
    bill.njobs = njobs ;
    bill.axis = axis ;

    // we feed discrete coordinates as input, don't modify them
    // and store to the tile store.

    zimt::get_crd < dtype , 3 , 3 > gc ( bill ) ;
    zimt::pass_through < dtype , 3 > act ;
    zimt::tile_storer < dtype , 3 , 3 > tp ( tile_drain , bill ) ;

    // showtime!

    zimt::process ( shape , gc , act , tp , bill ) ;
  }

  std::cout << zimt::load_count << " load operations" << std::endl ;
  std::cout << zimt::store_count << " store operations" << std::endl ;
}
