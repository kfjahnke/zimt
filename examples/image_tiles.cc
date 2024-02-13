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

// This example demonstrates processing an image file with a simple
// functor. It opens an image using OpenImageIO, reads the tiles,
// applies the functor and stores the tiles to a second file. Use
// this example with two image file names, first the input, then the
// output. Use a colour image without an alpha channel as input,
// e.g. a TIFF file. The output should be a file with the same shape,
// but the colour channels rotated.
// This is to demonstrate that we can 'bend' zimt's tile storage
// logic to handle the problem at hand - of course we might simply
// code a program directly reading, modifying and writing scanlines
// without accessing zimt tile code at all.

#include <zimt/zimt.h>
#include <zimt/scanlines.h>
#include <OpenImageIO/imageio.h>
#include <OpenImageIO/typedesc.h>

using namespace OIIO;

namespace zimt
{

// if the input does not contain tiles, we use this class from the
// example 'image_lines.cc' to provide input.

template < typename T , std::size_t N >
struct st_line_store_t
: public line_store_t < T , N >
{
  typedef line_store_t < T , N > base_t ;
  using typename base_t::tile_type ;
  using typename base_t::value_t ;
  using typename base_t::shape_type ;
  using typename base_t::index_type ;
  using typename base_t::line_f ;
  using typename base_t::line_cf ;
  using base_t::base_t ;
  using base_t::tile_shape ;

  // we'll run this example with single-threaded code only, and there
  // will only be one single scan line being processed. So we needn't
  // allocate or deallocate the 'tile' - instead, we use the same one
  // all over again:

  tile_type single_line ;

  // this single tile needs to be initialized, so we have to code
  // a c'tor for this class which passes on the arguments to the base
  // class and does the single-tile initialization:

  st_line_store_t ( std::size_t width ,
                    std::size_t height ,
                    line_f _load_line ,
                    line_cf _store_line )
  : base_t ( width , height , _load_line , _store_line ) ,
    single_line ( { width , 1UL } )
  {
    single_line.p_data = new typename tile_type::storage_t ( tile_shape ) ;
  }

  // good style: we allocate memory, so we also delete it.

  ~st_line_store_t()
  {
    delete single_line.p_data ;
  }

  // instead of actually allocating memory, we use the same memory
  // all over. This is to demonstrate overriding the tile de/allocation.

  virtual tile_type * allocate_tile()
  {
    return & single_line ;
  }

  virtual void deallocate_tile ( tile_type * p_tile )
  { }
} ;

} ;

// further down, we'll construct two st_line_store_t objects.
// st_line_store_t's c'tor expects a tile loading and a tile
// storing function, but since we're only loading with one and
// storing with the other, we have to pass something for the
// other function: this one, pass. It does nothing.

bool pass_line ( const unsigned char * p_trg ,
                 std::size_t nbytes ,
                 std::size_t line )
{
  return true ;
}

bool pass_tile ( const unsigned char * p_trg ,
                 std::size_t nbytes ,
                 std::size_t column ,
                 std::size_t line )
{
  return true ;
}

// type for colour pixels: xel_t of three unsigned char

typedef zimt::xel_t < unsigned char , 3 > px_t ;

// simple zimt pixel functor, rotating the colour channels.
// Note how we use a template for the incoming and outgoing
// data type. This is possible because we don't need SIMD-
// specific code for the 'simdized' version. And since we're
// not doing a reduction, we needn't code a 'capped' variant
// either.

struct rotate_rgb_t
: public zimt::unary_functor < px_t >
{
  template < typename I , typename O >
  void eval ( const I & in , O & out ) const
  {
    out [ 0 ] = in [ 1 ] ;
    out [ 1 ] = in [ 2 ] ;
    out [ 2 ] = in [ 0 ] ;
  }
} ;

int main ( int argc , char * argv[] )
{
  assert ( argc == 3 ) ;

  // first we open the input, and extract width and height of the image.
  // We also assert that there are precisely three channels.

  auto inp = ImageInput::open ( argv[1] ) ;
  assert ( inp != nullptr ) ;
  const ImageSpec & spec = inp->spec() ;
  std::size_t w = spec.width ;
  std::size_t h = spec.height ;
  std::size_t c = spec.nchannels ;
  assert ( c == 3 ) ;
  std::size_t tile_width = spec.tile_width ;
  std::size_t tile_height = spec.tile_height ;
  assert ( tile_width == tile_height ) ;

  // Next we open the output with matching parameters

  auto out = ImageOutput::create ( argv[2] );
  assert ( out != nullptr ) ;
  assert ( out->supports ("tiles") ) ;
  ImageSpec ospec ( w , h , c , TypeDesc::UINT8 ) ;
  ospec.tile_width = 256 ;
  ospec.tile_height = 256 ;
  out->open ( argv[2] , ospec ) ;

  // for the scanline-based read access, we use the same function that
  // we use in image_lines.cc. It will be used if the input does not
  // contain tiles.

  auto load_line = [&] ( unsigned char * p_trg ,
                         std::size_t nbytes ,
                         std::size_t line ) -> bool
  {
    return inp->read_scanline ( line , 0 , TypeDesc::UINT8 , p_trg ) ;
  } ;

  // if the input contains tiles, we use this function for read access:

  auto load_tile = [&] ( unsigned char * p_trg ,
                         std::size_t nbytes ,
                         std::size_t column ,
                         std::size_t line ) -> bool
  {
    return inp->read_tile ( column , line , 0 , TypeDesc::UINT8 , p_trg ) ;
  } ;

  // and this one for write access:

  auto store_tile = [&] ( const unsigned char * p_src ,
                          std::size_t nbytes ,
                          std::size_t column ,
                          std::size_t line ) -> bool
  {
    return out->write_tile ( column , line , 0 , TypeDesc::UINT8 , p_src ) ;
  } ;

  // we set up three tile stores: two as data sources, and one as
  // data drain. We'll only use one of the sources, depending on
  // whether the input file has tiles or not. For the output, we
  // unconditionally write tiles. Note how we can combine a scanline
  // source and a tile target without any problems: the zimt:process
  // logic takes care of extracting and inserting small-ish chunks
  // of data from/to the respective stores, which are completely
  // decoupled otherwise.

  zimt::st_line_store_t < unsigned char , 3 >
    line_source ( w , h , load_line , pass_line ) ;

  zimt::square_store_t < unsigned char , 3 >
    tile_source ( w , h , tile_width ? tile_width : 256 ,
                  load_tile , pass_tile ) ;

  zimt::square_store_t < unsigned char , 3 >
    tile_drain ( w , h , 256 , pass_tile , store_tile ) ;

  // we set up a common 'bill'

  zimt::bill_t bill ;

  // I tried with JPEGs, and OIIO can't handle a multithreaded access
  // without strictly sequential scanline access. Hence, for this example,
  // we also limit the number of threads to one - disk IO will be the
  // limiting factor anyway, so this isn't a problem.

  bill.njobs = 1 ;

  // now we set up get_t and put_t for zimt::process. This is
  // just the same for tile_loader and tile_storer as it is for any
  // of the other get_t/put_t objects, making the tiled storage
  // a 'zimt standard' source/sink. Note how we pass in line_source
  // and line_drain as reference to tile_store_t - their base class.
  // This routes the scanline access to fake_load/fake_store, and
  // there is no need to make the remainder of the tiled storage
  // processing code aware of the 'change of substrate', because the
  // 'tile' access code is coded via virtual member functions.  

  zimt::basic_tile_store_t < unsigned char , 3 , 2 > * p_source ;
  if ( tile_width == 0 )
    p_source = & line_source ;
  else
    p_source = & tile_source ;

  zimt::tile_loader < unsigned char , 3 , 2 > tl ( *p_source , bill ) ;
  zimt::tile_storer < unsigned char , 3 , 2 > tp ( tile_drain , bill ) ;

  // showtime! We use a rotate_rgb_t object as act functor, which
  // affects the RGB rotation, so that we can see in the output that
  // something has indeed happened to the data. The execution time of
  // the act functor is quite negligible compared to the disk I/O,
  // and even quite complex functionality can 'fit into' this
  // scheme of operation without causing much delay, even though
  // the operation isn't multithreaded.
  // Why not load the entire image, process it and store it again?
  // Because this here scheme needs much less memory: two line's
  // worth - one for the input, one for the output. So we can
  // process huge files, and even several of them, without too
  // much memory load and with better cache efficiency.

  zimt::process < 2 > ( { w , h } , tl , rotate_rgb_t() , tp , bill ) ;
}
