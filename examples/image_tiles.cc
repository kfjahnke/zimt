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

// TODO: there is a problem with a failing asertion in tiles.h
// see line 631ff. - this needs investigation. I removed the assertion
// for now, but the result is not correct: one channel is missing in
// the last columns.

// This example demonstrates processing a tiled image file with a
// functor. It opens an image using OpenImageIO, reads the tiles,
// applies the functor and stores the tiles to a second file. Use
// this example with two image file names, first the input, then the
// output. Use a colour image without an alpha channel as input,
// e.g. a TIFF file. The output should be a file with the same shape,
// but the colour channels rotated.
// This is to demonstrate that we can 'bend' zimt's tile storage
// logic to handle the problem at hand - of course we might simply
// code a program directly reading, modifying and writing tiles
// without accessing zimt tile code at all.
// If the input is not tiled, that's okay, this program will read
// scanline-based files as well. The output format must support tiles,
// though, and there aren't many which do - I found TIFF, exr and webp
// to work, the latter two being slow to write on my system, likely due
// to the use of compression and the likes. TIFF via OIIO accepts the
// compression = none specification, which I pass here, and that makes
// writing tiled TIFFs quite fast - close to writing individual tiles
// with a basic_tile_store_t object used for data storage, which is
// still a bit faster on my machine.
// Reading from scanline-based files and writing to tile-based ones
// shows the versatility of handling the process with zimt: The 'point
// of contact' is just a SIMD vector's worth of data, and picking them
// up from a scanline on the source side and depositing them in a tile
// on the target side is handled transparently by the zimt::transform
// logic, which fetches and stores data as needed.

#include <zimt/zimt.h>
#include <zimt/scanlines.h>
#include <OpenImageIO/imageio.h>
#include <OpenImageIO/typedesc.h>

using namespace OIIO;

// type used for internal storage of data and corresponding OIIO typedesc
// we can use unsigned char and UINT8 here for all SIMD backends except
// std::simd, which does not support SIMD types based on unsigned char.

#ifdef USE_STDSIMD

typedef unsigned short ele_type ;
auto typedesc = TypeDesc::UINT16 ;

#else

typedef unsigned char ele_type ;
auto typedesc = TypeDesc::UINT8 ;

#endif

// further down, we'll construct a line_store_t object.
// line_store_t's c'tor expects a tile loading and a tile
// storing function, but since we're only loading with one and
// storing with the other, we have to pass something for the
// other function: this one, pass. It does nothing.

bool pass_line ( const ele_type * p_trg ,
                 std::size_t nbytes ,
                 std::size_t line )
{
  return true ;
}

// for tile-based access, there's this 'pass' variant:

bool pass_tile ( const ele_type * p_trg ,
                 std::size_t nbytes ,
                 std::size_t column ,
                 std::size_t line )
{
  return true ;
}

// type for colour pixels: xel_t of three unsigned char

typedef zimt::xel_t < ele_type , 3 > px_t ;

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
    out = in ;
    // out [ 0 ] = in [ 1 ] ;
    // out [ 1 ] = in [ 2 ] ;
    // out [ 2 ] = in [ 0 ] ;
  }
} ;

void dump_tile ( std::string base_name , long x , long y ,
                 const unsigned char *pixels )
{
  base_name += std::to_string ( x ) + "_" + std::to_string ( y ) + ".jpg" ;
  const char* filename = base_name.c_str() ;
  const int xres = 256, yres = 256, channels = 3;

  std::unique_ptr<ImageOutput> out = ImageOutput::create(filename);
  if (!out)
      return;  // error
  ImageSpec spec(xres, yres, channels, TypeDesc::UINT8);
  out->open(filename, spec);
  out->write_image(TypeDesc::UINT8, pixels);
  out->close();
}

// main takes two arguments: input and output filename.

std::mutex cout_mutex ;

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

  std::cout << "input spec's tile_width: " << tile_width << std::endl ;

  // Next we open the output with matching parameters. We assert that
  // the output format can accept tiles.

  auto out = ImageOutput::create ( argv[2] );
  assert ( out != nullptr ) ;
  assert ( out->supports ("tiles") ) ;

  // The output may accept tiles in random order, which we'll exploit
  // by using multithreaded code and traversing the tiles column-first.
  // Note, though, that as of this writing, this only worked for webp
  // output, and after the zimt::process was done, the program took a
  // good while to terminate after the call to close the target was
  // issued, likely doing all the compression and writing to disk
  // then and not during the zimt_process run.

  bool random_access = out->supports ("random_access") ;

  // note how the Typedesc::UINT8 here prescribes the data type used
  // in the output file, whereas 'typedesc' defined further up refers
  // to the type used for internal data storage. OIIO translates the
  // data automatically.

  ImageSpec ospec ( w , h , c , TypeDesc::UINT8 ) ;

  // we want maximum speed here, so no compression. This will only affect
  // some formats, but for TIFF, it works. Some tiled formats can be very
  // slow - you'll see that the input was closed, but it may take a while
  // until the output also closes and the program terminates. This time
  // is taken to encode the data to the target format - I think they are
  // held in 'raw' form until then. AFAICT, working with uncompressed
  // TIFF tiles, the data are written 'straight through' and even large
  // datasets are processed quickly (provided disk I/O is fast). But
  // it can't multithread, because it doesn't support random access,
  // so the tiles have to be written in sequence.

  ospec [ "compression" ] = "none" ;
  ospec.tile_width = 256 ;
  ospec.tile_height = 256 ;
  out->open ( argv[2] , ospec ) ;

  // for the scanline-based read access, we use the same function that
  // we use in image_lines.cc. It will be used if the input does not
  // contain tiles. We refrain from echoing the line numbers to avoid
  // too much clutter, if you want to see them, follow the pattern
  // used in the tile-accessing functions below.

  auto load_line = [&] ( ele_type * p_trg ,
                         std::size_t nbytes ,
                         std::size_t line ) -> bool
  {
    return inp->read_scanline ( line , 0 , typedesc , p_trg ) ;
  } ;

  // if the input contains tiles, we use this function for read access.
  // note that the process may multithread, so we have to lock_guard
  // the echo to avoid garbled output. To save the individual tiles
  // to image files for inspection, uncomment the line with dump_tile.

  auto load_tile = [&] ( ele_type * p_trg ,
                         std::size_t nbytes ,
                         std::size_t column ,
                         std::size_t line ) -> bool
  {
    auto success = inp->read_tile ( column , line , 0 , typedesc , p_trg ) ;
    // dump_tile ( "input" , column , line , p_trg ) ;
    std::lock_guard < std::mutex > lk ( cout_mutex ) ;
    std::cout << "read_tile: x " << column << " y " << line ;
    if ( ! success )
      std::cout << " failed" ;
    std::cout << std::endl ;
    return success ;
  } ;

  // the same for write access:

  auto store_tile = [&] ( const ele_type * p_src ,
                          std::size_t nbytes ,
                          std::size_t column ,
                          std::size_t line ) -> bool
  {
    bool success = out->write_tile ( column , line , 0 , typedesc , p_src ) ;
    // dump_tile ( "output" , column , line , p_src ) ;
    std::lock_guard < std::mutex > lk ( cout_mutex ) ;
    std::cout << "write_tile: x " << column << " y " << line ;
    if ( ! success )
      std::cout << " failed" ;
    std::cout << std::endl ;
    return success ;
  } ;

  // we set up three tile stores: two as data sources, and one as
  // data drain. We'll only use one of the sources, depending on
  // whether the input file has tiles or not. For the output, we
  // unconditionally write tiles. Note how we can combine a scanline
  // source and a tile target without any problems: the zimt:process
  // logic takes care of extracting and inserting small-ish chunks
  // of data from/to the respective stores, which are completely
  // decoupled otherwise.
  // Note that if we 'go columns-first' (see further down) for
  // targets which support random access, and if the input is
  // scanline-based, this will result in all scanlines being held
  // in memory, because they can only be released after all
  // columns of tiles have been processed. This isn't code for
  // efficiency, but to check that the tile-processing code does
  // indeed work.

  zimt::line_store_t < ele_type , 3 >
    line_source ( w , h , load_line , pass_line ) ;

  zimt::square_store_t < ele_type , 3 >
    tile_source ( w , h , tile_width ? tile_width : 256 ,
                  load_tile , pass_tile ) ;

  zimt::square_store_t < ele_type , 3 >
    tile_drain ( w , h , 256 , pass_tile , store_tile ) ;

  // we set up a common 'bill'

  zimt::bill_t bill ;

  if ( random_access )
  {
    // more to show that we can than for any other reason - accessing
    // the data with aggregation along axis 1 is less efficient, but
    // needed for some purposes.
  
    std::cout << "target supports random access, will go columns-first"
              << std::endl ;
  
    bill.axis = 1 ;
  }
  else
  {
    // multithreaded access is only safe if the target can accept
    // tiles in random order. So if it can't, we limit the process
    // to a single thread.

    std::cout << "target doesn't supports random access, will single-thread"
              << std::endl ;
  
    bill.njobs = 1 ;
  }

  // now we set up get_t and put_t for zimt::process. This is
  // just the same for tile_loader and tile_storer as it is for any
  // of the other get_t/put_t objects, making the tiled storage
  // a 'zimt standard' source/sink. Note how we pass in line_source
  // or tile_source as references to tile_store_t - their base class.
  // This works, since the functions which differ are all virtual.

  zimt::tile_store_t < ele_type , 3 , 2 > * p_source ;
  if ( tile_width == 0 )
    p_source = & line_source ;
  else
    p_source = & tile_source ;

  zimt::tile_loader < ele_type , 3 , 2 > tl ( *p_source , bill ) ;
  zimt::tile_storer < ele_type , 3 , 2 > tp ( tile_drain , bill ) ;

  // showtime! We use a rotate_rgb_t object as act functor, which
  // affects the RGB rotation, so that we can see in the output that
  // something has indeed happened to the data. The execution time of
  // the act functor is quite negligible compared to the disk I/O,
  // and even quite complex functionality can 'fit into' this
  // scheme of operation without causing much delay, even though
  // the operation isn't multithreaded.
  // Why not load the entire image, process it and store it again?
  // Because this here scheme needs much less memory: two rows
  // of tiles - one for the input, one for the output. So we can
  // process huge files, and even several of them, without too
  // much memory load and with better cache efficiency. Of course
  // this depends on the coorperation of the I/O process: if that
  // buffers the data in memory, we can't help it. If direct writing
  // of the tiles to individual tile files is wanted, use a
  // basic_tile_store_t object for storing data instead.

  zimt::process < 2 > ( { w , h } , tl , rotate_rgb_t() , tp , bill ) ;

  std::cout << "load count: " << load_count << std::endl ;
  std::cout << "store count: " << store_count << std::endl ;

  inp->close() ;
  std::cout << "inp->close() returned" << std::endl ;

  out->close() ;
  std::cout << "out->close() returned" << std::endl ;
}
