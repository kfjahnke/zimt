/************************************************************************/
/*                                                                      */
/*    zimt - abstraction layer for SIMD programming                     */
/*                                                                      */
/*            Copyright 2024 by Kay F. Jahnke                           */
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

/*! \file scanlines.h

    \brief using tiled storage access to interface with files providing
           data as individual scanlines, like many image files do. This
           is more a proposition/example than 'library code proper' -
           derived classes might override load_tile and store_tile
           directly.

*/

#include "tiles.h"

namespace zimt
{

template < typename T , std::size_t N >
struct line_store_t
: public basic_tile_store_t < T , N , 2 >
{
  typedef basic_tile_store_t < T , N , 2 > base_t ;
  using typename base_t::tile_type ;
  using typename base_t::value_t ;
  using typename base_t::shape_type ;
  using typename base_t::index_type ;
  using base_t::base_t ;
  using base_t::tile_shape ;

  // we will use two std::functions to move scanlines from external
  // storage to the tile's memory and back. These are their types:

  typedef std::function < bool ( T * ,
                                 std::size_t ,
                                 std::size_t ) > line_f ;

  typedef std::function < bool ( const T * ,
                                 std::size_t,
                                 std::size_t ) > line_cf ;

  // And these are the corresponding objects, as passed to the 'ctor

  line_f load_line ;
  line_cf store_line ;

  // the c'tor takes the width and height of the image and the two
  // std::functions moving data.

  line_store_t ( std::size_t width ,
                 std::size_t height ,
                 line_f _load_line ,
                 line_cf _store_line )
  : base_t ( { width , height } , { width , 1UL } , "" ) ,
    load_line ( _load_line ) ,
    store_line ( _store_line )
  { }

  // load_tile and store_tile adapt the simple load_line and store_line
  // functions to the zimt tile store logic and conventions.

  virtual bool load_tile ( tile_type * p_tile ,
                           const index_type & tile_index ) const
  {
    auto p_data ( p_tile->p_data->data() ) ;
    assert ( p_data != nullptr ) ; // might alocate instead

    std::size_t nbytes = tile_shape[0] * sizeof(T) * N ;
    bool success = load_line ( (T *) p_data ,
                               nbytes , tile_index[1] ) ;
    if ( success )
      ++ load_count ; // for statistics, can go later
    return success ;
  }

  // Store a tile's storage array to a file. The function assumes
  // that the tile holds a single compact block of data, as it is
  // created by the allocation routine.

  virtual bool store_tile ( tile_type * p_tile ,
                            const index_type & tile_index ) const
  {
    auto const p_data ( p_tile->p_data->data() ) ;
    assert ( p_data != nullptr ) ;

    std::size_t nbytes = tile_shape[0] * sizeof(T) * N ;
    bool success = store_line ( (const T *) p_data ,
                                nbytes , tile_index[1] ) ;
    if ( success )
      ++ store_count ; // for statistics, can go later
    return success ;
  }

} ;

// this class presents an interface for external storage in the form
// of square tiles.

template < typename T , std::size_t N >
struct square_store_t
: public basic_tile_store_t < T , N , 2 >
{
  typedef basic_tile_store_t < T , N , 2 > base_t ;
  using typename base_t::tile_type ;
  using typename base_t::value_t ;
  using typename base_t::shape_type ;
  using typename base_t::index_type ;
  using base_t::base_t ;
  using base_t::tile_shape ;

  // we will use two std::functions to move tile data from external
  // storage to the tile's memory and back. The column and line parameters
  // refer to image coordinates, we'll pass the upper left corner.
  // These are their types

  typedef std::function < bool ( T * ,                    // target
                                 std::size_t ,            // nbytes
                                 std::size_t ,            // column
                                 std::size_t ) > line_f ; // line

  typedef std::function < bool ( const T * ,               // source
                                 std::size_t,              // nbytes
                                 std::size_t ,             // column
                                 std::size_t ) > line_cf ; // line

  // And these are the corresponding objects, as passed to the 'ctor

  line_f _load_tile ;
  line_cf _store_tile ;

  // the c'tor takes the width and height of the image and the two
  // std::functions moving data.

  square_store_t ( std::size_t width ,
                   std::size_t height ,
                   std::size_t tile_size ,
                   line_f p_load_tile ,
                   line_cf p_store_tile )
  : base_t ( { width , height } , { tile_size , tile_size } , "" ) ,
    _load_tile ( p_load_tile ) ,
    _store_tile ( p_store_tile )
  { }

  // load_tile and store_tile adapt the simple load_line and store_line
  // functions to the zimt tile store logic and conventions.

  virtual bool load_tile ( tile_type * p_tile ,
                           const index_type & tile_index ) const
  {
    auto p_data ( p_tile->p_data->data() ) ;
    assert ( p_data != nullptr ) ; // might alocate instead

    std::size_t nbytes = tile_shape[0] * tile_shape[1] * sizeof(T) * N ;
    bool success = _load_tile ( (T *) p_data ,
                                nbytes ,
                                tile_index[0] * tile_shape[0] ,
                                tile_index[1] * tile_shape[1] ) ;
    if ( success )
      ++ load_count ; // for statistics, can go later
    return success ;
  }

  // Store a tile's storage array to a file. The function assumes
  // that the tile holds a single compact block of data, as it is
  // created by the allocation routine.

  virtual bool store_tile ( tile_type * p_tile ,
                            const index_type & tile_index ) const
  {
    auto const p_data ( p_tile->p_data->data() ) ;
    assert ( p_data != nullptr ) ;

    std::size_t nbytes = tile_shape[0] * tile_shape[1] * sizeof(T) * N ;
    bool success = _store_tile ( (const T *) p_data ,
                                 nbytes ,
                                 tile_index[0] * tile_shape[0] ,
                                 tile_index[1] * tile_shape[1] ) ;
    if ( success )
      ++ store_count ; // for statistics, can go later
    return success ;
  }

} ;

} ; // namespace zimt
