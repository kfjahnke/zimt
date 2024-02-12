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
           data as individual scanlines, like many image files do.

*/

#include "tiles.h"

namespace zimt
{

template < typename T , std::size_t N , std::size_t D >
struct line_store_t
: public tile_store_t < T , N , D >
{
  typedef tile_store_t < T , N , D > base_t ;
  using typename base_t::tile_type ;
  using typename base_t::value_t ;
  using typename base_t::shape_type ;
  using typename base_t::index_type ;
  using base_t::base_t ;
  using base_t::tile_shape ;

  typedef std::function < bool ( unsigned char* ,
                                 std::size_t ,
                                 std::size_t ) > line_f ;

  typedef std::function < bool ( const unsigned char* ,
                                 std::size_t,
                                 std::size_t ) > line_cf ;

  line_f load_line ;
  line_cf store_line ;

  line_store_t ( shape_type _array_shape ,
                 shape_type _tile_shape ,
                 line_f _load_line ,
                 line_cf _store_line )
  : base_t ( _array_shape , _tile_shape , "" ) ,
    load_line ( _load_line ) ,
    store_line ( _store_line )
  {
    assert ( tile_shape[1] == 1 ) ;
  }

  virtual bool load_tile ( tile_type * p_tile ,
                           const index_type & tile_index ) const
  {
    auto p_data ( p_tile->p_data->data() ) ;
    assert ( p_data != nullptr ) ; // might alocate instead

    std::size_t nbytes = tile_shape[0] * sizeof(T) * N ;
    bool success = load_line ( (unsigned char*) p_data ,
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
    bool success = store_line ( (const unsigned char*) p_data ,
                                nbytes , tile_index[1] ) ;
    if ( success )
      ++ store_count ; // for statistics, can go later
    return success ;
  }

} ;

} ; // namespace zimt
