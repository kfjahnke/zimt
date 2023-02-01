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

// example for the use of zimt:unary_functor with the code blocks
// given in README.md

#include "../zimt.h"

typedef zimt::xel_t < float , 3 > pixel_t ;

struct double_pixel
: public zimt::unary_functor < pixel_t , pixel_t , 8 >
{
  template < typename I , typename O >
  void eval ( const I & i , O & o ) const
  {
    o = i * 2.0f ;
  }
} ;

struct capped_double_pixel
: public zimt::unary_functor < pixel_t , pixel_t , 8 >
{
  void eval ( const in_type & i , out_type & o ) const
  {
    o = i * 2.0f ;
    for ( int l = 0 ; l < 3  ; l++ )
    {
      if ( o[l] > 255.0f )
        o[l] = 255.0f ;
    }
  }
  void eval ( const in_v & i , out_v & o ) const
  {
    o = i * 2.0f ;
    for ( int l = 0 ; l < 3  ; l++ )
      o[l] ( o[l] > 255.0f ) = 255.0f ;
  }
} ;

pixel_t halve ( const pixel_t & in )
{
  return in * .5f ;
}

int main ( int argc , char * argv[] )
{
  pixel_t px1 { 100.0f , 200.0f , 300.0f } , px2 ;
  typedef zimt::vector_traits < pixel_t , 8 > :: type pixel_v ;
  pixel_v pxv1 { px1[0] , px1[1] , px1[2] } , pxv2 ;
  capped_double_pixel f ;
  f.eval ( px1 , px2 ) ;
  f.eval ( pxv1 , pxv2 ) ;
  std::cout << px1 << " -> " << px2 << std::endl ;
  std::cout << pxv1 << " -> " << pxv2 << std::endl ;

  auto ff = f + f ;
  ff.eval ( pxv1 , pxv2 ) ;
  std::cout << pxv1 << " -> " << pxv2 << std::endl ;

  zimt::broadcast_type < pixel_t , pixel_t , 8 > halve_f ( halve ) ;
  halve_f.eval ( pxv1 , pxv2 ) ;
  std::cout << pxv1 << " -> " << pxv2 << std::endl ;

}
