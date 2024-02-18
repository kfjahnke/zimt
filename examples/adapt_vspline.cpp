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

// adapt a vspline::unary_functor which uses vigra::TinyVectors rather
// than xel_t. vspline::unary_functor and zimt::unary_functor are
// just about the same thing, but vspline uses vigra::TinyVector where
// zimt uses zimt::xel_t. Both are - internally - nothing but a C vector
// of a few T, so they are binary-compatible and we can adapt the vspline
// version to the zimt version by reinterpret-casting the arguments to
// eval. There is a handy factory function named zimt::uf_adapt (it's
// in unary_functor.h) just for the purpose.
// This example needs a vspline installation, so it has a .cpp
// extension to mark it for 'special treatment'.

#include <zimt/zimt.h>
#include <vspline/vspline.h>

typedef vigra::TinyVector < float , 3 > vigra_pixel_t ;
typedef zimt::xel_t < float , 3 > pixel_t ;

int main ( int argc , char * argv[] )
{
  typedef vspline::amplify_type < vigra_pixel_t > wrappee_t ;
  auto wrapped = zimt::uf_adapt ( wrappee_t ( 2.0 ) ) ;

  pixel_t in { 1.0f , 2.0f , 3.0f } ;
  pixel_t out ;

  wrapped.eval ( in , out ) ;
  std::cout << in << " -> " << out << std::endl ;

  typedef decltype(wrapped)::out_v pixel_v ;

  pixel_v in_v = in ;
  pixel_v out_v ;

  wrapped.eval ( in_v , out_v ) ;
  std::cout << in_v << " -> " << out_v << std::endl ;
}
