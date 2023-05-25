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

// This example demonstrates 'cherrypicking': Here, we use std::simd as
// the standard backend, but implement two functors calculating atan2
// of a 2D float coordinate: one using std::simd, and one using Vc.
// Then we apply the functor 1000 times to a 1000000-coordinate array,
// measuring the time. The difference between the two is significant
// (ca. fourfold on my system) and shows shat the std::simd implementation
// (as of this writing, and as far as I can tell) simply uses a loop over
// the vector's lanes, whereas Vc uses dedicated hand-written SIMD code.
// So even with the move from the std::simd data type to the Vc type and
// back - which, I assume, is optimized away - we can speed code using
// atan2 up by a fair amount if we delegate to Vc's superior implementation.
//
// note that you can't use examples.sh for this file. compile like this:
// g++ -Ofast -std=c++17 -march=native -ocherrypicking cherrypicking.cpp -lVc

#include <memory>
#include <assert.h>

// we want to use two of zimt's 'backend' SIMD types, the one using
// std::simd and the one using Vc. Note how we directly include
// the backend headers - if we were to go through the usual motions
// and #define USE_... already, we'd only have either header included.

#include <zimt/common.h>
#include <zimt/simd/std_simd_type.h>
#include <zimt/simd/vc_simd_type.h>
#include <zimt/simd/gen_simd_type.h>
#include <zimt/simd/hwy_simd_type.h>

typedef simd::std_simd_type < float , 16 > stdf16_t ;
typedef simd::vc_simd_type < float , 16 > vcf16_t ;
typedef simd::hwy_simd_type < float , 16 > hwyf16_t ;
typedef simd::gen_simd_type < float , 16 > genf16_t ;

// first version: use std_simd_type's (inefficient) atan2 function

void f ( const stdf16_t & x , const stdf16_t & y , stdf16_t & out )
{
  out = atan2 ( y , x ) ;
}

// second version: delegate to vc_simd_type instead. As in f(), we
// have stdf16_t incoming and outgoing, but here, the values are moved
// to corresponding vc_simd_type objects, Vc's atan2 function is used
// to produce a vc_simd_type result which is moved to 'out'. Even
// with the copying around of the values, this one is much faster.
// The optimzer likely takes care of the unnecessary loads and stores
// and keeps the arguments 'afloat' in registers.

void g ( const stdf16_t & x , const stdf16_t & y , stdf16_t & out )
{
  vcf16_t h_x , h_y , h_out ;
  float help [ 16 ] ;

  x.store ( help ) ;
  h_x.load ( help ) ;

  y.store ( help ) ;
  h_y.load ( help ) ;

  h_out = atan2 ( h_y , h_x ) ;

  h_out.store ( help ) ;
  out.load ( help ) ;
}

void h ( const stdf16_t & x , const stdf16_t & y , stdf16_t & out )
{
  hwyf16_t h_x , h_y , h_out ;
  float help [ 16 ] ;

  x.store ( help ) ;
  h_x.load ( help ) ;

  y.store ( help ) ;
  h_y.load ( help ) ;

  h_out = atan2 ( h_y , h_x ) ;

  h_out.store ( help ) ;
  out.load ( help ) ;
}

void k ( const stdf16_t & x , const stdf16_t & y , stdf16_t & out )
{
  genf16_t h_x , h_y , h_out ;
  float help [ 16 ] ;

  x.store ( help ) ;
  h_x.load ( help ) ;

  y.store ( help ) ;
  h_y.load ( help ) ;

  h_out = atan2 ( h_y , h_x ) ;

  h_out.store ( help ) ;
  out.load ( help ) ;
}

// use std::simd backend as standard for the remainder.

#define USE_STDSIMD
#include <zimt/zimt.h>

// we build unary functors, the first uses f() , the second g():

struct atan_f1
: public zimt::unary_functor < zimt::xel_t < float , 2 > , float , 16 >
{
  void eval ( const in_v & in , out_v & out ) const
  {
    f ( in[1] , in[0] , out ) ;
  }
} ;

struct atan_f2
: public zimt::unary_functor < zimt::xel_t < float , 2 > , float , 16 >
{
  void eval ( const in_v & in , out_v & out ) const
  {
    g ( in[1] , in[0] , out ) ;
  }
} ;

struct atan_f3
: public zimt::unary_functor < zimt::xel_t < float , 2 > , float , 16 >
{
  void eval ( const in_v & in , out_v & out ) const
  {
    h ( in[1] , in[0] , out ) ;
  }
} ;

struct atan_f4
: public zimt::unary_functor < zimt::xel_t < float , 2 > , float , 16 >
{
  void eval ( const in_v & in , out_v & out ) const
  {
    k ( in[1] , in[0] , out ) ;
  }
} ;

// main transforms array 'in' to array 'out' 1000 times, using one of
// the back-end-specififc functors. To get an informative result, compile
// the .cc files with -Ofast and -march=native, then link and time. On my
// system, the first version takes significantly longer. This shows that
// std::simd's atan2 is inefficient. It also shows that it's feasible
// to use one backend (here the Vc backend) to provide an implementation
// of a function for code using another backend (std::simd), implying that
// by using several backends we can 'cherrypick' so that those capabilities
// which a particular backend excels in can amend deficiencies in others.
// Of course, there's an overhead due to moving the data from one simd type
// to another.
// With the code as it stands, the compiler is also a big issue. With
// g++, as of this writing, using no SIMD library (choice 4) is faster
// than all the others.

#include <random>
#include <ctime>
#include <chrono>

int main ( int argc , char * argv[] )
{
  if ( argc < 2 )
  {
    std::cerr << "pass '1' to use std::simd's atan2, '2' to use Vc's"
              << std::endl ;
    std::cerr << "pass '3' to use highway's atan2, '16' to use goading"
              << std::endl ;
    exit ( 1 ) ;
  }

  zimt::xel_t < long , 1 > shape = 1000000 ;
  zimt::array_t < 1 , zimt::xel_t < float , 2 > > in ( shape ) ;
  zimt::array_t < 1 , float > out ( shape ) ;

  // produce random 2D coordinates

  std::random_device rd ;
  std::mt19937 gen ( rd() ) ;

  std::uniform_real_distribution<> crd_dis ( -300.0 , 300.0 ) ;
  for ( std::size_t i = 0 ; i < shape[0] ; i++ )
    in[i] = { float ( crd_dis ( gen ) ) , float ( crd_dis ( gen ) ) } ;

  auto start = std::chrono::system_clock::now() ;
  for ( int times = 0 ; times < 1000 ; times++ )
  {
    if ( argv[1][0] == '1' )
      zimt::transform ( atan_f1() , in , out ) ;
    else if ( argv[1][0] == '2' )
      zimt::transform ( atan_f2() , in , out ) ;
    else if ( argv[1][0] == '3' )
      zimt::transform ( atan_f3() , in , out ) ;
    else if ( argv[1][0] == '4' )
      zimt::transform ( atan_f4() , in , out ) ;
    else
    {
      std::cerr << "pass a value between one and four" << std::endl ;
      exit ( -1 ) ;
    }
  }

  auto end = std::chrono::system_clock::now();
  std::cout << "atan2 of 1e9 2D coordinates took "
            << std::chrono::duration_cast<std::chrono::milliseconds>
                (end - start).count()
            << " ms" << std::endl ;
}

