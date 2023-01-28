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

/// \file bill.h
///
/// \brief 'loading bill' for transform function family
///
/// the transform family of functions take a bill_t object which has
/// added information about how the operation should be executed.
/// This is in preference of several individual arguments which
/// was the status quo in vspline when I factored out this code,
/// but it started to become unwieldy, and most of the time I found
/// I used the defaults anyway, so I opted to throw the 'collateral'
/// arguments together in this class.

#ifndef ZIMT_BILL_H

// we need zimt::default_njobs from this header:

#include "multithread.h"

// if you don't #define WIELDING_SEGMENT_SIZE, the default is 512,
// meaning that the tarnsform family of functions will break down
// long lines into segments of up to 512 elements for processing.
// This is especially important for 1D arrays - if they are not
// 'segmented' the code becomes effectively single-threaded.
// You can override the default by passing a value in the 'loading
// bill'.

#ifndef WIELDING_SEGMENT_SIZE
#define WIELDING_SEGMENT_SIZE 512
#endif

namespace zimt
{

struct bill_t
{
  // 'aggregation axis'. This should be the axis with the smallest
  // stride, so for the standard memory order which zimt uses, this
  // would be axis 0. Which axis is chosen is merely a performance
  // issue - choosing an axis with longer strides makes the memory
  // access less performant. When one view is transformed into
  // a second one, the aggregation axis has to be the same.
  // Note that when you 'unfold' arrays containing xel_t data
  // to arrays of fundamentals, you'll have a short axis (as long
  // as the xel has channels) as axis 0. This axis should *not*
  // be used as aggregation axis, use axis 1 in this case. If you
  // use such arrays with aggregation axis zero, the processing
  // can't use SIMD code, because it can never find enough values
  // along the aggregation axis to fill a SIMD vector.

  std::size_t axis = 0 ;

  // the 'segment size' is another performance-relevant parameter.
  // zimt 'chops up' the view into individual lines for processing,
  // and the lines are - optionally - subdivided further into smaller
  // 'segments' to increase granularity. If you pass segment_size
  // zero, the lines will be processed whole.

  std::size_t segment_size = WIELDING_SEGMENT_SIZE ;

  // njobs tells zimt's multithreading how many threads should be
  // dedicated to the job at hand. This datum only has an effect if
  // multithreading is enabled - if ZIMT_SINGLETHREAD is defined,
  // multithreading is disabled altogether. If the views which are
  // to be processed are rather small, using several, or 'too many'
  // threads is detrimental due to the inevitable overhead. It's
  // hard to tell beforehand how many jobs will be optimal, because
  // this depends on the functor used for processing and the view
  // size. Hence the parameter.

  int njobs = zimt::default_njobs ;

  // the transform family of functions can take a long time, so at
  // times user code may wish to stop them prematurely, e.g. when it
  // becomes obvious that the result won't be needed after all or the
  // program has to be terminated. If this is anticipated, user code
  // can pass a pointer to a zimt::atomic<bool> which is checked
  // regularly, and if the contained value turns 'true' the transform
  // is cancelled prematurely - but orderly. the default is to pass
  // nullptr, indicating that the caller does not need a way to abort
  // the transform run prematurely. Why a pointer? If the atomic were
  // passed directly, every access would have to be mediated by the OS
  // to avoid clashes. If premature abort is needed, this can't be
  // helped - then some way of inter-thread communication is necessary.
  // But the test whether p_cancel is nullptr can be done thread-local.

  zimt::atomic < bool > * p_cancel = nullptr ;
} ;

} ;

#define ZIMT_BILL_H
#endif
