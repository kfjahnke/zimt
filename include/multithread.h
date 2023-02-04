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

/// \file multithread.h
///
/// \brief code to distribute the processing of bulk data to several threads
/// 
/// The code in this header provides a resonably general method to perform
/// processing of manifolds of data with several threads in parallel. In zimt,
/// there are several areas where potentially large numbers of individual values
/// have to be processed independently of each other or in a dependence which
/// can be preserved in partitioning. To process such 'bulk' data effectively,
/// zimt employs two strategies: multithreading and vectorization.
/// This file handles the multithreading.
///
/// multithreading, the use of related headers and linkage with pthread are
/// optional in zimt and can be switched off by #defining ZIMT_SINGLETHREAD.
/// This is the reason for the template zimt::atomic, which normally uses
/// std::atomic, but instead uses a stand-in type providing 'mock' functionality
/// if ZIMT_SINGLETHREAD is defined, to allow the use of the same logic.
///
/// As of March 2019, the multithreading code has been simplified: The routine
/// 'multithread' takes a std::function<void()> as 'payload'. The payload code
/// is wrapped in an outer function keeping track of worker threads terminating,
/// and when the last of a set of worker threads who have been assigned a job
/// terminates, control is returned to the caller of 'maultithread'. This logic
/// ensures that a multithreaded task is complete when control returns.
///
/// This logic implies that the caller and the payload code cooperate to
/// split up the work load, since 'multithread' itself does not concern itself
/// with this aspect. In zimt, this is done using a zimt::atomic instance
/// in the caller, which is passed into the workers and used by them to obtain
/// 'joblet numbers', which they interpret as signifying a (small-ish) share of
/// some larger data set. The passing-to-the-workers is done by per-reference
/// lambda capture, which conveniently transports the caller's context into
/// the workers. The payload code keeps taking 'joblet numbers' from the atomic
/// until they are finished - then it terminates.
///
/// This strategy separates the granularity of the workload distribution from
/// the number of worker threads, resulting in even load distribution and little
/// tail-end idling (when most jobs are complete and only a few aren't) - and
/// it also makes any data partitioning code unnecessary: jobs which are laid
/// to rest by the OS may, on re-awakening, have some data left to process, but
/// if the remainder of the job is done (joblet numbers are finished) that's all
/// they have to do, taking much less time than having to complete some previously
/// scheduled fixed work load. It also allows running some task 'on the back
/// burner', employing only a small number of workers: Since load distribution
/// is automatic, the job will only take longer to execute.
///
/// I like this technique, and I've dubbed it 'atomic surfing' ;)
///
/// So now it should be clear why a stand-in type is needed if ZIMT_SINGLETHREAD
/// is #defined: using the atomic to share joblet numbers between caller and
/// workers is part of the logic, and it's easier to just code using them and
/// using the stand-in type for the single-threaded case, which preserves the
/// logic, but gets by without any reference to multithreading-related headers
/// or libraries.
///
/// In zimt, the granularity of 'joblets' is single 1D subarrays of a larger
/// data set. If the data are images, joblets process lines. The number of threads
/// used to process an entire data set is fixed in thread_pool.h, and is some small
/// multiple of the number of available physical cores. This seems strange, because
/// one might assume that it should be best to have as many threads as cores, but
/// I found that using more (up to a point) increases performance, and since one of
/// my main concerns in zimt is speed, I've coded so that per default a number
/// of threads is chosen which runs best *on my system* - thiy may not be optimal
/// for other hardware. To change the number, see the definition of default_njobs
/// in thread_pool.h.

#ifndef ZIMT_MULTITHREAD_H
#define ZIMT_MULTITHREAD_H

#include <assert.h>

#ifndef ZIMT_SINGLETHREAD

// only include multithreading-related headers if ZIMT_SINGLETHREAD
// is *not* defined. With ZIMT_SINGLETHREAD defined, all use of threading
// code is switched off and linking with pthread is unnecessary.

#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <atomic>
#include "thread_pool.h"

#endif // #ifndef ZIMT_SINGLETHREAD

#include "common.h"

namespace zimt
{

// if ZIMT_SINGLETHREAD is defined, we provide some fallback code which
// allows the remainder of zimt to remain unaware of the fact and use the
// same logic that's used for multithreaded operation

#ifdef ZIMT_SINGLETHREAD

const int ncores = 1 ;

const int default_njobs = 1 ;

// 'multithread' itself collapses to merely executing the payload code in the
// current thread:

template < bool dummy = true >
int multithread ( std::function < void() > payload ,
                  std::size_t nr_workers = 1 )
{
  // guard against empty or wrong number

  if ( nr_workers <= 0 )
  {
    return 0 ;
  }

  payload() ;
  return 1 ;
}

// we need a stand-in for zimt::atomic since atomics are used where
// multithreading normally takes place, but with ZIMT_SINGLETHREAD
// <atomic> is not #included. The stand-in type only provides a minimal
// set of functions, namely those which are used inside zimt.

template < typename T >
struct atomic
{
  typedef T value_type ;

  value_type value ;
  
  atomic ( const value_type & _value )
  : value ( _value ) { } ;

  value_type load()
  {
    return value ;
  }

  value_type operator++()
  {
    return ++value ;
  }
  
  value_type operator--()
  {
    return --value ;
  }
  
  value_type operator++ ( int )
  {
    return value++ ;
  }
  
  value_type operator-- ( int )
  {
    return value-- ;
  }
  
  value_type fetch_sub ( value_type arg )
  {
    value_type v = value ;
    value -= arg ;
    return v ;
  }
  
  value_type fetch_add ( value_type arg )
  {
    value_type v = value ;
    value += arg ;
    return v ;
  }
} ;

#else // ZIMT_SINGLETHREAD

const int ncores = zimt_threadpool::ncores ;
const int default_njobs = zimt_threadpool::default_njobs ;

// when using multithreading, zimt::atomic is an alias for std::atomic

template < typename T > using atomic = std::atomic < T > ;

#endif // ZIMT_SINGLETHREAD

// we start out with a bit of collateral code. I have changed the
// multithreading code to not use ranges anymore. This simplifies
// the code greatly, and calling code will now follow a specific
// pattern: it will set up a zimt::atomic initialized to some total
// number of 'joblets', which are taken to mean indexes which can be
// applied either directly to pointers and C++ arrays, or to iterators.
// The payload code receives a pointer or reference to this atomic.
// With the new multithreading logic, all workers get precisely the
// same payload routine and are responsible to obtain shares of the
// total work load autonomously. This is realized by obtaining
// indexes from the atomic. There are two typical cases: the payload
// code may want to process the indexes singly, or in batches of
// a certain size. The fetch_XXX routines below are utility code to
// obtain such indexes. I provide two variants for each case, one
// variant counting the indexes up from zero, the other counting
// down to zero. The payload code runs a loop repeatedly calling
// fetch_XXX. If fetch_XXX returns false, there are no more indexes
// to be had, the loop is exited, and the payload routine finishes.
// If fetch_XXX returns true, the index(es) were set and the payload
// code can use them in whichever way is appropriate, and then
// proceed to the next iteration.
//
// If ZIMT_SINGLETHREAD is defined, the stand-in type above
// is used in stead of a std::atomic, so the logic can remain the same:
// the joblet numbers are now consumed by the caller's thread one after
// the other, just like in an ordinary loop, and with minimal overhead,
// which may even be optimized away completely.

/// fetch_descending fetches the next index from an atomic,
/// counting down. Indexes will range from one less than the value
/// the atomic was initialized with, down to zero. If all indexes were
/// distributed already, false is returned, true otherwise. Like the
/// other fetch_XXX routines, the return of a boolean makes these
/// functions good candidates to be used as conditional in a loop.

template < typename index_t >
bool fetch_descending ( zimt::atomic < index_t > & source ,
                        index_t & index )
{
  index_t _index = --source ;
  
  if ( _index < 0 )
    return false ;

  index = _index ;
  return true ;
}

/// fetch_ascending counts up from zero to total-1, which is more
/// efficient if the indexes are used to address memory. This is due
/// to the side effects of accessing memory: if memory is accessed at
/// an address x, the OS will typically fetch a chunk of data starting
/// at or shortly before x. If the next fetch requires data just after
/// x, there is a good chance that they are already in cache.

template < typename index_t >
bool fetch_ascending ( zimt::atomic < index_t > & source ,
                       const index_t & total ,
                       index_t & index )
{
  index_t _index = --source ;
  
  if ( _index < 0 )
    return false ;

  index = total - 1 - _index ;
  return true ;
}

/// fetch_range_ascending fetches the beginning and end of a range of
/// indexes (in iterator semantic, low <= i < high) from a
/// zimt::atomic which has been initialized with the total number
/// of indexes that are to be processed. If the zimt::atomic, when
/// accessed, already holds a value of or below zero, fetch_index_range
/// returns false and leaves low and high unchanged. Otherwise it
/// returns true and low and high will be set to the values gleaned
/// from the atomic, raising 'low 'to zero if it would come out below.
/// this function (and the next) enable calling code to process batches
/// of adjacent indexes without any index artistry: the code is perfectly
/// general and simple, and the use of the atomic and fetch_sub garantees
/// that each fetch provides a distinct batch of indexes.

template < typename index_t >
bool fetch_range_descending ( zimt::atomic < index_t > & source ,
                              const index_t & count ,
                              index_t & low ,
                              index_t & high )
{
  index_t high_index = source.fetch_sub ( count ) ;
  index_t low_index = high_index - count ;
  
  if ( high_index <= 0 )
    return false ;

  if ( low_index < 0 )
    low_index = 0 ;

  low = low_index ;
  high = high_index ;
  return true ;
}

/// fetch_range_ascending also uses an atomic initialized to the total
/// number of indexes to be distributed, but the successive ranges are
/// handed out in ascending order, which is more efficient if the indexes
/// are used to address memory.

template < typename index_t >
bool fetch_range_ascending ( zimt::atomic < index_t > & source ,
                             const index_t & count ,
                             const index_t & total ,
                             index_t & low ,
                             index_t & high )
{
  index_t high_index = source.fetch_sub ( count ) ;
  index_t low_index = high_index - count ;
  
  if ( high_index <= 0 )
    return false ;

  if ( low_index < 0 )
    low_index = 0 ;

  high = total - low_index ;
  low = total - high_index ;
  
  return true ;
}

#ifndef ZIMT_SINGLETHREAD

/// multithread uses a thread pool of worker threads to perform
/// a multithreaded operation. It receives a functor (a single-threaded
/// function used for all individual tasks), and, optionally, the
/// desired number of worker instances to be used.
/// These tasks are wrapped with a wrapper which takes care of
/// signalling when the last task has completed.
///
/// This, in a way, is the purest implementation of 'multithread': this
/// implementation is not involved with any property of the jobs at hand, it
/// merely forwards the arguments for the payload function to a specified
/// number of workers. The code invoking 'multithread' has to set up
/// whatever scheme it deems appropriate to convey, via the arguments,
/// what the worker threads should do. with this mechanism in place, the
/// calling code has the very efficient option of setting up a zimt::atomic
/// holding some sort of 'joblet number' and passing a reference to this atomic
/// to the payload code. The worker(s) executing payload code all get
/// equal load until the job numbers are exhausted, which is when they
/// terminate one by one. Since there is no inter-thread communication
/// during the active phase, there is no signalling overhead at all,
/// which allows fine granularity. The fine granularity ensures little
/// tail-end idling (when the caller has to wait for the last worker to
/// finish) and also makes it possible to choose some sort of partitioning
/// which insinuates itself from the structure of the data at hand, rather
/// than some preconceived parcel of the total job. When there are many more
/// joblet numbers than workers, intermittent inactivity of a worker simply
/// makes it consume fewer job numbers, rather than delaying it on a way
/// to a preset goal.
///
/// Another effect is that, if job numbers relate to memory worked on,
/// (think of lines of an image) - all activity is focussed in a narrow
/// band of the memory, because the currently processed job numbers are
/// all usually in sequence (unless the OS throws a spanner in the works
/// by halting threads). This may or may not help - in some situations
/// having several threads access adjacent memory locations may make it
/// harder for the system to synchronize access. But payload code is
/// free to use any interpretation of job numbers anyway, so that's an
/// issue on the payload side.
///
/// Since the number of threads in the pool is static, requesting more
/// workers than there are threads is futile (but still works). Requesting
/// fewer may be useful to have some task 'on the back burner' while some
/// critical task receives more workers to complete faster.
///
/// As I've pointed out in thread_pool.h, it seems beneficial (at least for
/// zimt) to have a good deal more threads than physical cores. See there
/// for reasons why this may be so.
///
/// Last but not least: the code is very simple :)
///
/// Why is multithread a template? So that several TUs can #include
/// zimt without linker errors.

template < bool dummy = true >
int multithread ( std::function < void() > payload ,
                  std::size_t  nr_workers = default_njobs )
{
  // guard against empty or wrong number

  if ( nr_workers <= 0 )
  {
    return 0 ;
  }

  if ( nr_workers == 1 )
  {
    // if only one worker is to be used, we take a shortcut
    // and execute the payload function right here:

    payload() ;
    return 1 ;
  }

  // TODO: I'd rather use the code where the count is kept in an atomic,
  // but I get failure to join in the thread pool's d'tor.
  // This variant seems to terminate reliably.
  
  int count = nr_workers ;          // number of tasks
  std::mutex pool_mutex ;           // mutex guarding count and pool_cv
  std::condition_variable pool_cv ; // cv for signalling completion

  // first we create the callable which is passed to the worker threads.
  // this wrapper around the 'payload' takes care of signalling when the
  // last worker thread has finished with it's current job.

  auto action = [&]
  {

    // execute the 'payload'

    payload() ;

    {
      // under pool_mutex, check if this was the last missing worker to
      // terminate. The lock is released with the closing scope; the
      // notify call to the condition variable does not need the lock,
      // the docu says that that would even be a pessimization.
      // but - here the notify is back under the lock_guard, had
      // random crashes again, see if this fixes it.

      std::lock_guard<std::mutex> lk ( pool_mutex ) ;

      if ( ( -- count ) == 0 )

        pool_cv.notify_one() ;

    }

  } ;

  {
    // acquire a lock on pool_mutex to stop any action finishing early
    // from modifying 'count'

    std::unique_lock<std::mutex> lk_pool ( pool_mutex ) ;

    zimt_threadpool::common_thread_pool.launch ( action , nr_workers ) ;

    // now wait for the last task to complete. This is signalled by
    // the action code by notifying on pool_cv
    // the predicate count == 0 rejects spurious wakes
    
    pool_cv.wait ( lk_pool , [&] { return count == 0 ; } ) ;
  }
  
  // all jobs are done

  return nr_workers ;
}

#endif // ZIMT_SINGLETHREAD

} ; // end of namespace zimt

#endif // #ifndef ZIMT_MULTITHREAD_H
