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

/// \file thread_pool.h
///
/// \brief provides a thread pool for zimt's multithread() routine
///
/// class thread_pool aims to provide a simple and straightforward implementation
/// of a thread pool for multithread() in multithread.h, but the class might find
/// use elsewhere. The operation is simple, I think of it as 'piranha mode' ;)
///
/// a set of worker threads is launched which wait for 'tasks', which come in the shape
/// of std::function<void()>, from a queue. When woken, a worker thread tries to obtain
/// a task. If it succeeds, the task is executed, and the worker thread tries to get
/// another task. If none is to be had, it goes to sleep, waiting to be woken once
/// there are new tasks.

#ifndef ZIMT_THREADPOOL_H
#define ZIMT_THREADPOOL_H

#include <thread>
#include <mutex>
#include <queue>
#include <functional>
#include <condition_variable>

// KFJ 2019-09-02 new namespace zimt_threadpool
// I am now keeping class thread_pool and common_thread_pool in this
// separate namespace to facilitate 'dubbing' of zimt. With dubbing
// I mean using preprocessor manoevres like "#define zimt NS_AVX"
// used to create independent, ISA-specific compiles of zimt in
// several TUs, to be linked together into a 'monolithic' binary.
// While this worked with the thread pool code in namespace zimt,
// it created a separate thread pool for each ISA-specific TU. This
// is wasteful and unnecessary; the thread pool code is the same and
// it's ISA-independent. With the new namespace, the ISA-specific TUs
// are compiled with -D ZIMT_EXTERN_THREAD_POOL, and one TU has to
// provide the pool by containing
//
// namespace zimt_threadpool
// {
//   thread_pool common_thread_pool ;
// } ;
//
// less complex scenarios where only one TU #includes zimt
// get a static thread pool, as before - only that it now lives in
// a separate namespace.

namespace zimt_threadpool
{

/// number of CPU cores in the system

const int ncores = std::thread::hardware_concurrency() ;

/// when multithreading, use this number of jobs per default.
/// This looks like overkill and unnecessary signalling overhead,
/// but it improves performance over just having as many threads as
/// there are physical cores. Why is this so? There are several
/// possibilities I've considered:
/// - while one task waits e.g. for memory, another task can perform
///   computations
/// - the scheduler might assign time slices to each thread, so
///   having more threads yields more time slices

const int default_njobs = 2 * ncores ;

class thread_pool
{
  // used to switch off the worker threads at program termination.
  // access under task_mutex.

  bool stay_alive = true ;

  // the thread pool itself is held in this variable. The pool
  // does not change after construction

  std::vector < std::thread * > pool ;
  
public:

  // mutex and condition variable for interaction with the task queue
  // and stay_alive

  std::mutex task_mutex ;
  std::condition_variable task_cv ;
  
  // queue to hold tasks. access under task_mutex

  std::queue < std::function < void() > > task_queue ;

private:
  
  /// code to run a worker thread
  /// We use a thread pool of worker threads. These threads have a very 
  /// simple cycle: They try and obtain a task (std::function<void()>). 
  /// If there is one to be had, it is invoked, otherwise they wait on
  /// task_cv. When woken up, the flag stay_alive is checked, and if it
  /// is found to be false, the worker thread ends.
  
  void worker_thread()
  {
    while ( true )
    {
      // under task_mutex, check stay_alive and try to obtain a task

      std::unique_lock<std::mutex> task_lock ( task_mutex ) ;

      if ( ! stay_alive )
      {
        task_lock.unlock() ;
        break ; // die
      }

      if ( task_queue.size() )
      {
        // there are tasks in the queue, take one, unlock

        auto task = task_queue.front() ;
        task_queue.pop() ;
        task_lock.unlock() ;

        // got a task, perform it, then try for another one

        task() ;
      }
      else
      {
        // no luck. wait until alerted

        task_cv.wait ( task_lock ) ; // spurious alert is okay
      }

      // now start next cycle, either after having completed a job
      // or after having been woken by an alert
    }
  }

public:

  // Only as many threads as there are physical cores can run at the same
  // time, so one might assume that having more threads is futile.
  // Surprisingly - at least on my system - this is not so: I get the
  // best performance with a significantly larger number of threads. I'm
  // not sure why this is so, see above for some possible reasons.

  thread_pool ( int nthreads = default_njobs )
  {
    // to launch a thread with a method, we need to bind it to the object:

    std::function < void() > wf
      = std::bind ( &thread_pool::worker_thread , this ) ;

    // now we can fill the pool with worker threads

    for ( int t = 0 ; t < nthreads ; t++ )
      pool.push_back ( new std::thread ( wf ) ) ;
  }

  int get_nthreads() const
  {
    return pool.size() ;
  }

  /// launch simply enqueues a job and calls notify_one. Such a job
  /// will run to it's completion and end silently - any communication
  /// of it's state has to be managed by the job itself. See
  /// multithread.h for code which takes care of managing the life cycle
  /// of a group of jobs by wrapping them in an additional outer function

  void launch ( std::function < void() > job )
  {
    {
      std::lock_guard<std::mutex> lk_task ( task_mutex ) ;
      task_queue.push ( job ) ;
    }

    task_cv.notify_one() ;
  }

  /// overload of launch invoking the payload on several worker threads

  void launch ( std::function < void() > job , int njobs )
  {
    if ( njobs <= 0 )
      return ;

    {
      std::lock_guard<std::mutex> lk_task ( task_mutex ) ;
      for ( int i = 0 ; i < njobs ; i++ )
        task_queue.push ( job ) ;
    }

    task_cv.notify_all() ;
  }

  ~thread_pool()
  {
    {
      // under task_mutex, set stay_alive to false

      std::lock_guard<std::mutex> task_lock ( task_mutex ) ;
      stay_alive = false ;      
    }

    // wake all inactive worker threads,
    // join all worker threads once they are finished

    task_cv.notify_all() ;

    for ( auto threadp : pool )
    {
      threadp->join() ;
    }

    // once all are joined, delete their std::thread object

    for ( auto threadp : pool )
    {
      delete threadp ;
    }
  }
} ;

#ifndef ZIMT_SINGLETHREAD

// if ZIMT_EXTERN_THREAD_POOL is #defined, we rely on some other TU
// providing the common thread pool. If it is not #defined, we use a static
// thread pool for the TU #including this header. This should be safe
// because the include guards should guarantee that each TU includes
// this section here precisely once.
 
#ifdef ZIMT_EXTERN_THREAD_POOL

extern thread_pool common_thread_pool ;

#else

static thread_pool common_thread_pool ;

#endif // ZIMT_EXTERN_THREAD_POOL

#endif // ZIMT_SINGLETHREAD

} ; // end of namespace zimt_threadpool

#endif
