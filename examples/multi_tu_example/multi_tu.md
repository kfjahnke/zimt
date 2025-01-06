## Using ISA-specific Object Files

So far we've concentrated on ease of building and used highway's foreach_target mechanism to affect internal dispatch to SIMD-ISA-specific versions of of our code which highway places into a common object file. While this works well, this method also has disadvantages, which I'll try and explain. I'll also offer ways to work around the issues, but I can't offer solutions with keep the object file 'monolithic': instead, I'll split up the large multi-SIMD-ISA object file into several single-ISA object files and use the linker to put them together. In doing so, I still exploit highway's 'wisdom' as much as possible, delegating the nitty-gritty of how to generate ISA-specific binary, and how to detect the best ISA at run-time, to highway. All that's needed is a bit of effort in setting up the more complex build, and I'll use cmake for this task. But before I launch into code, let me explain when the 'split' is needed.

### Disadvantages of the multi-SIMD-ISA binary approach

There is one really obvious drawback with multi-SIMD-ISA binaries: the mere fact that they are monolithic. If you've ever tried to build a larger project in this way, you may have noticed that the compiler may take a fair while to get it's job done: it can only work through the single big compilation job sequentially, going through all possible SIMD ISAs one after the other until they have all been dealt with. This is counterproductive, because modern CPUs have several cores and might just as well perform several independent compilation jobs at the same time. It also takes up a lot of memory, so if your computer is low on RAM, the single large compilation job may require more memory than you have RAM available, sending the system swapping and reducing processing speed to a crawl. This is obviously also counterproductive - if you can split the compile job up into several and work them in sequence, each part will require only a fraction of the memory you need for the 'monolithic' job. While I'm writing these lines, I ran a monolithic build of 'envutil' on an old mac mini, and after two hours it still was not ready! CPU load was right down, because the system is busy swapping to the old HDD. I stopped it and did a 'split' build instead which terminated after a few minutes. You may be able to split your large block of code into several smaller ones and produce several multi-SIMD-ISA object files, then link them together. But this can be quite hard to do - and in my opinion it's not the most 'natural' way of 'splitting' the code: if you only produce one object file per ISA, chances are you don't need to change your code at all, apart from a bit of reshuffling and scaffolding, and on platforms with many prospective ISAs (x86 currently has seven) you can reduce the scale of the compile jobs to a corresponding fraction.

The next issue with 'monolithic' builds is a bit harder to explain. Consider the following program:

### template_problem.cc

    #ifndef SENTINEL
    #define SENTINEL

    template < typename T >
    T tsum ( T a , T b )
    {
      return a + b ;
    }

    #endif // SENTINEL

    #undef HWY_TARGET_INCLUDE
    #define HWY_TARGET_INCLUDE "template_problem.cc" 

    #include <hwy/foreach_target.h> 
    #include <hwy/highway.h>

    HWY_BEFORE_NAMESPACE() ;

    namespace project
    {
    namespace HWY_NAMESPACE
    {
      namespace hn = hwy::HWY_NAMESPACE ;
      typedef hn::ScalableTag < float > D ;
      typedef typename hn::Vec < D > vec_t ;

      vec_t add ( vec_t a , vec_t b )
      {
        return tsum ( a , b ) ;
      }
    } ;
    } ;

    HWY_AFTER_NAMESPACE() ;

This is a very simple test program using highway's foreach_target mechanism - or, rather, trying to do so - but failing (you need to compile with at least -O1)! What is wrong with this program? From the code at hand, the problem is not really obvious, but it has to do with the way how vec_t is coded in highway. It sports operator+ which is labeled 'HWY_INLINE', meaning that it should be inlined when used in a template. Here, we try and inline it into the 'tsum' template, but the compiler refuses to do so:

    $ clang++ -O1 template_problem.cc -I.
    template_problem.cc:7:12: error: always_inline function 'operator+' requires target feature 'ssse3', but would be inlined into function 'tsum' that is compiled without support for 'ssse3'
        7 |   return a + b ;
          |            ^
    template_problem.cc:7:12: error: always_inline function 'operator+' requires target feature 'ssse3', but would be inlined into function 'tsum' that is compiled without support for 'ssse3'
    ...

If we want to use the 'tsum' template with the 'add' function, we have to move it into the nested namespace, which we can't do if the template is defined in a header provided by an external library. Now you might argue that external libraries won't declare their operator functions with HWY_INLINE, but the problem goes deeper: Templates defined outside the nested namespace won't inline *anything* from inside the nested namespace. even if the compilation succeeds, the inlining won't. If inlining fails, this is not an error unless it was declared to be mandatory - but the binary will now issue a function call to the code which we would have liked to be inlined, and this 'throws a spanner in the works' of optimization. The result is code which is slower than an inlined version would be.

If you're reading this text, you probably don't want anything to slow down your binary. So what can you do? The answer is the same as the one to the first point I've made above: produce separate ISA-specific object files. This does the trick if you #include the foreign headers (the ones with the templates) *after* HWY_BEFORE_NAMESPACE(). Since ISA-specific objects are created without foreach_target.h, the compiler only 'sees' the header once, and it 'sees' it inside the #pragma-defined compilation environment, if we've set everything up so that highway's target specification is affected properly. Now, the template is in the same 'headspace' as the code in the nested namespace, inlining works, and the optimizer can do it's job properly. Of course, there is no free lunch: you pay with larger binaries, because the code for the template with inlined parts will be in every one of the ISA-specific object files, whereas before you had the template's 'own' code only once, and then function calls into what should have been inlined.

I have stated previously that handling dispatch to separate ISA-specific object files is laborious, and I've pointed to my image viewer, lux, to show just how laborious it is. This is true if one does it 'all on foot' without using helpful code to make the dispatching easier. If you do rely on highway to help with the dispatching, the job becomes easier - and if you use cmake as your build system, you can even automate it down to a point where you may occasionally have to update your build script to accomodate ISAs which highway adds over time. I'll start out presenting example code for x86 CPUs for compilation on the command line, and then I'll move on to using cmake code. You'll see that, exploiting highway's 'wisdom', the job isn't as daunting as what I did in lux, where I also cater for other SIMD back-ends which don't have anything to do with highway and need a lot of 'manual' code to handle the dispatch. Here we go with the example code!

### driver.cc

    // Again, we use a separate 'driver ' program, where we put 'main'.
    // The definition of class dispatch_base is now in a separate header:
    
    #include "dispatch.h"
    
    int main ( int argc , const char ** argv )
    {
      // obtain a dispatch_base pointer to the ISA-specific code.
      // get_dispatch is in dispatch.cc

      auto dp = project::get_dispatch() ;

      // call the payload function via the dispatch_base pointer

      int success = dp->payload ( argc , argv ) ;

      return success ;
    }

Neat, isn't it? We don't need any references to highway here - it's all plain old C++ code, in whatever ISA the compiler uses if you don't tell it to use a specific one. Since this part is not performance-critical, that's just fine.

### dispatch.h

    // dispatch.h has the class definition of dispatch_base and the
    // declaration of 'get_dispatch' which yields the dispatch_base
    // pointer at run-time.

    #ifndef DISPATCH_BASE
    #define DISPATCH_BASE

    struct dispatch_base
    {
      virtual int payload ( int argc , char * argv[] ) const = 0 ;
    } ;

    // get_dispatch will yield a dispatch_base pointer to the ISA-specific
    // payload code best suited for the CPU currently running the code.
    // The definition of this function is in 'dispatch.cc'.

    namespace project
    {
      extern const dispatch_base * const get_dispatch() ;
    } ;

    #endif // for #ifndef DISPATCH_BASE
    
Still no surprises, still no highway code. This changes now, and we'll use highway's foreach_target.h - but now we'll only use it to provide the 'conduit' to payload code in separate, ISA-specific TUs, which we'll come to with the next chapter. The code here is familiar - the difference is that we're not defining 'payload' in this file. We needn't even declare it: in dispatch.h we have declared a pure virtual payload function, so the linker will look for a definition for every concrete dispatch instantiation. We only need to supply an appropriate implementation in some other object file at link time. If there is no implementation, this is an error.

### dispatch.cc

    // dispatch.h declares class dispatch_base

    #include "dispatch.h"

    // we use foreach_target with the usual scaffolding code

    #undef HWY_TARGET_INCLUDE
    #define HWY_TARGET_INCLUDE "dispatch.cc" 

    #include <hwy/foreach_target.h> 
    #include <hwy/highway.h>

    HWY_BEFORE_NAMESPACE() ;

    namespace project
    {
      namespace HWY_NAMESPACE
      {
        // here, we're inside the ISA-specific nested namespace. We declare
        // _get_dispatch. The function definition for that resides in the
        // ISA-specific TU.  

        const dispatch_base * const _get_dispatch() ;
      } ;

    #if HWY_ONCE

      // we use highway's HWY_DYNAMIC_DISPATCH to automatically pick the
      // best version of _get_dispatch, which will return a dispatch_base
      // pointer pointing to an ISA-specfic dispatch object. We'll call
      // get_dispatch from the main program.

      HWY_EXPORT ( _get_dispatch ) ;

      const dispatch_base * const get_dispatch()
      {
        return HWY_DYNAMIC_DISPATCH(_get_dispatch) () ;
      }

    #endif

    } ;

    HWY_AFTER_NAMESPACE() ;

Finally we need to provide our payload code. We use *one file for all ISAs*, but we compile it several times, each time defining a preprocessor variable to select highway's target architecture for the ISA-specific compilation. This is the payload file:

### payload.cc

    #include <iostream>
    #include "dispatch.h"

    // This header defines all the macros having to do with targets:

    #include <hwy/detect_targets.h>

    // glean the target as 'TG_ISA' from outside - this file is intended
    // to produce ISA-specific separate TUs containing only binary for
    // one given ISA, but assuming that other files of similar structure,
    // but for different ISAs will also be made and all linked together
    // with more code which actually makes use of the single-ISA TUs.
    // 'Slotting in' the target ISA from the build system is enough to
    // produce a SIMD-ISA-specific TU - all the needed specifics are
    // derived from this single information. detect_targets.h sets
    // HWY_TARGET to HWY_STATIC_TARGET, so we #undef it and use the
    // target specification from outside instead.

    #undef HWY_TARGET
    #define HWY_TARGET TG_ISA

    // now we #include highway.h - as we would do after foreach_target.h
    // in a multi-ISA build. With foreach_target.h, the code is re-included
    // several times, each time with a different ISA. Here we have set one
    // specific ISA and there won't be any re-includes.

    #include <hwy/highway.h>

    HWY_BEFORE_NAMESPACE() ;

    // now we're inside the #pragma-defined ISA-specific compilation
    // environment - just as if we'd passed ISA-specific compiler flags.
    // We could now #include headers, or define templates, which would
    // all 'fully' cooperate with the payload code, because they would
    // share the same environment.

    namespace project
    {
      namespace HWY_NAMESPACE
      {
        // now we finally define dispatch::payload and _get_dispatch.
        // The definitions are happening in the ISA-specific environment
        // (set up with HWY_BEFORE_NAMESPACE), wich is what we need, once
        // for each in-play ISA.

        struct dispatch
        : public dispatch_base
          {
            int payload ( int argc , char * argv[] ) const
            {
              // finally, the 'payload code' itself. Just to show that
              // all our efforts have put us in the right environment,
              // we echo the name of the current target ISA:

              std::cout << "paylod: target = "
                        << hwy::TargetName ( HWY_TARGET )
                        << std::endl ;
              return 0 ;
            }
          } ;

        // _get_dispatch holds a static object of the derived class
        // 'dispatch' and returnd a base class pointer to it.

        const dispatch_base * const _get_dispatch()
        {
          static dispatch d ;
          return &d ;
        }
      } ;
    } ;

    HWY_AFTER_NAMESPACE() ;

### Putting It All Together

How do we know which ISAs highway will produce code for? highway
provides a small utility 'hwy_list_targets', which emits just the
information which is needed here. On my system, this utility produces
this output:

    $ ./hwy_list_targets
    Config: emu128:0 scalar:0 static:0 all_attain:0 is_test:0
    Compiled HWY_TARGETS:   AVX3_SPR AVX3_ZEN4 AVX3 AVX2 SSE4 SSSE3 SSE2
    HWY_ATTAINABLE_TARGETS: AVX3_SPR AVX3_ZEN4 AVX3 AVX2 SSE4 SSSE3 SSE2 EMU128
    HWY_BASELINE_TARGETS:   SSE2 EMU128
    HWY_STATIC_TARGET:      SSE2
    HWY_BROKEN_TARGETS:    
    HWY_DISABLED_TARGETS:  
    Current CPU supports:   AVX2 SSE4 SSSE3 SSE2 EMU128 SCALAR

The list of ISA names we need is the first one, 'Compiled HWY_TARGETS'

Let's start with a simple way of compiling the code. We need to feed the ISA-specific compilations of payload.cc with the information about the highway target. Let's assume you're coding on/for an x86 environment. Here's a shell script to produce several payload_XXX.o files, covering all ISAs:

    for TARGET in SSE2 SSSE3 SSE4 AVX2 AVX3 AVX3_ZEN4 AVX3_SPR
    do
      echo $TARGET
      clang++ -O3 -c -o payload_$TARGET.o payload.cc
    done

Next we want to add the other components and form the final binary:

    clang++ driver.cc dispatch.cc -I. -O3 -o multi_tu payload_*.o -lhwy

Let's try it out. Here I get:

    $ ./multi_tu
    paylod: target = AVX2

Bingo! It works - but what's slightly annoying is the fact that we need to know the set of targets for the CPU family we're currently working on, and we need to keep it up-to-date, when highway releases code for new targets on that CPU line. This is the extra bit of maintenance we can't get around - we can delegate it to a build script, but then we need to keep the build script up-to-date - or we need a way to have it updated for us. The latter would be preferable. Now let's employ CMake for the task instead of using a shell script. CMake can produce 'object libraries', and we'll use this feature to affect the building of several ISA-specific object files from a single source. Here's the CMake code:

### CMakeLists.txt

    # cmake script to make the executable 'multi_tu' from the sources
    # in this directory. To build, do this:
    #   mkdir build
    #   cd build
    #   cmake ..
    #   make

    cmake_minimum_required(VERSION 3.31)
    project ( multi_tu )

    # we first need to figure out 'where we are' CPU-wise. This section
    # is incomplete, but it should work on x86 and ARM targets.

    if (     ${CMAKE_SYSTEM_PROCESSOR} STREQUAL x86_64
          OR ${CMAKE_SYSTEM_PROCESSOR} STREQUAL AMD64 )
      set(x86_64 TRUE)
      message(STATUS "***** setting i86 TRUE for an intel/AMD target")
    else()
      set(x86_64 FALSE)
      message(STATUS "***** setting i86 FALSE; not an intel/AMD target")
      if ( ${CMAKE_SYSTEM_PROCESSOR} STREQUAL arm64 )
        message(STATUS "***** setting arm64 TRUE for an ARM target")
        set ( arm64 TRUE )
      else()
        message(STATUS "***** setting arm64 FALSE, not an ARM target")
        set ( arm64 FALSE )
        if ( ${CMAKE_SYSTEM_PROCESSOR} STREQUAL aarch64 )
          message(STATUS "***** setting aarch64 TRUE for an SVE target")
          set ( aarch64 TRUE )
        else()
          message(STATUS "***** setting aarch64 FALSE, not an SVE target")
          set ( aarch64 FALSE )
        endif()
      endif()
    endif()

    # assuming we have correctly detected the architecture, we list ISAs
    # which can occur on that architecture. This is the section we need
    # to maintain and update when highway provides new targets. If
    # highway were to cooperate, it might provide this section as part
    # of the cmake code it deploys, which would make the process
    # automatic. Alternatively, we might use an invocation of the utility
    # 'hwy_list_targets' and extract the list from it's output.

    if ( x86_64 )
      list ( APPEND isa_l SSE2 SSSE3 SSE4 AVX2 AVX3 AVX3_ZEN4 AVX3_SPR )
    # the NEON_BF16 ISA seems not to be built by default. We've used this
    # code to build envutil on a macBook pro with M1 processor:
    elseif ( arm64 )
      list ( APPEND isa_l NEON_WITHOUT_AES NEON ) # NEON_BF16
    # tentative, we have no SVE systems
    elseif ( aarch64 )
      list ( APPEND isa_l ALL_SVE SVE SVE2 SVE_256 SVE2_128 )
    else()
      # tentative. on my system, this doesn't work.
      list ( APPEND isa_l EMU128 )
    endif()

    # for the test program composed of several ISA-specific TUs,
    # we have the main program disp_to_tu.cc, which does the
    # dispatching, and basic.cc, which has ISA-independent code

    add_executable(multi_tu driver.cc dispatch.cc)

    # we need to link with libhwy

    target_link_libraries(multi_tu hwy)

    # the main program needs specific compile options:

    set_source_files_properties ( driver.cc PROPERTIES COMPILE_FLAGS "-O3" )
    set_source_files_properties ( dispatch.cc PROPERTIES COMPILE_FLAGS "-O3 -I.." )

    # for the ISA-specific object files holding 'payload' code,
    # we use cmake 'object libraries'. This places the ISA-specific
    # object files in separate directories, for which we use the
    # same name as the ISA. For this program, each of the object
    # libraries will only contain a single object file made from
    # inset.cc with ISA-specific compilation instructions. Since
    # we're already running a loop over the ISAs, we add a line
    # to tell cmake to link the object file in.

    foreach ( isa IN LISTS isa_l )

        add_library ( ${isa} OBJECT payload.cc )

        target_compile_options ( ${isa} PUBLIC -DTG_ISA=HWY_${isa} -O3 )

        target_link_libraries(multi_tu $<TARGET_OBJECTS:${isa}>)

    endforeach()

Quite a mouthful, not quite complete, but - AFAICT - usable on x86 and ARM. The critical bits are CPU family detection and the choice of ISA list for the detected CPU family. The driver, dispatch and payload files may be named differently in an actual project, but it should be clear where they should go in the CMake code. Here's what I get on my system:

    $ cmake ..
    -- The C compiler identification is GNU 14.2.0
    -- The CXX compiler identification is GNU 14.2.0
    -- Detecting C compiler ABI info
    -- Detecting C compiler ABI info - done
    -- Check for working C compiler: /usr/bin/cc - skipped
    -- Detecting C compile features
    -- Detecting C compile features - done
    -- Detecting CXX compiler ABI info
    -- Detecting CXX compiler ABI info - done
    -- Check for working CXX compiler: /usr/bin/c++ - skipped
    -- Detecting CXX compile features
    -- Detecting CXX compile features - done
    -- ***** setting i86 TRUE for an intel/AMD target
    -- Configuring done (0.3s)
    -- Generating done (0.0s)
    -- Build files have been written to: <snip>/multi_tu_example/build
    $ make
    [ 10%] Building CXX object CMakeFiles/AVX3.dir/payload.cc.o
    [ 10%] Built target AVX3
    [ 20%] Building CXX object CMakeFiles/SSE2.dir/payload.cc.o
    [ 20%] Built target SSE2
    [ 30%] Building CXX object CMakeFiles/SSSE3.dir/payload.cc.o
    [ 30%] Built target SSSE3
    [ 40%] Building CXX object CMakeFiles/AVX3_ZEN4.dir/payload.cc.o
    [ 40%] Built target AVX3_ZEN4
    [ 50%] Building CXX object CMakeFiles/AVX3_SPR.dir/payload.cc.o
    [ 50%] Built target AVX3_SPR
    [ 60%] Building CXX object CMakeFiles/SSE4.dir/payload.cc.o
    [ 60%] Built target SSE4
    [ 70%] Building CXX object CMakeFiles/AVX2.dir/payload.cc.o
    [ 70%] Built target AVX2
    [ 80%] Building CXX object CMakeFiles/multi_tu.dir/driver.cc.o
    [ 90%] Building CXX object CMakeFiles/multi_tu.dir/dispatch.cc.o
    [100%] Linking CXX executable multi_tu
    [100%] Built target multi_tu
    $ ./multi_tu
    paylod: target = AVX2

You see how the loop over the ISAs listed for the given CPU family produces separate 'object libraries' - each in their own directory and containing precisely one object each: the ISA-specific object for the given ISA. Since these objects are all added with the 'target_link_libraries' command in the same loop, they are automatically linked to the other TUs, and we get the desired result: the binary 'multi_tu' which dispatches to the ISA-specific 'payload' function, which, in turn, echos the ISA for which it was made. We use highway's HWY_DYNAMIC_DISPATCH to pick the right version, and we're using highway's capability of telling the compiler with #pragmas how to build the ISA-specific code, so we've delegated as much as possible to highway. As I mentioned before, we're left to maintain the build script, as long as there is no way to have it generated or obtain it from elsewhere.

I have elaborated the code in this repository and added a few extras:

- adding 'metadata' to the dispatch objects
- adding highway-managed objects which foreach_target doesn't usually produce
- adding 'external' objects which may or may not use highway
- putting payload code in a shared library to be used as s plugin

You may want to read through the sources to see how it all fits together, I've added ample comments which should help you along.
