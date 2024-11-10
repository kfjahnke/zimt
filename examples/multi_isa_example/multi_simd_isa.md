# The Multi-SIMD-ISA Dilemma

For many years - even decades - CPUs have been supplied with SIMD units. What started out as the MMX instruction set in what now feels like the Dark Ages has involved into highly sophisticated and capable SIMD units like AVX2, which are now available on most x86 CPUs still in use. But the 'better' SIMD ISAs - specifically AVX and better on x86 - are rarely fully exploited, as can be seen by scanning machine code for assembly instructions which are typical for these more advanced SIMD ISAs. This observation is puzzling at first - if the more evolved SIMD units are better than their predecessors, one would expect that they are widely adopted! The answer is simple: inertia, ignorance and tricky access.

## ... but it works!

Let's say you have a program which is computationally expensive - something like an image processor. With every new CPU, you will notice performance gains, and you suspect that these gains are due to the new CPU's better architecture. This is, in part, true - but if your software is still the same, chances are that it will only exploit the CPU's features which were available at the program's inception, or use features which represent some 'lowest common denominator' shared by all CPUs not deemed totally obsolete. So what you do is in fact 'strangle' the beautiful new CPU and force it to execute machine code from a few generations back, leaving it's shiny new advanced SIMD units unused. The code will work, and due to overall improvements in CPU speed and architecture it will run faster, but you're not reaping all the benefits you might - if you could exploit the new SIMD ISA. The program is still faster than on the previous box: 'it works'. You get small improvements and are content, but this may be much less than your new CPU has to offer.

## Let's Try the New ISA

To use a new SIMD ISA, you have to *tell your compiler* - By default, the compiler will pick the lowest-common-denominator SIMD ISA deemed the acceptable minimum, which only fails to run on *very* old CPUs. On x86, this is usually some level of SSE. If you want anything beyond that, you need specific compiler flags, like -mavx2 on gnu and clang compilers. The resulting binary may now contain instructions specific to the chosen SIMD ISA. If your machine actually has this specific SIMD unit, the code will run and should be faster - provided you have optimization on which allows *autovectorization* (usually -O2 does the trick) and there are inner loops which can be autovectorized. But what if the CPU doesn't support the new SIMD ISA? Then your program will terminate with an *Illegal Instruction* error. Obviously, you can't ship code like this without safeguards, so let's think of ways how to deal with this issue.

## Dealing with several SIMD ISAs

The first - and simplest - approach is to compile several 'incarnations' of your program, each targeting a specific SIMD ISA, and to use some mechanism to pick the right one for a given target machine. You might ship a bundle holding all versions, then use a small dispatching program on-site to pick the right one. Or you deploy to a specific machine, where you know the right version beforehand. Obviously this is cumbersome and error-prone - what happens, for example, if the target machine receives a CPU upgrade and you deployed code for the initial CPU? And if you use a dispatcher program, you have to maintain it and make sure the user doesn't execute an incarnation which isn't suitable.

So instead you might look for a way to create a 'monolithic' binary which contains all the variants and dispatches internally. It turns out that this is suprisingly difficult. I devised a scheme to achieve this, compiling ISA-specific code into ISA-specific TUs and dispatched via the VFT of a pure virtual base class, overriding the virtual member functions in the ISA-specific implementations in each ISA-specific TU. This works and is a reasonably general approach, but it's also cumbersome and requires a fair amount of compiler and build system artistry.

What I really wanted was a method to achieve the dispatch without massive intrusion into my coding, and without having to 'manually' decide which SIMD ISAs to address and how to 'tell the compiler' about it. As it turns out, there is a ready-made solution which does just that - it's part of the highway library!

## How Does highway Do It?

highway's internal dispatching code is much more sophisticated than one might see at a brief glance. There are several components which work hand in hand, and each of them saves a lot of work. This puts it beyond most SIMD library approaches, which simply offer to create a specific bit of binary for a given target ISA.´

The first step is CPU detection. While the CPU family (like x86, ARM, RISC-V) is detected and fixed at compile time, highway can figure out *at runtime* which specific SIMD ISA is available! This is the foundation of successful dispatch, and highway does it reliably.

The second step is the capability to make the compiler emit machine code for any given ISA. While the previously described approach works 'from the outside' by using specific compiler flags, this can also be achieved 'from the inside' by using - typically compiler-specific - code which is made up of instructions outside the normal C++ standard and passed as, e.g., pragmas. This obviously requires first detecting which compiler is currently being used and then intimate knowledge of what the given compiler 'understands'. highway does both.

With this capability of generating ISA-specific code 'from the inside', it becomes possible to generate a monolithic TU containing code for several SIMD ISAs, rather than having to rely on the linker to combine ISA-specific object files. And with the possibilty of monolithic object files the dispatch can now occur *inside a single TU*. The final component highway provides to solve the Multi-SIMD-ISA dilemma is to offer a well-thought-out dispatching mechanism, placing each ISA-specific variant in a separate nested namespace and offering code to dispatch to the variant which is best on the CPU on which the code is executed! All it takes is to understand the mechanism, and understanding it isn't *very* hard - what I found missing is simply an introductory text which is less terse than the highway documentation. I'll try and provide just that in the next chapter.

# highway's Dispatch Mechanism

I've already hinted at the way how highway provides access to the ISA-specific code: via nested namespaces. This sounds scary if you don't usually use namespaces much, but it's really quite simple once you get the hang of it. Let's first start out without ISA-specific code, but set up namespaces which we'll later populate with the nested namespaces. We'll start up with highway code in a namespace named 'hwy' (that's where it usually lives) and our own code in a namespace names 'project'. When calling into highway code from our code, we'd *qualify* the access with a hwy:: prefix. This is simple and straightforward - and good practise anyway, keeping our own code in a separate namespace and qualifying use of code from other components with their namespace prefix (or using a 'using' directive to the same effect, which can become obscure).

If we want to have several variants of a body code, we can enclose these variants each into a separate scope. Then we can use the same set of symbols in each of those scopes, and we can introduce a selector to address the symbol set in a specific scope: We use indirection. One way of setting up a scope is a namespace: it's simply a scope with a name. Another way is using classes. Class members share the same scope, and they are distinct from members with the same name in a different scope.

highway uses namespaces. To be more specific, it uses nested namespaces: The ISA-specific code versions 'live' in namespaces 'one level down' from the 'hwy' namespace. highway's 'own' ISA-specific code 'lives' in hwy::N_SSE2, hwy::N_AVX2 etc. - and we'll follow the same scheme for our own code in namespace 'project' and put SIMD_ISA-specific code inside project::N_SSE2, project::N_AVX2 etc.

You can interface with these specific nested namespaces if you must - by qualifying access with the nested namespace's name - but what's much more attractive is to *write code which does not use these nested namespace explicitly*. highway's foreach_target mechanism - which we'll come to shortly - provides 'client code' with a macro: HWY_NAMESPACE. The value of this macro is set to one of the SIMD architectures available *on the CPU running the code*, e.g. N_AVX2. Instead of qualifying your code to access symbols in e.g. hwy::N_AVX2, you code access to hwy::HWY_NAMESPACE, and your code becomes *generalized* to call into any given variant, while you can *delegate* which one is picked to a *dispatch* mechanism, which highway also provides.

Dispatch is, again, coded with macros: HWY_EXPORT and HWY_DYNAMIC_DISPATCH. You write functions which are 'fed' to highway's foreach_target mechanism, HWY_EXPORT 'picks up' all variants highway deems suitable for the client CPU architecture (x86, ARM, RISC...) and holds them under a common handle, and HWY_DYNAMIC_DISPATCH picks the variant which highway deems most suitable at the time when the dispatch is invoked at run-time. All it takes to use the mechanism is placing the 'workhorse' code into a bit of 'scaffolding' code which submits it to highways dispatching mechanism.

## Getting our hands dirty

We'll now code a simple program which uses highway's dispatching mechanism - the bare minimum to see it happen. I'll provide ample comments throughout - I think repetition is a good didactic tool, so please bear with my verbose repetitive style. I'll switch to code space for now and carry on inside the comments. We'll use two source files, 'driver.cc' and 'payload.cc'.

### driver.cc

    namespace project
    {
      // we expect to call a function named 'payload' living in the 'project'
      // namespace, so we declare it here - the definition is in 'payload'cc'.
      
      void payload() ;
    }

    int main ( int argc , char * argv[] )
    {
      // let's call the payload function
      
      payload() ;
    }

    // that's it for now!

### payload.cc

    // we want to echo stuff to the console

    #include <iostream>

    // we start out with a bit of 'scaffolding' code to use highway's 
    // foreach_target mechanism

    // 'clear' HWY_TARGET_INCLUDE if it's set already

    #undef HWY_TARGET_INCLUDE

    // then set it to the name of this very file

    #define HWY_TARGET_INCLUDE "payload.cc"

    // the compiler now 'knows' which file we want to 'submit' to the
    // foreach_target mechanism and we can #include the 'magic header'

    #include <hwy/foreach_target.h>

    // next we include highway.h, which gives us access to highway
    // library code.

    #include <hwy/highway.h>

    // that's it for scaffolding. The foreach_target mechanism works by
    // repeatedly #including HWY_TARGET_INCLUDE, each time using different
    // compiler settings, specific to each supported SIMD ISA. It sets the
    // macro 'HWY_NAMESPACE', so we now switch to the SIMD-ISA-specific
    // code by continuing in it's own specific nested namespace:

    namespace project
    {
      namespace HWY_NAMESPACE
      {
        // here we are inside the SIMD-ISA-specific namespace, but we
        // dont' want to know *which specific one* that might be - we
        // keep the code general.
        // Let's, finally, write the 'workhorse code' - we'll prefix it's
        // name with an underscore - as befits a symbol with internal use.
        // Code which is meant to use highway's SIMD capabilities has to
        // be prefixed with HWY_ATTR or have the code between
        // `HWY_BEFORE_NAMESPACE()` and `HWY_AFTER_NAMESPACE()`. Here
        // we add HWY_ATTR.
        
        HWY_ATTR void _payload()
        {
          // let's do something ISA-specific, using more utility macros.
          // Here we obtain the specific SIMD ISA's name in printable form
          // and echo it to std::cout
          
          std::cout << "paylod: target = "
                    << hwy::TargetName ( HWY_TARGET )
                    << std::endl ;
        }
      } ; // end of nested namespace HWY_NAMESPACE

      #if HWY_ONCE

      // now we tell highway that we will want to dispatch to the SIMD-ISA-
      // specific variants of _payload. This is only done once for the
      // entire program, hence the test for HWY_ONCE above.

      HWY_EXPORT ( _payload ) ;

      // here comes the definition of the function 'payload' 'one level up',
      // so in namespace 'project' without further qualifiers. This is meant
      // 'for export', just as it's declared and expected in 'driver.cc'.
      
      void payload()
      {
        // The 'outer' payload routine relies on HWY_DYNAMIC_DISPATCH to call
        // the version of _payload which is best for the currently used CPU:

        return HWY_DYNAMIC_DISPATCH(_payload) () ;
      }

      #endif
    }

    // and that's us done!

## Compile it

All we need now is to compile and link the program and try out what we get. Let's assume you're using g++. You can use this one-liner - note the -I.
statement, which is needed because highway needs to#include 'payload.cc' several times and it's located inside this very folder, or '.' for short.

    g++ driver.cc payload.cc -I. -lhwy

By convention, the compiler will emit a binary named a.out, let's call it

    ./a.out

On my system I get this output:

    payload: target = AVX2

If you were to run a.out on another system, you might get different output, depending on the CPU's best supported SIMD ISA.

## Bingo!

We can now dispatch to SIMD-ISA-specific code without having written a single line of SIMD_ISA-specific code!

You may argue that we need a fair bit of scaffolding for something which seems trivial, but think twice: highway analyzes *at run-time* which CPU your program runs on. It has already stashed binary variants for each SIMD ISA which might possibly occur, and now it picks the appropriate variant and calls into it via it's dispatch mechanism. That is highly sophisticated and involved code, of which you can remain blissfully unware! And inside the scaffolding can go many more functions which are set up the same way, so you don't have to 'erect' scaffolding for each function. What you do have to do at this level is repeat the 'HWY_EXPORT' invocation to register your _payload function with highway, and provide the implementation of 'payload' (the externally visible one) which invokes HWY_DYNAMIC_DISPATCH.

## Artistry

What we have, so far, is a way to provide our program with free functions in the 'project' namespace which, in turn, invoke SIMD-ISA-specific variants. If you only have a few functions using SIMD, that's okay, but if your program grows to use more of these functions, coding becomes somewhat repetitive. Can we do better? I've come up with a scheme to 'bundle' payload functions using a dispatch class. Let's start with the gist of it and elaborate later. We add a new (base) class 'dispatch_base', where we introduce declarations of our payload functions *as pure virtual member functions*. Let's go step by step. The first thing we do is to factor out the declaration for namespace 'project' into a separate header file 'project.h' and add the definition of the 'dispatch_base' base class.

### project.h

    // We use a 'sentinel' to make sure a TU 'sees' this code precisely once

    #ifndef PROJECT_H
    #define PROJECT_H

    namespace project
    {
      // We want a 'dispatch object' with - for now - one member function.
      // This is the 'payload' function we had as a free function - now
      // it's a pure virtual member function of class dispatch_base.

      struct dispatch_base
      {
        // we declare 'payload' as a *pure virtual member function'. That
        // makes it impossible to invoke the 'payload' member of struct
        // dispatch itself (because there is none) - only derived classes
        // with an actual implementation can be used to invoke their
        // specific implementation.

        virtual void payload() = 0 ;
      } ;

      // We have one free function: get_dispatch will yield a dispatch_base
      // pointer. This is our 'conduit' to the derived dispatch classes and
      // makes use of the C++ virtual member function mechanism: even though
      // we'll invoke it via a pointer to dispatch_base, the function in
      // the derived class will be invoked.

      dispatch_base * get_dispatch() ;
    } ;

    #endif // to #ifndef PROJECT_H


### payload.cc with 'VFT dispatch'

    // we want to echo stuff to the console

    #include <iostream>

    // here we include the new header 'project.h'

    #include "project.h"

    // we start out with a bit of 'scaffolding' code to use highway's 
    // foreach_target mechanism

    // 'clear' HWY_TARGET_INCLUDE if it's set already

    #undef HWY_TARGET_INCLUDE

    // then set it to the name of this very file

    #define HWY_TARGET_INCLUDE "payload.cc"

    // the compiler now 'knows' which file we want to 'submit' to the
    // foreach_target mechanism and we can #include the 'magic header'

    #include <hwy/foreach_target.h>

    // next we include highway.h, which gives us access to highway
    // library code.

    #include <hwy/highway.h>

    // that's it for scaffolding. The foreach_target mechanism works by
    // repeatedly #including HWY_TARGET_INCLUDE, each time using different
    // compiler settings, specific to each supported SIMD ISA. It sets the
    // macro 'HWY_NAMESPACE', so we now switch to the SIMD-ISA-specific
    // code by continuing in it's own specific nested namespace:

    namespace project
    {
      namespace HWY_NAMESPACE
      {
        // here we are inside the SIMD-ISA-specific namespace, but we
        // dont' want to know *which specific one* that might be - we
        // keep the code general.

        // Let's define a 'dispatch' class from the 'dispatch_base' base
        // class we have in 'project.h'. It's important that the derived
        // class is in the SIMD-ISA-specific namespace!

        struct dispatch
        : public project::dispatch_base
        {
          // here, we provide an override of the pure virtual member
          // function in the base class with a concrete implementation.
          // We use the same 'payload' body, but now it's a member
          // function of class dispatch and not a free function as before.
          // Again we prefix the function declaration with HWY_ATTR.

          HWY_ATTR virtual void payload()
          {
            // let's do something ISA-specific
          
            std::cout << "payload: target = "
                      << hwy::TargetName ( HWY_TARGET )
                      << std::endl ;
          }
        } ;

        // We still use highway's dispatch mechanism - but now we don't
        // HWY_EXPORT the payload function any more - we'll access it via
        // the dispatch class. The only function we submit to HWY_EXPORT
        // for now is the _get_disptach function. This function returns
        // a pointer to an object of class project::dispatch_base - the
        // base class. But the return statement inside the function takes
        // a dispatch object local to the current nested namespace. C++
        // semantics automatically casts that to the desired base class
        // pointer, which is returned to the caller.
        
        project::dispatch_base * _get_dispatch()
        {
          static dispatch d ;
          return &d ;
        }
      } ;
        
      #if HWY_ONCE

      // now we tell highway that we will want to dispatch to the SIMD-ISA-
      // specific variant of _get_dispatch

      HWY_EXPORT ( _get_dispatch ) ;

      // here comes the definition of the function 'get_dispatch', now 'one
      // level up', in namespace 'project' without further qualifiers. This 
      // is meant 'for export', just as it's declared and expected in
      // 'driver.cc'. Driver.cc can now acquire a dispatch_base pointer,
      // and calling the virtual member function through it will invoke the
      // SIMD-ISA-specific variant.
      
      project::dispatch_base * get_dispatch()
      {
        return HWY_DYNAMIC_DISPATCH(_get_dispatch) () ;
      }

      #endif // #if HWY_ONCE
    }


### driver.cc with 'VFT dispatch'

    #include "project.h"

    int main ( int argc , char * argv[] )
    {
      // Now we'll not call the payload function, but get_dispatch
      
      auto * dp = project::get_dispatch() ;

      // If we now call 'payload' via dp we get the ISA-specific version:
      
      dp->payload() ;
    }

## Now what?

What did we gain? We now have a 'conduit' into SIMD-ISA-specific code, and we can add more member functions to the dispatch class, building up a base of SIMD-ISA-specific code which can be invoked through the dispatch mechanism. This is pretty much it, but I found one more thing which makes this method better suited to handle larger collections of functions. If you add more and more functions, it may be better to separate the function declarations and the definitions. When we declare the pure virtual member functions in dispatch_base and then the implementations in the derived 'dispatch' class, we need precisely the same argument lists, so we supply them via a macro which we define first to produce pure virtual member function declarations inside class dispatch_base, then we re-define it to produce concrete member declarations. I offer this here as a suggestion, the example code does not use this extra feature.

### modified dispatch_base class in project.h

      struct dispatch_base
      {
        // we define the macro 'SIMD_REGISTER' to generate pure virtual
        // meber function declarations. The invocations of this macro are
        // taken from 'interface.h'.

        #define SIMD_REGISTER(RET,NAME,...) \
          virtual RET NAME ( __VA_ARGS__ ) const = 0 ;

        #include "interface.h"

        #undef SIMD_REGISTER
      } ;

### modified derived dispatch class in payload.cc

        struct dispatch
        : public project::dispatch_base
        {
          // here, we provide an override of the pure virtual member
          // function in the base class with a concrete implementation.
          // We don't provide the implementation now, but only it's
          // declaration - the implementation will be elsewhere.
          // Let's start out with a definition of SIMD_REGISTER:

          #define SIMD_REGISTER(RET,NAME,...) RET NAME ( __VA_ARGS__ ) const ;

          // now we use the macro invocations in 'interface.h' to generate
          // the declarations.

          #include "interface.h"

          #undef SIMD_REGISTER
        } ;

### interface.h

SIMD_REGISTER(void, payload, /* no arguments */ )

## Building a code base

We now have all the scaffolding we need even for a larger project. If we want to add a new function, we only need to edit two locations in the code: We need to add a macro invocation of SIMD_REGISTER inside interface.h, and an implementation of the function inside the SIMD-ISA-specific namespace. So a lot of the code we've evolved is 'boilerplate', and it enables us to add new functions quickly and exploit the dispatch mechanism fully without having to touch the details of how this is done. We have a single conduit to the base of SIMD-ISA-specific code: the dispatch pointer, which we receive by calling get_dispatch.

## Widening the scope

Can we do more useful stuff with the framework we've set up? One thing which springs to mind is to provide more derived 'dispatch' classes from other sources and code to 'slot them in' as alternatives to the ones we have available so far. This is left to the user: it's a simple matter of writing a new version of get_dispatch and implementing the derived dispatch classes. One good use case is setting up the derived dispatch classes in separate TUs which are compiled with different compiler flags or use different back-end libraries. The main program can then dispatch to a whole range of code variants - among them, if desired, the set of variants we've introduced here by exploiting highway's foreach_target mechanism. The dispatch mechanism we have now is a generalization - we can use it quite independently of highway, but then we have to do a lot of work ourselves, rather than exploiting highway's powerful features which make our lives easy. As you see from the code we've evolved here, using highway makes the entire affair straightforward and concise - and if you want to see the amount of work needed to provide CPU detection and SIMD-ISA-specific code in a monolithic binary you can have a look at [lux' CMakeLists.txt](https://bitbucket.org/kfj/pv/src/master/CMakeLists.txt) (look for 'FLAVOUR'), where - as of this writing - the job is handled for several x86 SIMD ISAs by using separate TUs and a lot of cmake scaffolding. 

## Using zimt with MULTI_SIMD_ISA and automatic dispatch

Some zimt headers don't use SIMD, but most do. The ones which do are modified to also cooperate with automatic SIMD ISA dispatch, adding another set of nested namespaces living inside the zimt namespace and following the same scheme as above. You can ignore the extended capabilities of the headers and #include them like 'any old header' - some of the examples do just that, and the binary will reflect whatever compiler flags you have passed to determine the SIMD ISA. To use the extended capabilities, best follow the scheme laid out in examples which do the same - [bsp_eval.cc](https://github.com/kfjahnke/zimt/blob/multi_isa/examples/bsp_eval.cc) is the first one I have on offer. This example 'lumps together' the driver, project.h and payload.cc. Here it is - minus the payload code, which doesn't help in this context:

### bsp_eval.cc

    // This is a test program for zimt's recently acquired b-spline
    // processing capabilites and also serves to measure performance of the
    // b-spline evaluation code with splines of various degrees and boundary
    // conditions and varying SIMD back-ends/ISAs.

    // There are two different ways of compiling it. The first is to create 
    // a single-ISA binary, fixing the SIMD ISA at compile time by passing
    // appropriate flags to the compiler. This will work with all SIMD
    // back-ends - here, I show the compiler invocations for an AVX2 version.
    // Note the -I. directive to tell the compiler to find files to #include
    // in the current folder as well.

    //   clang++ -mavx2 bsp_eval.ccc -O2 -I. -DUSE_HWY -lhwy
    //   clang++ -mavx2 bsp_eval.ccc -O2 -I. -DUSE_VC -lVc
    //   clang++ -mavx2 bsp_eval.ccc -O2 -I. -DUSE_STDSIMD
    //   clang++ -mavx2 bsp_eval.ccc -O2 -I.

    // The second way is to use highway's automatic dispatch to embedded
    // variants of the code running with different ISAs. This requires the
    // definition of MULTI_SIMD_ISA and linkage to libhwy and can only
    // be used for the highway and the 'goading' back-end. Here, no
    // architecture flags are passed to the compiler:

    //   clang++ bsp_eval.ccc -O2 -I. -DMULTI_SIMD_ISA -DUSE_HWY -lhwy
    //   clang++ bsp_eval.ccc -O2 -I. -DMULTI_SIMD_ISA -lhwy

    // binaries made with the second method will dispatch to what is deemd
    // the best SIMD ISA available on the CPU on which the binary is run.
    // Because this is done meticulously by highway's CPU detection code,
    // the binary variant which is picked is usually optimal and may
    // out-perform single-ISA variants with 'manually' supplied ISA flags,
    // if the set of flags isn't optimal as well. The disadvantage of the
    // multi-SIMD-ISA variants is (much) longer compile time and code size.
    // Due to the 'commodification' the source code itself doesn't have
    // to be modified in any way to produce one variant or another.
    // This suggests that during the implementation of a new program a
    // fixed-ISA build can be used to evolve the code with fast turn-around
    // times, adding dispatch capability later on by passing the relevant
    // compiler flags.

    // if the code is compiled to use the Vc or std::simd back-ends, we
    // can't (yet) use highway's foreach_target mechanism, so we #undef
    // MULTI_SIMD_ISA, which is zimt's way of activating that mechanism.

    #if defined MULTI_SIMD_ISA && ( defined USE_VC || defined USE_STDSIMD )
    #warning "un-defining MULTI_SIMD_ISA due to use of Vc or std::simd"
    #undef MULTI_SIMD_ISA
    #endif

    // I'll mark code sections which will differ from one example to the
    // next, prefixing with ////////... and postfixing with //-------...
    // You'll notice that there are only four places where you have to
    // change stuff to set up your own program, and all the additions
    // are simple (except for your 'client code', which may be complex).

    /////////////////////// #include 'regular' headers here:

    #include <iostream>
    #include <iomanip>
    #include <vector>
    #include <random>
    #include <chrono>

    //--------------------------------------------------------------------

    // we define a dispatch base class. All the 'payload' code is called
    // through virtual member functions of this class. In this example,
    // we only have a single payload function. We have to enclose this
    // base class definition in an include guard, because it must not
    // be compiled repeatedly, which happens when highway's foreach_target
    // mechansim is used.

    #ifndef DISPATCH_BASE
    #define DISPATCH_BASE

    struct dispatch_base
    {
      // in dispatch_base and derived classes, we keep two flags.
      // 'backend' holds a value indicating which of zimt's back-end
      // libraries is used. 'hwy_isa' is only set when the highway
      // backend is used and holds highway's HWY_TARGET value for
      // the given nested namespace.

      int backend = -1 ;
      unsigned long hwy_isa = 0 ;

      // next we have pure virtual member function definitions for
      // payload code. In this example, we only have one payload
      // function which calls what would be 'main' in a simple
      // program without multiple SIMD ISAs or SIMD back-ends

      virtual int payload ( int argc , char * argv[] ) const = 0 ;
    } ;

    #endif

    #ifdef MULTI_SIMD_ISA

    // if we're using MULTI_SIMD_ISA, we have to define HWY_TARGET_INCLUDE
    // to tell the foreach_target mechanism which file should be repeatedly
    // re-included and re-copmpiled with SIMD-ISA-specific flags

    #undef HWY_TARGET_INCLUDE

    /////////////// Tell highway which file to submit to foreach_target

    // TODO: can't we somehow use __FILE__ here?

    #define HWY_TARGET_INCLUDE "bsp_eval.cc"  // this very file

    //--------------------------------------------------------------------

    #include <hwy/foreach_target.h>  // must come before highway.h

    #include <hwy/highway.h>

    #endif // #ifdef MULTI_SIMD_ISA

    /////////////////////// #include zimt headers here:

    #include "../zimt/eval.h"

    //--------------------------------------------------------------------

    // to make highway's use of #pragma directives to the compiler
    // effective, we surround the SIMD-ISA-specific code with
    // HWY_BEFORE_NAMESPACE() and HWY_AFTER_NAMESPACE().

    HWY_BEFORE_NAMESPACE() ;

    // this macro puts us into a nested namespace inside namespace 'project'.
    // For single-SIMD-ISA builds, this is conventionally project::zsimd,
    // and for multi-SIMD-ISA builds it is project::HWY_NAMESPACE. The macro
    // is defined in common.h. After the macro invocation, we can access
    // all zimt names with a simple zimt:: prefix - both 'true' zimt names
    // and SIMD-ISA-specific versions living in the nested namespace.

    BEGIN_ZIMT_SIMD_NAMESPACE(project)

    // you can use float, but then can't use very high spline degrees.

    typedef float dtype ;
    typedef zimt::xel_t < dtype , 2 > crd_t ;
    typedef zimt::xel_t < dtype , 3 > px_t ;

    // Here, we define the SIMD-ISA-specific derived 'dispatch' class:

    struct dispatch
    : public dispatch_base
    {
      // We fit the derived dispatch class with a c'tor which fills in
      // information about the nested SIMD ISA we're currently in.

      dispatch()
      {
        backend = int ( zimt::simdized_type<int,4>::backend ) ;
        #if defined USE_HWY || defined MULTI_SIMD_ISA
          hwy_isa = HWY_TARGET ;
        #endif
      }

      // 'payload', the SIMD-ISA-specific overload of dispatch_base's
      // pure virtual member function, now has the code which was in
      // main() when this example was first coded without dispatch.
      // One might be more tight-fisted with which part of the former
      // 'main' should go here and which part should remain in the
      // new 'main', but the little extra code which wouldn't benefit
      // from vectorization doesn't make much of a difference here.
      // Larger projects would have both several payload-type functions
      // and a body of code which is independent of vectorization.

    ///////////////// write a payload function with a 'main' signature
      
      int payload ( int argc , char * argv[] ) const
      {
        // the actual payload code is omitted here!
      }
    } ;

    //--------------------------------------------------------------------

    // we also code a local function _get_dispatch which returns a pointer
    // to 'dispatch_base', which points to an object of the derived class
    // 'dispatch'. This is used with highway's HWY_DYNAMIC_DISPATCH and
    // returns the dispatch pointer for the SIMD ISA which highway deems
    // most appropriate for the CPU on which the code is currently running.

    const dispatch_base * const _get_dispatch()
    {
      static dispatch d ;
      return &d ;
    }

    END_ZIMT_SIMD_NAMESPACE

    HWY_AFTER_NAMESPACE() ;

    // Now for code which isn't SIMD-ISA-specific. ZIMT_ONCE is defined
    // as either HWY_ONCE (if MULTI_SIMD_ISA is #defined) or simply true
    // otherwise - then, there is only one compilation anyway.

    #if ZIMT_ONCE

    namespace project {

    #ifdef MULTI_SIMD_ISA

    // we're using highway's foreach_target mechanism. To get access to the
    // SIMD-ISA-specific variant of _get_dispatch (in project::HWY_NAMESPACE)
    // we use the HWY_EXPORT macro:

    HWY_EXPORT(_get_dispatch);

    // now we can code get_dispatch: it simply uses HWY_DYNAMIC_DISPATCH
    // to pick the SIMD-ISA-specific get_dispatch variant, which in turn
    // yields the desired dispatch_base pointer.

    const dispatch_base * const get_dispatch()
    {
      return HWY_DYNAMIC_DISPATCH(_get_dispatch)() ;
    }

    #else // #ifdef MULTI_SIMD_ISA

    // if we're not using highway's foreach_target mechanism, there is
    // only a single _get_dispatch variant in namespace project::zsimd.
    // So we call that one, to receive the desired dispatch_base pointer.

    const dispatch_base * const get_dispatch()
    {
      return zsimd::_get_dispatch() ;
    }

    #endif // #ifdef MULTI_SIMD_ISA

    }  // namespace project

    int main ( int argc , char * argv[] )
    {
      // Here we use zimt's dispatch mechanism: first, we get a pointer
      // to the dispatcher, then we invoke a member function of the
      // dispatcher. What's the point? We can call a SIMD-ISA-specific
      // bit of code without having to concern ourselves with figuring
      // out which SIMD ISA to use on the current CPU: this happens via
      // highway's dispatch mechanism, or is fixed at compile time, but
      // in any case we receive a dispatch_base pointer routing to the
      // concrete variant. project::get_dispatch might even be coded
      // to provide pointers to dispatch objects in separate TUs, e.g.
      // when these TUs use different back-ends or compiler flags. Here,
      // we can remain unaware of how the concrete dispatch object is
      // set up and the pointer obtained.

      auto dp = project::get_dispatch() ;

      // we can get information about the specific dispatch object:

      std::cout << "obtained dispatch pointer " << dp << std::endl ;
      std::cout << "dispatching to back-end   "
                << zimt::backend_name [ dp->backend ] << std::endl ;
    #if defined USE_HWY || defined MULTI_SIMD_ISA
      std::cout << "dispatch hwy_isa is       "
                << hwy::TargetName ( dp->hwy_isa ) << std::endl ;
    #endif

      // now we call the payload via the dispatch_base pointer.

      int success = dp->payload ( argc , argv ) ;
      std::cout << "payload returned " << success << std::endl ;
    }

    #endif  // ZIMT_ONCE

This example should be easy to adapt to your own needs - the sections where you would have to fill in code of your own are marked and there are only four places - one where you #include non-zimt headers, one where you #include zimt headers, one place where you #define a highway macro to the current filename and finally the place where payload code goes. The set-up is so that the payload code can refer to all zimt symbols with a simple zimt:: qualifier, and this will 'find' 'ordinary' zimt code (like xel_t and view_t, which don't use SIMD) as well as zimt code which uses SIMD, which is the majority. Depending on how you compile the code, you'll either get a single-ISA binary with one of the four SIMD back-ends (goading, highway, Vc or std::simd) using whichever SIMD ISA the compiler flags choose - or a multi-SIMD-ISA binary with automatic dispatch. The latter is only possible with the 'goading' and highway back-ends, and you need to #define MULTI_SIMD_ISA - see a list of possible compilations in the example file near the top.

If you look through the code, you'll see that this is not much more than the simple dispatch example given previously - there is a bit more macro trickery to handle the additional zimt namespace and make it easily accessible. The code is written to accomodate smaller programs where the original 'main' function can easily move to the 'payload' function. Larger programs will typically have a large body of code which is left outside the multi-SIMD-ISA context - stuff like the UI, I/O and the likes - and some code which is time-critical and benefits from SIMD - number crunching like image processing. Large programs also often need several 'payload' type functions. If you want to go down that way, just set up these functions where 'payload' is in this example and add declarations in dispatch_base as well. It should be safe to acquire the dispatch pointer once at program start-up. Change of SIMD ISA during run-time of the program should not occur - if this is at all possible, re-fetch the dispatch pointer before calling any of the payload functions through it.
