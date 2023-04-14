# zimt - C++11 template library to process n-dimensional arrays with multi-threaded SIMD code

Zimt is the German word for cinnamon, and it sounds vaguely like SIMD. It should imply something spicy and yummy, and apart from the silly pun there is no deeper meaning.

The code is 'just forked', there is yet much to be done, but I prefer to get it out early, also to attract others to join in. Because it's in large parts from an extant library which has seen years of development, the code is already at a reasonably good level, but I haven't yet done extensive tests or added examples. The code is heavily commented, though.

The code in this library is based on code from my library [vspline](https://bitbucket.org/kfj/vspline/ "git repository of the vspline library"). I found that the tools I developed there to process n-dimensional arrays with multi-threaded SIMD code would be useful outside of a b-spline library's context, and that I might be better off letting go of the [vigra](http://ukoethe.github.io/vigra/ "The VIGRA Computer Vision Library") library, which I use in vspline for data handling and small aggregates.

I intend to - eventually - pull zimt 'back into' vspline, but for now I work the code as a stand-alone entity to be free to make design changes without having to worry about breaking my extant applications.

## installation

Installation is simple: soft-link the folder containing the zimt headers to a location which the compiler will find, so that includes à la "#include <zimt/zimt.h>" will succeed. On my Linux system, I use:

    sudo ln -s path-to-source/zimt/zimt /usr/local/include

Where "path-to-source" is the parent folder of the clone of the zimt repo. With this 'installation' the examples will compile out-of-the-box. If you don't like linking stuff into your /usr/local, you'll have to tell the compiler where the zimt headers are, e.g. with a -I command line argument.

The top-level header is called "zimt.h", it includes all of zimt.

To help with the examples, the 'examples' folder has a shell script 'examples.sh' which will compile al C++ files you pass to it with clang++ and g++, and with all available SIMD backends. Of course using all available backends will require that you have them installed as well - if you are missing some, make a copy of the shell script, comment out the parts in question, and use the copy instead of examples.sh. If you don't have any of the SIMD backends, zimt will provide it's own, which does a fairly good job but is usually not as fast as the other backends. But you can go ahead and start coding with zimt, and if you need more speed you can install one of the back-ends later

### zimt backends

On top of 'zimt's own', zimt currently supports three backends:

  - highway
  - Vc
  - std::simd

My recommendation is to use [highway](https://github.com/google/highway/ "Performance-portable, length-agnostic SIMD with run-time dispatch") - it's the most recent development, supports a wide variety of hardware, ind the integration into zimt is well-used because it's now the default back-end for [lux](https://bitbucket.org/kfj/lux/ "git repository of the lux image and panorama viewer"). highway is available from github, and it's also quite available via package managers, so it's a good idea to see if your package manager offers it. zimt currently relies on highway 1.0.4. To use this backend with zimt, #define USE_HWY

[Vc](https://github.com/VcDevel/Vc/ "SIMD Vector Classes for C++") is good for intel/AMD processors up to AVX2, and it has dedicated AVX support, which highway does not provide - there, the next-bast thing is SSE4.2, which is quite similar but a bit slower. Apart from that, using Vc with zimt offers few advantages over using the highway backend. You may find Vc packages via your package manager - if you install from source, pick the 1.4 branch explicitly. To use this backend with zimt, #define USE_VC, and link with -lVc

std::simd is part of the C++ standard and requires C++17. Some C++ libraries now provide an implementation of std::simd, but the implementation coming with g++ is - IMHO and of this writing - not very comprehensive, e.g. missing explict SIMD implementations of some math routines. It is a step up from no backend, though, and if you have a recent libstdc++ installed, it's definitely worth a try. To use this backend with zimt, #define USE_STDSIMD, and use -std=c++17.

For all compiles, it's crucial to use optimization (use -O3) and to specify use of the native CPU (unless you need portability). The latter is affected by '-march=native' on intel/AMD CPUs; for other platforms you may need to specify the actual CPU - e.g. -mcpu=apple-m1 for builds on Apple's M1. I prefer using clang++ over g++, but I'll refrain from a clear recommendation.

## zimt components

In zimt, I provide a type for small aggregates, named 'xel_t' - xel is for things like pi*xel*s etc. - and a type for multidimensional arrays with arbitrary striding, named array_t/view_t . These stand in for the use of vigra::TinyVector and vigra::MultiArray(View) in vspline - vspline does not use much else from vigra, and the vigra library is very large, so I decided to replace these types with my own stripped-down versions to reduce a rather cumbersome dependency and make the code simple, focusing on what's needed in zimt. I also rely heavily on optimization, so I feel that I'm better off with simple constructs, because the chance that the optimizer will recognize optimizable patterns in the code should be higher this way. It's important to note that zimt is made to deal with 'xel' data and fundamentals only - you can create arrays of other data, but you can't use zimt's raison d'être, the transform family of functions, on them, because zimt has no notion of their 'simdized' equivalent. To use arrays of, say, structured types in zimt, you have to reformulate the data as several arrays of fundamentals (typically by creating views with appropriate striding, you don't have to copy the data).

The next important component is a type system for SIMD data. zimt provides a common interface to four different SIMD backends, which allows for easy prototyping, switching of backend library/code base, cherry-picking, etc. The interface is modelled after Vc::SimdArray, which was the first SIMD data type I used extensively. It's part of the [Vc](https://github.com/VcDevel/Vc/ "SIMD Vector Classes for C++") library and it's well-thought-out and pretty much just what I wanted, but Vc is now 'in maintenance mode' and newer ISAs and processors are no longer supported. Much of Vc's functionality made it into the C++ [std::simd](https://en.cppreference.com/w/cpp/experimental/simd/ "cppreference.com page: Data-parallel vector library") standard, and that's the second backend zimt supports. The third one is [highway](https://github.com/google/highway/ "Performance-portable, length-agnostic SIMD with run-time dispatch"), which is a more recent development. It's conceptually quite different from Vc and std::simd, but my interface 'bends it' to something more like Vc, using highway functionality to implement the highway 'flavour' of zimt. Finally, there's a simple small container type (similar to zimt::xel_t) which exhibits the same interface, but uses small loops and hopes for their autovectorization by the compiler's optimizer - a technique I call 'goading'. The 'simdized' types which are provided by the separate backends are all 'substantial' - they are normal objects which can be used as members in other objects. zimt typically operates with xel_t of such simdized types to represent the SIMD-typical SoA pattern.

Next in line is a unary functor type which I call a 'bimodal functor', because it normally offers two member functions 'eval', one for scalar data and one for SIMD-capable data types. The bimodal functors inherit from zimt::unary_functor and offer rudimentary functional composition, which can produce complex functors from simple ones to code, e.g., pixel pipelines for image processing - or, in other words - stencil code. The scalar 'eval' version was used in vspline after peeling, zimt now avoids it's use, but the scalar eval's argument signature still determines the type system. Having it does no harm, but zimt's transform family of functions does not use it. Simple functors can cover both variants with one template, but if syntactic elements like conditionals come into play, the SIMD code will have to be coded differently, using masking instead, and some constructs in SIMD code have no direct equivalent in scalar code. Where the scalar version of a functor would process xel_t of some type T, the simdized eval variant would process xel_t of simdized T. When processing large amounts of data with SIMD code, every now and then you run into a situation when you have a batch of 'leftover' scalar values which don't make up a whole vector. There are various ways of dealing with the situation. vspline currently processes the leftover values with scalar code, but zimt 'stuffs' the vectors so that the initially unoccupied lanes are filled up with copies of values in occupied lanes and subsequently processes the - now full - vector. Most of the time the vector-processing code can be unaware of this 'stuffing' action (an exception being reductions), and zimt offers to use 'capped' processing where the invoked code is told how many of it's lanes are 'genuine' and how many are copied from elsewhere, so where necessary this information is made available to the SIMD pipeline.

What brings it all together is a body of code which performs a concurrent iteration over arbitrary nD arrays of fundamentals or 'xel' data, picking out chunks of data, vectorizing them to SIMD-capable types and invoking the bimodal functors on them, storing back the results to the same or another array. These functions go by the name of 'transform-like' functions, because they are similar to std::transform. There's a variety of these functions, allowing processing of the data or their coordinates, including - with a bit of extra coding effor on behalf of the functor - reductions. Note here that zimt is designed for 'reasonably large bodies of data' - e.g. pixels in an image. The multithreading will - by default - employ twice as many threads as the CPU has cores, which is clearly overkill for small arrays. For such arrays, consider using less 'jobs' by specifying a smaller number in the 'loading bill' (class zimt::bill_t, see bill.h), down to using just one which will effectively singlethread the operation with no threading overhead.

Find the zimt headers in the folder 'include' - apart from the 'include-all' header zimt.h in the repository root. Examples are in the 'examples' folder, and there's a script examples.sh which will compile them with the different backends and clang++ and g++, assuming you have a std::simd implementation, Vc and highway installed.

## How does 'multi-backend' work?

zimt is designed to work with several 'backends' which 'map' the code in zimt to use different SIMD libraries for actual SIMD operations. This may sound confusing at first, but it's in fact very helpful: You only need to understand one API but you can use all the backends with it. Choosing one becomes a matter of simply passing a preprocessor #define. Coding an interface for a specific backend can be a difficult and laborious task; e.g. the highway interface took several weeks of development time. But it's doable, and I'd welcome other SIMD libraries to join in and develop zimt as a common API and widen it's spectrum of backends.

Nevertheless you can opt to use code specific to one or several backends if that suits your needs - you may have legacy code written for one backend, or find that one is particularly suited for a specific purpose. The idea is that zimt should also make the backends interoperable, as demonstrated in 'cherrypicking.cpp'.

To explain how this is done, we need a bit of background.

### horizontal vs. vertical vectorization

zimt deals with fundamental and 'xel' data - and their 'simdized' form. To understand the concept of the simdized form, it's essential to understand that zimt uses a strategy of 'horizontal vectorization'. Let's look at an RGB pixel. 'Vertical vectorization' would look at the pixel itself as a datum which might be processed with SIMD - single instruction, multiple data. One might expect code facilities offering, e.g., arithmetic operations on entire pixels, rather than on their R, G or B component. But this type of vectorization doesn't play well with the hardware: vector units offer registers with a fixed set of N - say, eight or sixteen - fundamentals which can be processed as SIMD units. If we were to put in a single pixel, most of the register's 'lanes' would go empty. And what about operations like white balancing which modify individual colour channels? This leads to the concept of horizontal vectorization: Instead of processing single pixels, we process groups of N pixels, representing them by three registers of N fundamentals each. This is a conceptual shift and *decouples 'MD' from structure semantics*. If we can rely on some mechanism to parcel N pixels into a 'simdized' pixel, we needn't even be aware of the precise number 'N', whereas the simdized pixel's components are now three SIMD vectors, the R, G and B vector of N values each.

Another way to look at this is the conceptual difference between SoA and AoS: 'Structure of Arrays' and 'Array of Structures'. Think of N pixels in memory. This is an AoS: the 'structures' are the pixels, and there's an array of them. Now think of the 'simdized' form: you now have three vectors of N values, a structure (three registers with different meaning: R, G or B) of arrays (the individual registers with N semantically uniform values).

How do raster data fit in? There are - broadly speaking - two ways of holding large bodies of 'xel' data in memory: the 'interleaved' form where each 'xel' occupies it's 'own' region of memory, and the 'per-channel' form where you have one chunk of memory for each of the xel's channels and pixels have to be put together by extracting values with equal offset from each chunk. An in-between mode is storage in successive segments in 'per-channel' form. All of these forms have advantages and disadvantages. The interleaved form's advantage is preservation of locality (each xel has precisely one address in memory), but the disadvantage is that you have to 'deinterleave' the data to a simdized form. Interleaved data are also very common. The per-channel form allows for fast load and store operations, but you need several of them (one per channel) to widely separated regions of memory to assemble a simdized xel. The in-between mode of storing readily simdized data 'imbues' the stored data with the lane count, which is an issue when the data are meant to be portable. zimt focuses on interleaved data, while the other schemes are easy to handle with zimt, but not explicitly provided. To mitigate the effect of de/interleaving, zimt has specialized code for the purpose (see 'interleave.h') - and most of the time you can even be unaware of the de/interleaving operations because the 'transform family' of zimt functions does it automatically, using elaborate code in the backends to handle this task efficiently.

zimt encourages you to build 'pipeline code' by functional composition, so that your entire operation ends up being represented by a single functor which you use with the transform family of functions to interfere with your raster data. If the raster data are interleaved, you have one deinterleave at the beginning of the pipeline, and one interleave at the end. With a sufficiently complex pipeline, the amount of time spent on de/interleaving becomes comparably small, but you have data which are compatible with many other applications. Using this scheme (pipeline functor + transform) requires explicit coding of the pipeline, which is more effort than, say, arithmetics defined on a per-array basis (like std::valarray), but for long pipelines it can be daunting to formulate the entire operation as an expression, luring you into producing intermediates. But memory access is typically the slowest component, so you want to avoid it if at all possible. Using zimt's 'index-based' tansform, you can even 'fill arrays from thin air', using pipeline code which starts out accepting coordinates of target data and taking it from there, producing the target data in the final stage of the pipeline and depositing them in the target array. This tends to be the most efficent way of generating data, because you only have one single memory access when the result of the pipeline is stored, all else is done in registers or, if the registers aren't enough, at least in cache.

### zimt's simd_types: horizontal vectorization with fixed lane count

My explanations above referred to the hardware's registers and their 'lane count', the number of fundamentals a register can hold. At this level of vectorization, the code would deal with 'what hardware registers can hold', but this soon becomes cumbersome: there are fundamentals of different size, and the code may want to have differently-sized fundamentals interacting. This leads to the idea of decoupling the simdized form from the register lane count and using simdized forms which may occupy one *or several* registers, or only a fraction of a register. Vc made this possible by adding the SimdArray data type, which could be made to hold any number of lanes and made the in-register representation of the simdized form an internal issue. This results in SimdArray objects with the same lane count becoming interoperable, even if the fundamentals have different size. Vc offers both the hardware-compatible vector and the SimdArray. std::simd offers fixed-size vectors which work just the same. highway does not - it's closer to the hardware, so zimt adds a layer closing the gap.

zimt offers it's own 'simd_type' data type which has a fixed lane count. It's coded as an object holding a small C-style array of fundamentals and 'typical' SIMD operations, like loading and storing data from/to memory, arithmetic, etc. zimt::simd_type is the 'role model', and it's used as the 'fallback' SIMD type which can contain any lane count of all types of fundamentals. This is the same spectrum that std::simd fixed-size vectors cover. When zimt is made to use Vc, it uses Vc::SimdArray where possible, and zimt::simd_type for the rest. The same applies for use of highway: zimt provides an extra layer to provide fixed-size simdized types which may use several hardware registers. The 'internal' data types used with Vc and highway are vc_simd_type and hwy_simd_type. zimt's speciality is the abstraction from the concrete implementation: user code only has to decide which 'backend' zimt should use, and the concrete types used for SIMD operations can be obtained by querying set of a traits classes. So the actual type doesn't have to become explicit:

    #include "zimt/vector.h"
    ...
    typedef zimt::simdized_type < float , 3 > pixel_v ;
    pixel_v px ;
    std::cout << px << std::endl ;

the concrete type of pixel_v will depend on which backend was chosen. If USE_VC is defined, the type will be a zimt::vc_simd_type, with USE_HWY it will be a zimt::hwy_simd_type and so on. The echo to std::cout will look different for each 'flavour'.

### zimt::unary_functor: providing a type system for SIMD functions

zimt::unary_functor is the building block for the creation of SIMD functions with zimt, and it takes care of providing types to code SIMD operations. Again, user code can remain ignorant of the concrete SIMD types used - zimt 'abstracts them away', and allows user code to simply state which backend it would like to be used. zimt::unary_functors are unary functors insofar as they process one datum as input and produce one datum as output. They usually contain a 'scalar form' and a 'simdized' form of the operation. the data can be simple fundamentals or 'xels' and their vectorized equivalent for the simdized form. Let's look at a simple functor which multiplies pixels with 2.0f:

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

You may ask: "where is the SIMD code?" The 'eval' function is very simple - it uses no conditionals, just plain arithmetic. For such data, the 'simdized' code and the scalar code are the same, and so 'eval' is coded as a template. The two 'modes' of the 'bimodal functor' don't have to be coded explicitly. This is only necessary for code which needs different code for the scalar and simdized form. Let's say we want to double the pixels, but limit the value to a given maximum, let's pick 255.0f:

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

Now we have two separate eval functions. Note how the code is unaware of the precise types of in_v and out_v - simdized pixels - and relies on the parent class zimt::unary_functor to provide them, depending on the chosen backend. The simdized form uses an idiom from Vc: a masked assignment. The idiom

    target ( mask ) = value ;

assigns value to all lanes for which 'mask' is true. Because this is such a succinct way of coding the operation, I have adopted it in zimt and provide it as a common feature for all backends. We can now use the functor to process both scalar and simdized data:

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
    }

What do I mean by 'building block'? zimt offers 'chaining' of functors. This combines two zimt::unary_functors so that the first one's output is taken as the second one's input. The combined functor is a new unary functor in it's own right. using 'f' from above:

    auto ff = f + f ;
    f.eval ( pxv1 , pxv2 ) ;
    std::cout << pxv1 << " -> " << pxv2 << std::endl ;

Now you can see that all results are capped at 255, even the R component has exceeded the limit after having been doubled twice. Chaining functors is a bit like lambda calculus, and takes some getting used to. But it's possible to structure code so that it becomes a chain of unary functors, so chaining gets you a long way. Of course you're free to extend on the basic functional programming offered by zimt. Note, again, that the scalar form is nice-to-have, but the zimt::transform family of functions won't use it. I recommend to code it nevertheless - it's quite feasible to code so that an initial version of the pipeline code relies only on the scalar code (using 'broadcasting'). Once you have ascertained that the code does what it's supposed to do, you can tackle the simdized form, and you can doublecheck it's results against the scalar form, so you can be assured that the simdized form is equally correct. With complex pipeline code, this is more useful than you may think. Finally, if your code stands 'both ways', if you are pressed for binary volume, you can just comment out the scalar version.

Next to chaining, the most common functional construct is probably 'embedding' one functor into another, so that the 'host' can use it for some aspect of it's work, adding more functionality which modifies it's own 'incoming' value before passing it on to the embedded functor as it's input - and/or more functionality to further process the embedded functor's output to produce the host's output. There is no generalized infrastructure code for functor embedding in zimt, you have to hand-code it, typically working with RAII, passing the to-be-embedded functor to the host functor's c'tor. It's not 'rocket science', but previous exposition to the functional paradigm helps ;)

### cherry-picking SIMD code

The different SIMD backends have different strengths and weaknesses. When using zimt with one particular backend, like using Vc by #defining USE_VC, you get the 'package deal', and all SIMD code will be generated using the chosen backend. But with zimt's multi-backend approach, you can also use several of the backends at the same time, and use one backend's strengths to compensate another backend's weaknesses. I've included an example called 'cherrypicking.cpp' which does just that with a program using the std::simd backend, but slotting in a functor using Vc. I've picked code using atan2, which is currently inefficient in std::simd, but fast in Vc, and the performance differs significantly - the precise factor depending on the CPU. Note that this is true on intel/AMD platforms and the std::simd implementation coming with g++ at the time of this writing. For other architectures, Vc offers no efficient implementation of atan2, whereas the local std::simd implementation may.

The upshot is that, with little programming effort, you can put together SIMD code from different libraries. Of course you might do that 'manually', but you'd have to handle the different APIs of the libraries involved, whereas with zimt you get compatible types with identical behaviour and you can easily move from one to the other, with only a small amount of 'friction' when it comes to move your data from one type to the other, which may even be optimized away by the compiler if you're lucky.

Please note that this feature is currently 'evolving', so the act of moving from one type to the other by simple assignment may not yet work as well for all of zimt's SIMD backend types as in my example.

## using several ISAs/backends in one binary

If you look at the zimt code, you'll notice that there is no inherent facility to use several ISAs in one binary - instead, everything is made so that you can compile the code for one specific ISA. But there is a simple method to combine code for several ISAs in one binary which I have established in lux, and I'll outline it here.

The trick is to 'dub' the zimt namespace to something ISA-specific with a simple preprocessor #define, and then compile separate TUs, each with it's own set of compiler switches. The calling code can then pick the ISA-specific version. So if we have ISA specific code like this (file isa-specific.cc):

    #include "../../zimt.h"

    namespace zimt
    {
      void foo()
      {
        typedef zimt::simdized_type < float , 16 > f16_t ;
        std::cout << "base vector size: "
                  << sizeof ( f16_t::vec_t ) << std::endl ;
        std::cout << f16_t::iota() << std::endl ;
      }
    } ;

We might wrap it in the 'dubbing' code like this for an AVX version (foo_avx.cc):

    #define zimt zimt_AVX
    #include "isa_specific.cc"

And like this for an SSE version:

    #define zimt zimt_SSE
    #include "isa_specific.cc"

Calling code can pick the ISA-specific code like this (main.cc):

    namespace zimt_AVX
    {
      extern void foo() ;
    } ;

    namespace zimt_SSE
    {
      extern void foo() ;
    } ;

    int main ( int argc , char * argv[] )
    {
      zimt_SSE::foo() ;
      zimt_AVX::foo() ;
    }

Now we can put it all together:

    $ g++ -msse2 -c -DUSE_HWY -o foo_sse.o foo_sse.cc
    $ g++ -mavx2 -c -DUSE_HWY -o foo_avx.o foo_avx.cc
    $ g++ main.cc *.o
    $ ./a.out
      base vector size: 4
      (* 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 *)
      base vector size: 16
      (* 0, 1, 2, 3 | 4, 5, 6, 7 | 8, 9, 10, 11 | 12, 13, 14, 15 *)

I've picked the highway backend for this example because hwy_simd_type has a vec_t member, and we can actually see a difference in the output, USE_VC works as well. With this technique, you can combine any number of TUs with different compiler directives into a single binary, and you can also combine code from different backends - e.g. if you intend to compare performance and want everything else to be the same.

Having to declare each function in the dubbed namespace can become annoying and error-prone - a simple way out is to use a 'dispatcher' object: a class with pure virtual member functions which you instantiate in each ISA-specific TU. With a bit of header artistry, you can make this to work so that you only need to write the function declaration once, to see this in action have a look at how it's done in [lux](https://bitbucket.org/kfj/lux/ "git repository of the lux image and panorama viewer"), see interface.h and dispatch.h there. Then you use a set of dispatcher objects, one for each ISA-specific TU, and call the member functions through them, resulting in calls to the ISA-specific overloads via the virtual member functions.

## n-dimensional arrays, zimt::transform and relatives

Wa can use zimt's unary functors to process n-dimensional arrays. We'll start with an explanation of n-dimensional arrays and 'views' - there is more to them than what you might expect when you just think of multidimensional C arrays. The C++ standard library has a data type which is similar to the arrays zimt processes, but it's mildly exotic, and chances are you haven't heard about it: it's std::gslice. We won't look at this similarity here, I only mention it for reference. Another close relative is NumPy's ndarray. What's common between these data types is that they describe a block of equally-typed data in n-dimensional space which has a fixed size in every direction - so, in 2D, it's like a rectangular grid, i 3D, it's box-shaped etc..

### nD arrays, strides and memory

Memory presents itself to the programmer as a one-dimensional array of bytes. Any notion of multidimensionality is created by chopping up this continuous array of bytes into chunks - and possibly chopping up the chunks again into smaller chunks and so forth. Let's assume we have 1000 bytes. If we chop them up into 10 chunks of 100 bytes each, we can think of these 10 chunks as lines of a 2D array, where each line contains 100 columns. Now how do we get from one column to the next? We add one to it's adress. But if we want to get to the same column in the next line, we have to add 100! This 'number we must add to get to the next one' is called the stride, and in this example we have 'two' strides: the first one - from one column to the next - is 1, the second one is 100. If we were to furter subdivide the lines into - let's call them 'segments' - of 10 bytes each, we'd now have three strides: 1, 10, and 100, which get us from one byte to it's successor in the line, the byte at the same position in the next segment, or the next line, respectively.

With the example above, we see that we have two metrics which describe an array. The first is the extent along each axis, this is called the array's 'shape'. In the example above, we had a shape of (100, 10) for the 2D array and a shape of (10, 10, 10) for the 3D array. Note the notation I use: I start out with the extent of the smallest chunk. The product of the shape's components is the size of the entire array - 1000 in both cases. The other metric is the array's strides, as explained above: (1, 100) or (1, 10, 100), respectively. This should give you a notion of what the shape, size and strides of an array are, and you can use the shape and strides as a method to find a member of an array in memory or to find a memory location's nD coordinate in an array. Let's start with the first one, and let's not use a specific dimensionality. We'll use this notation: let a 'coordinate' Ck denote an n-tuple of k integers, M an address in memory, and the strides, Sk, another n-tuple of integers. The formula to find an array element's location in memory is

    m = M + sum ( Ck * Sk )

provided that M is the 'base' address of the array, the location of the array element with an all-zero coordinate. Let's use the 2D example above and let's say M is zero. If we want to find the array element (3, 5), we apply the formula and get

    m = 0 + sum ( (3, 5) * (1, 100) )
    m = sum ( 3 , 500 )
    m = 503

If we just use the formula, we needn't even have a notion of memory or data - it can stand by itself. The method is applicable with coordinates of any dimensionality. As long as the strides have certain properties, you'll even get a unique result, making the process reversible:

    m % 100 = 3
    m / 100 = 5

In zimt, array and view objects are thin wrappers around the set of M, Ck and Sk - plus a std::shared_ptr for arrays, which keeps track of memory ownership.

What's left is an important abstraction: you don't have to deal with a block of memory, instead your data can be spread out in the address space with gaps between them and you may get 'from one to the next' along with arbitrary strides. Your array - or view - now becomes the set of memory locations which follow the M + sum ( Ck * Sk ) formula, but you can pick any Sk (set of strides) you like so as to fit any 'regular', grid-like distribution of data. One common 'deviation' from 'ordinary' arrays is, for example, to put M at the location of the array element with the highest address and use negative strides. Other common techniques with arrays/views are 'slicing' and 'windowing', which both produce new views to part of the original array's data by using different M and Sk and different Ck to address them.

In zimt, you can access members of an array/view easily with n-dimensional discrete coordinates. Let's say you have a 2D zimt view v. If you want it's element in line 3, column 8, you can access it with

    v [ { 8 , 3 } ]

Where the part in curly braces can also be kept in a zimt xel_t variable:

    auto crd = zimt::xel_t < int , 2 > { 8 , 3 }
    v [ crd ]

### zimt::transform

If you have - or intend to generate - data in a form which fits the 'nD array model' - sporting a set of M, Ck and Sk - zimt offers you a powerful family of functions to get the job done with several threads and SIMD code. zimt will handle all the nitty-gritty detail and leave you to do the interesting bits: on the inside, to write 'pipeline' code - functions which are capable of processing 'simdized' data, and on the outside, your application with it's arrays of data which you can create and manipulate with zimt. zimt::transform and relatives are go-between code which 'rolls out' your pipeline code to your arrays, using efficient methods to 'feed' your pipeline code with input and dispose of the pipeline's output.

### an abstraction

Looking at zimt::transform and family, you can see that they bring together the pipeline code (the 'act' functor) and one or two arrays. But there is a deeper and more abstract layer of code which is used to actually implement the 'transform family': it's the code in wielding.h. There you can find the backend function template 'process' which is an abstraction: it may or may not relate to arrays, but it follows their structural properties. Instead of necessarily fetching input data from an array, it uses 'get' objects with a specific set of capabilites to provide data which can serve as input to the 'act' functor - and 'put' objects which can dispose of the act functor's result. With this abstraction, and your own 'get' and 'put' classes, the process becomes very versatile. You might even set up your code so that there is no 'real' act functor but rather an 'empty shell' object which merely passes it's input through. But this doesn't make for the best designs unless you really only need to do something trivial. The rule where to draw the line between code to go into the 'get' and 'put' objects and the 'pipeline code' in the act functor is that you want the code 'in the pipeline' when the usefulness of handling the data segments as 1D entities ends. An example would be handling interpolation: you might use the get object to produce coordinates in some 'model space', readily scaled and shifted. The pipeline code can then take over and do the job of finding RGBA values for the coordinates in model space. Once you have the RGBA values, you may write a put object to 'brush up' your RGBA data, applying stuff like colour space conversions, gradation or quantization. Decoupling the three steps makes it easier to write your program to be flexible: to stick with the given example, if you want to store to different formats, you don't have to modify the pipeline, just use different put objects. And you can still use a zimt function to orchestrate the dispatch to the worker threads and the data parcelling strategy *as if* you were processing an array. The get and put object act as *accessors*, while the wielding code holds a notion of something N-dimensional with a given N-dimensional shape where content is uniquely related to discrete N-dimensional coordinates - with this coordinate acting as an *iterator*. Because of the specific way in which zimt's 'wielding' code traverses - or iterates over - the given coordinate space, the accessors - the get and put objects - can be made to precisely fit their niche (rather than using the C++ standard 'mechanics') and get along with a minimum amount of CPU cycles, giving you maximal performance without sacrificing flexibility.

To sum this up, you can think of what happens in pseudocode:

    for all discrete coordinates Ik
      vi = get ( Ik )
      vo = act ( vi )
      put ( vo )

Where zimt::tansform is less general and does stuff like

    for all values V in array A
      vo = act ( V )
      store vo in array B

### grokking

I've borrowed a term from SciFi which I first started using in vspline, and it's turned out useful, so I carried it over to zimt: I call it 'grokking' functors - from the neologism ["grok"](https://en.wikipedia.org/wiki/Grok "wikipedia entry on 'grok'") in it's meaning 'to understand'. functors are 'standard fare' in zimt. Every call to zimt::transform needs an 'act' functor, and zimt::process additionally takes a get_t and a put_t functor. The concrete type of these functors may vary, to allow a wide variety of operations. But all act functors share the same interface, and the same is true for all get_t and put_t functors. One way to handle such polymorphism is to use a common base class with virtual functions and derive concrete classes from it. I've chosen a similar design, but I avoid the common base class, and instead use a technique called 'type erasure' which captures the specific functionality in std::functions and presents a type which has the same interface, but no visible peculiarities apart from that. The implementation of the 'grokking' process and the associated objects is quite a mouthful, but use is simple due to a bunch of factory functions. For an act functor 'af', it's as simple as this:

    auto ga = zimt::grok ( af ) ;

For a get_t object 'g' and a put_t object 'p' it's

    auto gg = zimt::grok_get ( g ) ;
    auto gp = zimt::grok_put ( p ) ;

ga, gg, and gp can now be used wherever an object of the 'grokked' type might have been used, they can be copied freely, passed as arguments etc. - but because they are of grok_type, grok_get_t and grok_put_t, you can write code which, for example, accepts such objects rather than template code accepting all kinds of functor types. It makes coding complex functor compositions much easier.
