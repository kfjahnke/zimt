# zimt - C++11 template library to process n-dimensional arrays with multi-threaded SIMD code

Zimt is the German word for cinnamon, and it sounds vaguely like SIMD. It should imply something spicy and yummy, and apart from the silly pun there is no deeper meaning.

The code is 'just forked', there is yet much to be done, but I prefer to get it out early, also to attract others to join in. Because it's in large parts from an extant library which has seen years of development, the code is already at a reasonably good level, but I haven't yet done extensive tests or added examples. The code is heavily commented, though.

The code in this library is based on code from my library [vspline](https://bitbucket.org/kfj/vspline/ "git repository of the vspline library"). I found that the tools I developed there to process n-dimensional arrays with multi-threaded SIMD code would be useful outside of a b-spline library's context, and that I might be better off letting go of the [vigra](http://ukoethe.github.io/vigra/ "The VIGRA Computer Vision Library") library, which I use in vspline for data handling and small aggregates.

## zimt components

In zimt, I provide a type for small aggregates, named 'xel_t' - xel is for things like pi*xel*s etc. - and a type for multidimensional arrays with arbitrary striding, named array_t/view_t . These stand in for the use of vigra::TinyVector and vigra::MultiArray(View) in vspline - vspline does not use much else from vigra, and the vigra library is very large, so I decided to replace these types with my own stripped-down versions to reduce a rather cumbersome dependency and make the code simple, focusing on what's needed in zimt.

The next important component is a type system for SIMD data. zimt provides a common interface to four different SIMD backends, which allows for easy prototyping, switching of back-end library/code base, cherry-picking, etc. The interface is modelled after Vc::SimdArray, which was the first SIMD data type I used extensively. It's part of the [Vc](https://github.com/VcDevel/Vc/ "SIMD Vector Classes for C++") library and it's well-thought-out and pretty much just what I wanted, but Vc is now 'in maintenance mode' and newer ISAs and processors are no longer supported. Much of Vc's functionality made it into the C++ [std::simd](https://en.cppreference.com/w/cpp/experimental/simd/ "cppreference.com page: Data-parallel vector library") standard, and that's the second back-end zimt supports. The third one is [highway](https://github.com/google/highway/ "Performance-portable, length-agnostic SIMD with run-time dispatch"), which is a more recent development. It's conceptually quite different from Vc and std::simd, but my interface 'bends it' to something more like Vc, using highway functionality to implement the highway 'flavour' of zimt. Finally, there's a simple small container type (similar to zimt::xel_t) which exhibits the same interface, but uses small loops and hopes for their autovectorization by the compiler's optimizer - a technique I call 'goading'.

Next in line is a unary functor type which I call a 'bimodal functor', because it offers two member functions 'eval', one for scalar data and one for SIMD-capable data types. The bimodal functors inherit from zimt::unary_functor and offer rudimentary functional composition, which can produce complex functors from simple ones to code, e.g., pixel pipelines for image processing - or, in other words - stencil code.

What brings it all together is a body of code which performs a concurrent iterations over arbitrary nD arrays of fundamentals or 'xel' data, picking out chunks of data, vectorizing them to SIMD-capable types and invoking the bimodal functors on them, storing back the results to the same or another array. These functions go by the name of 'transform-like' functions, because they are similar to std::transform. There's a variety of these functions, allowing processing of the data or their coordinates, and also reductions.

Find the zimt headers in the folder 'include' - apart from the 'include-all'
header zimt.h in the repository root. Examples are in the 'examples' folder,
and there's a script examples.sh which will compile them with the different
backends and clang++ and g++, assuming you have a std::simd implementation,
Vc and highway installed.

## How does 'multi-back-end' work?

zimt is designed to work with several 'back-ends' which 'map' the code
in zimt to use different SIMD libraries for actual SIMD operations. This
may sound confusing at first, but it's in fact very helpful: You only need
to understand one API but you can use all the back-ends with it. Choosing
one becomes a matter of simply passing a preprocessor #define. Coding an
interface for a specific back-end can be a difficult and laborious task;
e.g. the highway interface took several weeks of development time. But
it's doable, and I'd welcome other SIMD libraries to join in and develop
zimt as a common API and widen it's spectrum of backends.

Nevertheless you can opt to use code specific to one or several back-ends
if that suits your needs - you may have legacy code written for one
back-end, or find that one is particularly suited for a specific purpose.
The idea is that zimt should also make the back-ends interoperable, as
demonstrated in 'cherrypicking.cpp'.

To explain how this is done, we need a bit of background.

### horizontal vs. vertical vectorization

zimt deals with fundamental and 'xel' data - and their 'simdized' form. To understand the concept of the simdized form, it's essential to understand that zimt uses a strategy of 'horizontal vectorization'. Let's look at an RGB pixel. 'Vertical vectorization' would look at the pixel itself as a datum which might be processed with SIMD - single instruction, multiple data. One might expect code facilities offering, e.g., arithmetic operations on entire pixels, rather than on their R, G or B component. But this type of vectorization doesn't play well with the hardware: vector units offer registers with a fixed set of N - say, eight or sixteen - fundamentals which can be processed as SIMD units. If we were to put in a single pixel, most of the register's 'lanes' would go empty. And what about operations like white balancing which modify individual colour channels? This leads to the concept of horizontal vectorization: Instead of processing single pixels, we process groups of N pixels, representing them by three registers of N fundamentals each. This is a conceptual shift and *decouples 'MD' from structure semantics*. If we can rely on some mechanism to parcel N pixels into a 'simdized' pixel, we needn't even be aware of the precise number 'N', whereas the simdized pixel's components are now three SIMD vectors, the R, G and B vector of N values each.

Another way to look at this is the conceptual difference between SoA and Aos: 'Structure of Arrays' and 'Array of Structures'. Think of N pixels in memory. This is an AoS: the 'structures' are the pixels, and there's an array of them. Now think of the 'simdized' form: you now have three vectors of N values, a structure (three registers with different meaning: R, G or B) of arrays (the individual registers with N semantically uniform values).

How do raster data fit in? There are - broadly speaking - two ways of holding large bodies of 'xel' data in memory: the 'interleaved' form where each 'xel' occupies it's 'own' region of memory, and the 'per-channel' form where you have one chunk of memory for each of the xel's channels and pixels have to be put together by extracting values with equal offset from each chunk. An in-between mode is storage in successive segments in 'per-channel' form. All of these forms have advantages and disadvantages. The interleaved form's advantage is preservation of locality (each xel has precisely one address in memory), but the disadvantage is that you have to 'deinterleave' the data to a simdized form. interleaved data are also very common. The per-channel form allows for fast load and store operations, but you need several of them (one per channel) to widely separated regions of memory to assemble a simdized xel. The in-between mode of storing readily simdized data 'imbues' the stored data with the lane count, which is an issue when the data are meant to be portable. zimt focuses on interleaved data, while the other schemes are easy to handle with zimt, but not explicitly provided. To mitigate the effect of de/interleaving, zimt has specialized code for the purpose (see 'interleave.h') - and most of the time you can even be unaware of the de/interleaving operations because the 'transform family' of zimt functions does it automatically.

zimt encourages you to build 'pipeline code' by functional composition, so that your entire operation ends up being represented by a single functor which you use with the transform family of functions to interfere with your raster data. If the raster data are interleaved, you have one deinterleave at the beginning of the pipeline, and one interleave at the end. With a sufficiently complex pipeline, the amount of time spent on de/interleaving becomes comparably small, but you have data which are compatible with many other applications. Using this scheme (pipeline functor + transform) requires explicit coding of the pipeline, which is more effort than, say, arithmetics defined on a per-array basis (like std::valarray), but for long pipelines it can be daunting to formulate the entire operation as an expression, luring you into producing intermediates. But memory access is typically the slowest component, so you want to avoid it if at all possible.

### zimt's simd_types: horizontal vectorization with fixed lane count

My explanations above referred to the hardware's registers and their 'lane count', the number of fundamentals a register can hold. At this level of vectorization, the code would deal with 'what hardware registers can hold', but this soon becomes cumbersome: there are fundamentals of different size, and the code may want to have differently-sized fundamentals interacting. This leads to the idea of decoupling the simdized form from the register lane count and using simdized forms which may occupy one *or several* registers, or only a fraction of a register. Vc made this possible by adding the SimdArray data type, which could be made to hold any number of lanes and made the in-register representation of the simdized form an internal issue. This results in SimdArray object with the same lane count becoming interoperable, even if the fundamentals have different size. Vc offers both the hardware-compatible vector and the SimdArray. std::simd offers fixed-size vectors which work just the same. highway does not - it's closer to the hardware, so zimt adds a layer closing the gap.

zimt offers the 'simd_type' data type which has a fixed lane count. It's coded as an object holding a small C-style array of fundamentals and 'typical' SIMD operations, like loading and storing data from/to memory, arithmetic, etc. zimt::simd_type is the 'role model', and it's used as the 'fallback' SIMD type which can contain any lane count of all types of fundamentals. This is the same spectrum that std::simd fixed-size vectors cover. When zimt is made to use Vc, it uses Vc::SimdArray where possible, and zimt::simd_type for the rest. The same applies for use of highway: zimt provides an extra layer to provide fixed-size simdized types which may use several hardware registers. The 'internal' data types used with Vc and highway are vc_simd_type and hwy_simd_type. zimt's speciality is the abstraction from the concrete implementation: user code only has to decide which 'back-end' zimt should use, and the concrete types used for SIMD operations can be obtained by querying set of a traits classes. So the actual type doesn't have to become explicit:

    #include "zimt/vector.h"
    ...
    typedef zimt::simdized_type < float , 3 > pixel_v ;
    pixel_v px ;
    std::cout << px << std::endl ;

the concrete type of pixel_v will depend on which back-end was chosen. If USE_VC is defined, the type will be a zimt::vc_simd_type, with USE_HWY it will be a zimt::hwy_simd_type and so on. The echo to std::cout will look different for each 'flavour'.

### zimt::unary_functor: providing a type system for SIMD functions

zimt::unary_functor is the building block for the creation of SIMD functions with zimt, and it takes care of providing types to code SIMD operations. Again, user code can remain ignorant of the concrete SIMD types used - zimt 'abstracts them away', and allows user code to simply state which back-end it would like to be used. zimt::unary_functors are unary functors insofar as they process one datum as input and produce one datum as output. They contain a 'scalar form' and a 'simdized' form of the operation. the data can be simple fundamentals or 'xels' and their vectorized equivalent for the simdized form. Let's look at a simple functor which multiplies pixels with 2.0f:

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

Now we have two separate eval functions. Note how the code is unaware of the precise types of in_v and out_v - simdized pixels - and relies on the parent class zimt::unary_functor to provide them, depending on the chosen back-end. The simdized form uses an idiom from Vc: a masked assignment. The idiom

    target ( mask ) = value ;

assigns value to all lanes for which 'mask' is true. Because this is such a succinct way of coding the operation, I have adopted it in zimt and provide it as a common feature for all back-ends. We can now use the functor to process both scalar and simdized data:

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

Now you can see that all results are capped at 255, even the R component has exceeded the limit after having been doubled twice. Chaining functors is a bit like lambda calculus, and takes some getting used to. But it's possible to structure code so that it becomes a chain of unary functors, so chaining gets you a long way. Of course you're free to extend on the basic functional programming offered by zimt.

### cherry-picking SIMD code

The different SIMD back-ends have different strengths and weaknesses. When
using zimt with one particular back-end, like using Vc by #defining USE_VC,
you get the 'package deal', and all SIMD code will be generated using the
chosen back-end. But with zimt's multi-back-end approach, you can also use
several of the back-ends at the same time, and use one back-end's strengths
to compensate another back-end's weaknesses. I've included an example called
'cherrypicking.cpp' which does just that with a program using the std::simd
back-end, but slotting in a functor using Vc. I've picked code using atan2,
which is inefficient in std::simd, but fast in Vc, and the performance
differs significantly - the precise factor depending on the CPU. Note
that this is true on intel/AMD platforms and the std::simd implementation coming with g++ at the time of this writing. For other architectures, Vc
offers no efficient implementation of atan2, whereas the local std::simd implementation may.

The upshot is that, with little programming effort, you can put together
SIMD code from different libraries. Of course you might do that 'manually',
but you'd have to handle the different APIs of the libraries involved,
whereas with zimt you get compatible types with identical behaviour and
you can easily move from one to the other, with only a small amount of
'friction' when it comes to move your data from one type to the other,
which may even be optimized away by the compiler if you're lucky.

Please note that this feature is currently 'evolving', so the act of moving
from one type to the other by simple assignment may not yet work as well for
all of zimt's SIMD back-end types as in my example.

## using several ISAs/backends in one binary

If you look at the zimt code, you'll notice that there is no inherent facility
to use several ISAs in one binary - instead, everything is made so that you
can compile the code for one specific ISA. But there is a simple method to
combine code for several ISAs in one binary which I have established in lux,
and I'll outline it here.

the trick is to 'dub' the zimt namespace to something ISA-specific with a
simple preprocessor #define, and then compile separate TUs, each with it's
own set of compiler switches. The calling code can then pick the ISA-specific
version. So if we have ISA specific code like this (file isa-specific.cc):

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

We wrap it in the 'dubbing' code like this for an AVX version (foo_avx.cc):

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

I've picked the highway backend for this example because hwy_simd_type has
a vec_t member, and we can actually see a difference in the output, USE_VC
works as well. With this technique, you can combine any number of TUs with
different compiler directives into a single binary, and you can also combine
code from different backends - e.g. if you intend to compare performance and
want everything else to be the same.

Having to declare each function in the dubbed namespace can become annoying
and error-prone - a simple way out is to use a 'dispatcher' object: a class
with pure virtual member functions which you instantiate in each ISA-specific
TU. With a bit of header artistry, you can make this to work so that you only
need to write the function declaration once, to see this in action have a look
at how it's done in [lux](https://bitbucket.org/kfj/lux/ "git repository of the lux image and panorama viewer"), see interface.h and dispatch.h there. Then you
use a set of dispatcher objects, one for each ISA-specific TU, and call the member
functions through them, resulting in calls to the ISA-specific overloads via
the virtual member functions.

