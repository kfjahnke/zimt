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

/*! \file unary_functor.h

    \brief interface definition for unary functors

    The central element in zimt is the 'act' functor. This is the object
    containing the SIMD code which is 'rolled out' over the data with
    multiple threads. This functor is passed to the transform family of
    zimt functions, which in turn pass it to zimt::process, the core
    function of zimt.

    The functionality of processing data is coded as a member function
    called 'eval', taking it's input via a const& and yielding it's
    output to a plain reference. There is no return value. This slightly
    unusual construct has the advantage that both the input and the
    output are arguments to eval, which makes type deduction easier than
    with 'normal' functions taking input as argument and returning output
    as return values.

    A unary_functor processes 'simdized' arguments. In vspline,
    where the code saw it's first iteration, I used what I call 'bimodal'
    functors which had one simdized and one scalar overload, so that
    they could process scalar arguments as well. In the vspline design,
    this was necessary: After a 'peeling' run, which processed all data
    that could form 'full' SIMD vectors, the 'leftover' was processed by
    a small loop using the scalar evaluation function. zimt's design uses
    another method to deal with 'leftovers': It puts them into a SIMD
    vector and fills the vectors up with copies of previously used data
    in order to make sure the filled-up part of the vectors hold valid
    data, and subsequently processes the - now full - vectors with the
    simdized eval function. Apart from functors which have side effects
    this works just fine, but if there are side effects (think of
    reductions) the functor must be aware of the 'stuffing' and ignore
    these parts of the vectors. To handle such situations, 'act' functors
    with side effects must provide an overload of the eval function taking
    a size_t argument indicationg how many lanes are 'genuine'.
    zimt::process calls this overload (if it's present) when processing
    arrives at the point where stuffing happens. If no 'capped' overload
    exists, processing is routed to the 'uncapped' eval.

    unary_functors are made to process SIMD vectors of fundamentals
    and 'simdized' 'xel' data. A xel is a small homogeneous aggregate of
    fundamentals, like a pixel or a voxel, hence the 'xel'. Given a xel
    of N fundamentals of type T (zimt::xel_t < T , N >), it's 'simdized'
    form is a xel of SIMD vectors. If N == 1, there is no actual difference
    between the 'naked' SIMD vector and the single-channel xel, and the
    code accepts and processes both as equivalent.

    To parametrize a unary_functor, you don't need to explicitly
    specify the simdized arguments to 'eval', because they can be generated
    form the scalar form of the xel data. All that is needed is the type of
    the fundamental types of input and output, the respective channel counts,
    and the number of lanes to be used for the SIMD vectors. The latter
    datum can be chosen arbitrarily in zimt but should be a power of two;
    it does not have to be the hardware lane count for the given fundamentals.

    class unary_functor takes three template arguments:
    
    - the argument type, IN
    
    - the result type, OUT
    
    - the number of fundamentals (float, int etc.) in a vector, _vsize
    
    The vectorized argument and result type are deduced from IN, OUT and
    _vsize by querying vector_traits.

    IN and OUT can be fundamental types or zimt::xel_t of fundamentals,
    and _vsize (or, in zimt, oftentimes L) is the vector lane count,
    which should be a small-ish power of two, like 16.

    So, typical unary_functor instatiations would be

    unary_functor < int , float , 16 >
    unary_functor < zimt::xel_t<float,2> , zimt::xel_t<float,3> , 8 >
    unary_functor < zimt::xel_t < int , 2 > , double >
    etc.

    And the instatiated unary_functor has the type system which
    is needed to work with zimt::transform. zimt::process is a bit more
    'low-level' than zimt::transform and requires all arguments to be
    zimt::xel_t. Use zimt::vs_adapter to adapt unary_functors with
    fundamentals as IN or OUT; this will create a compatible functor
    which uses zimt::xel_t < T , 1 > instead of the fundamentals.
*/

// array.h is used only for yield_type, which might be moved to
// a different header. But it's an SIMD-ISA-independent header
// anyway, and will be included only once. array.h does in turn
// include xel.h, so we can use zimt::xel_t.
// The code 'proper' of this header needs to be coded separately
// for each SIMD ISA, though, so we start with the usual sentinel
// construct to make the code manageable by foreach_target.h

#include "array.h"

// #ifndef ZIMT_UNARY_FUNCTOR_H
// #define ZIMT_UNARY_FUNCTOR_H

#if defined(ZIMT_UNARY_FUNCTOR_H) == defined(HWY_TARGET_TOGGLE)
  #ifdef ZIMT_UNARY_FUNCTOR_H
    #undef ZIMT_UNARY_FUNCTOR_H
  #else
    #define ZIMT_UNARY_FUNCTOR_H
  #endif

#include "common.h"
#include "vector.h"

HWY_BEFORE_NAMESPACE() ;
BEGIN_ZIMT_SIMD_NAMESPACE(zimt)

/// we derive all unary_functors from this empty class, to have
/// a common base type for all of them. This enables us to easily check if
/// a type is a unary_functor without having to wrangle with
/// unary_functor's template arguments.
 
template < size_t _vsize >
struct unary_functor_tag { } ;

/// class unary_functor provides a functor base class which offers a
/// system of types for concrete unary functors derived from it.
/// If vectorization isn't used, this is trivial, but with
/// vectorization in use, we get vectorized types derived from plain
/// IN and OUT via query of vector_traits.
///
/// class unary_functor itself does not provide operator(). It is expected
/// that the derived classes provide evaluation capability in the form
/// of an 'eval' member function, optionally with a 'capped' overload.
/// eval is to be coded as taking it's first argument as a const&, and
/// writing it's result to it's second argument, which it receives by
/// reference. eval's return type is void.
///
/// The type system used in unary_functor is taken from
/// vector_traits, additionally prefixing the types with in_
/// and out_, for input and output types. The other elements of the
/// type names are the same as in vector_traits.

// KFJ 2020-11-15 it does no harm defining that the default for
// OUT should be IN, and this is a very common situation.

template < typename IN ,       // argument or input type
           typename OUT = IN , // result type
           size_t _vsize = vector_traits < IN > :: size
         >
struct unary_functor
: public unary_functor_tag < _vsize >
{
  static const bool has_capped_eval = false ;

  // number of fundamentals in simdized data.

  enum { vsize = _vsize } ;

  // number of dimensions. This may well be different for IN and OUT.

  enum { dim_in = vector_traits < IN > :: dimension } ;
  enum { dim_out = vector_traits < OUT > :: dimension } ;
  
  // typedefs for incoming (argument) and outgoing (result) type. These two types
  // are non-vectorized types, like zimt::xel_t < float , 2 >. Since such types
  // consist of elements of the same type, the corresponding vectorized type can be
  // easily automatically determined.
  
  typedef IN in_type ;
  typedef OUT out_type ;
  
  // elementary types of same. we rely on vector_traits to provide
  // these types.
  
  typedef typename vector_traits < IN > :: ele_type in_ele_type ;
  typedef typename vector_traits < OUT > :: ele_type out_ele_type ;
  
  // 'synthetic' types for input and output. These are always TinyVectors,
  // possibly of only one element, of the elementary type of in_type/out_type.
  // On top of providing a uniform container type (the TinyVector) the
  // synthetic type is also 'unaware' of any specific meaning the 'true'
  // input/output type may have, and arithmetic operations on the synthetic
  // types won't clash with arithmetic defined for the 'true' types.

  typedef zimt::xel_t < in_ele_type , dim_in > in_nd_ele_type ;
  typedef zimt::xel_t < out_ele_type , dim_out > out_nd_ele_type ;
  
  // for vectorized operation, we need a few extra typedefs. I use a _v
  // suffix instead of the _type suffix above to indicate vectorized types.
  // If vsize is 1, the _v types simply collapse to the unvectorized
  // types, having them does no harm, but it's not safe to assume that,
  // for example, in_v and in_type are in fact different types.

  /// a simdized type of the elementary type of result_type,
  /// which is used for coefficients and results. this is fixed via
  /// the traits class vector_traits (in vector.h). Note how we derive
  /// this type using vsize from the template argument, not what
  /// vector_traits deems appropriate for ele_type - though
  /// both numbers will be the same in most cases.
  
  typedef typename vector_traits < IN , vsize > :: ele_v in_ele_v ;
  typedef typename vector_traits < OUT , vsize > :: ele_v out_ele_v ;
  
  // 'synthetic' types for simdized input and output. These are always
  // TinyVectors, possibly of only one element, of the simdized input
  // and output type.

  typedef typename vector_traits < IN , vsize > :: nd_ele_v in_nd_ele_v ;
  typedef typename vector_traits < OUT , vsize > :: nd_ele_v out_nd_ele_v ;
  
  /// vectorized in_type and out_type. vector_traits supplies these
  /// types so that multidimensional/multichannel data come as zimt::xel_ts,
  /// while 'singular' data won't be made into TinyVectors of one element.
  
  typedef typename vector_traits < IN , vsize > :: type in_v ;
  typedef typename vector_traits < OUT , vsize > :: type out_v ;

  /// vsize wide vector of ints, used for gather/scatter indexes
  
  typedef typename vector_traits < int , vsize > :: ele_v ic_v ;

} ;

/// vs_adapter wraps a unary_functor to produce a functor which is
/// compatible with the wielding code. This is necessary, because zimt's
/// unary_functors take 'naked' arguments if the data are single-channel,
/// while the wielding code always passes xel_t. The operation of this
/// wrapper class should not have a run-time effect; it's simply converting
/// references. Note here that unary_functor is just an empty
/// shell with a bunch of type declarations and constants: it's there
/// as a base class for user code functors, to provide a uniform
/// interface and to facilitate the coding of the 'simdized xel'
/// types which zimt uses.
/// While it would be nice to simply pass through the unwrapped
/// unary_functor, this would force us to deal with the distinction
/// between data in xel_t and 'naked' fundamentals deeper down in the
/// code, and here is a good central place where we can route to uniform
/// access via xel_t - possibly with only one element. We rely on the
/// optimizer to optimize the xel_t of one value_type - use of that
/// construct is only for syntactic uniformity.
/// Note that the wielding code uses *only* the vectorized eval member
/// function of inner_type. The scalar version may be present but it
/// won't be called by the wielding code. If inner_type is to do a
/// reduction - e.g. cumulate statistics as it is repeatedly applied
/// during the transform - it must implement a separate eval member
/// function taking a third argument, 'cap', which is invoked if a
/// processing thread has odd values left over at the end of it's
/// 'run' which aren't enough to fill an entire vector. The wielding
/// code will pass input to this eval member function which is
/// appropriately padded, so the invocation of the vectorized eval
/// is safe, but the cap value will indicate how many of the vector's
/// lanes hold 'genuine' data - the rest are padding, repeating other
/// 'genuine' values. The padding is applied so that functors
/// which ignore 'cap' (this should be the majority) can do so without
/// causing exceptions due to unsuitable input, provided that the
/// 'genuine' lanes don't cause such exceptions.
/// If the wrapped functor, 'inner_type', does not have a capped eval
/// overload, vs_adapter will route to inner_type's 'uncapped' eval,
/// ignoring the cap value.
/// This mode of processing 'odd bits' may seem wasteful, but I think
/// that - compared to the usual 'segment sizes' used in zimt, which
/// are several hundred per 'run' - and many af them usually without
/// any odd bits (the segment size is a power of two, odd bits only
/// occur at the end of an entire line, if at all) - the extra effort
/// used for padding is quite negligible. I may add a separate code
/// path for segments with no odd bits if this assumption turns out
/// wrong. For now I am content with a clean, uniform design which
/// avoids even the possibility of exceptions due to unpopulated
/// SIMD lanes when processing the last batch of data in a run, and
/// makes it possible to process all data with SIMD code, rather than
/// performing a separate 'peeling' run with scalar code.

template < class inner_type >
struct vs_adapter
: public inner_type
{
  using typename inner_type::in_ele_v ;
  using typename inner_type::out_ele_v ;

  typedef typename inner_type::in_nd_ele_type in_type ;
  typedef typename inner_type::out_nd_ele_type out_type ;
  typedef typename inner_type::in_nd_ele_v in_v ;
  typedef typename inner_type::out_nd_ele_v out_v ;

  vs_adapter ( const inner_type & _inner )
  : inner_type ( _inner )
  { } ;

  void eval ( const in_v & in ,
                   out_v & out )
  {
    inner_type::eval
      ( reinterpret_cast < const typename inner_type::in_v & > ( in ) ,
        reinterpret_cast < typename inner_type::out_v & > ( out ) ) ;
  }

  // does inner_type have a capped eval function? If so, the adapter
  // will dispatch to it, otherwise it will call eval without cap.
  // Most of the time, inner_type won't have a capped variant - this
  // is only useful for reductions, all evaluations inside the
  // wiedling code are coded so that all vectors passed to eval are
  // padded if necessary and therefore 'technically' full and safe
  // to process. The cap is only a hint that some of the lanes were
  // generated by padding wih the last 'genuine' lane, but a reduction
  // needs to be aware of the fact and ignore the lanes filled with
  // padding. An alternative to this method would be the use of
  // masked code, but this would require additional coding effort
  // to handle the masked/capped vectors, and I much prefer to keep
  // the code as simple and straightforward as I can.

  static const bool has_capped_eval = inner_type::has_capped_eval ;

  void _eval ( std::true_type ,
               const in_v & in ,
                    out_v & out ,
               const std::size_t & cap )
  {
    inner_type::eval
      ( reinterpret_cast < const typename inner_type::in_v & > ( in ) ,
        reinterpret_cast < typename inner_type::out_v & > ( out ) ,
        cap ) ;
  }

  void _eval ( std::false_type ,
               const in_v & in ,
                    out_v & out ,
               const std::size_t & cap )
  {
    inner_type::eval
      ( reinterpret_cast < const typename inner_type::in_v & > ( in ) ,
        reinterpret_cast < typename inner_type::out_v & > ( out ) ) ;
  }

public:

  // the adapter's capped eval will invoke inner_type's capped eval
  // if it's present, and 'normal' uncapped eval otherwise.

  void eval ( const in_v & in ,
                   out_v & out ,
              const std::size_t & cap )
  {
    _eval ( std::integral_constant < bool , has_capped_eval >() ,
            in , out , cap ) ;
  }

} ;

/// class chain_type is a helper class to pass one unary functor's
/// result as argument to another one. We rely on T1 and T2 to
/// provide a few of the standard types used in unary functors.
/// Typically, T1 and T2 will both be unary_functors, but
/// the type requirements could also be fulfilled 'manually'.
/// Internally, the functors are wrapped in vs_adapters, to
/// provide a capped eval overload if T1 or T2 don't provide one.
/// The wrapping also takes care of functors processing plain
/// fundamentals.

template < typename T1 ,
           typename T2 >
struct chain_type
: public unary_functor < typename T1::in_type ,
                               typename T2::out_type ,
                               T1::vsize >
{
  // definition of base_type

  enum { vsize = T1::vsize } ;

  typedef unary_functor < typename T1::in_type ,
                                typename T2::out_type ,
                                vsize > base_type ;

  using typename base_type::in_type ;
  using typename base_type::out_type ;
  using typename base_type::in_v ;
  using typename base_type::out_v ;

  // we require both functors to share the same vectorization width

  static_assert ( T1::vsize == T2::vsize ,
                  "can only chain unary_functors with the same vector width" ) ;

   // hold the two functors by value

  vs_adapter < T1 > t1 ;
  vs_adapter < T2 > t2 ;

  // the constructor initializes them

  chain_type ( const T1 & _t1 , const T2 & _t2 )
  : t1 ( _t1 ) ,
    t2 ( _t2 )
    { } ;

  // intermediate_v is the first functor's out_v and the
  // second functor's in_v

  typedef typename vs_adapter<T1>::in_v lhs_v ;
  typedef typename vs_adapter<T1>::out_v intermediate_v ;
  typedef typename vs_adapter<T2>::out_v rhs_v ;

  static_assert ( std::is_same
                   < typename vs_adapter<T1>::out_v ,
                     typename vs_adapter<T2>::in_v
                   > :: value ,
                  "chain: output of first functor must match input of second functor" ) ;

  void eval ( const in_v & arg ,
                    out_v & result )
  {
    const auto & lhs = reinterpret_cast < const lhs_v & > ( arg ) ;
    auto & rhs = reinterpret_cast < rhs_v & > ( result ) ;

    intermediate_v intermediate ;
    // evaluate first functor into intermediate
    t1.eval ( lhs , intermediate ) ;
    // feed it as input to second functor
    t2.eval ( intermediate , rhs ) ;
  }

  // if the participating functors don't have capped eval overloads,
  // t1 and t2 will have them, because they are 'wrapped' versions
  // made with vs_adapter. So we can always provide a capped eval
  // overload here:

  void eval ( const in_v & arg ,
                    out_v & result ,
              const std::size_t & cap )
  {
    const auto & lhs = reinterpret_cast < const lhs_v & > ( arg ) ;
    auto & rhs = reinterpret_cast < rhs_v & > ( result ) ;

    intermediate_v intermediate ;
    // evaluate first functor into intermediate
    t1.eval ( lhs , intermediate , cap ) ;
    // feed it as input to second functor
    t2.eval ( intermediate , rhs , cap ) ;
  }
} ;

/// chain is a factory function yielding the result of chaining
/// two unary_functors.

template < class T1 , class T2 >
chain_type < T1 , T2 >
chain ( const T1 & t1 , const T2 & t2 )
{
  return chain_type < T1 , T2 > ( t1 , t2 ) ;
}

/// using operator overloading, we can exploit operator+'s semantics
/// to chain several unary functors. We need to specifically enable
/// this for types derived from unary_functor_tag to avoid a catch-all
/// situation.

template < typename T1 ,
           typename T2 ,
           typename enable = typename
             std::enable_if
             <    std::is_base_of
                  < unary_functor_tag < T2::vsize > ,
                    T1
                  > :: value
               && std::is_base_of
                  < unary_functor_tag < T1::vsize > ,
                    T2
                  > :: value
             > :: type
         >
chain_type < T1 , T2 >
operator+ ( const T1 & t1 , const T2 & t2 )
{
  return chain ( t1 , t2 ) ;
}

/// class grok_type is a helper class wrapping a unary_functor
/// so that it's type becomes opaque - a technique called 'type
/// erasure', here applied to unary_functors with their specific
/// capability of providing both uncapped and capped operation
/// in one common object.
///
/// While 'grokking' a unary_functor may degrade performance slightly,
/// the resulting type is less complex, and when working on complex
/// constructs involving several unary_functors, it can be helpful to
/// wrap the whole bunch into a grok_type for some time to make compiler
/// messages more palatable. I even suspect that the resulting functor,
/// which simply delegates to std::functions, may optimize better at
/// times than a more complex functor in the 'grokkee'.
///
/// Performance aside, 'grokking' a unary_functor produces a
/// simple, consistent type that can hold *any* unary_functor with the
/// given input type, output type and lane count, so it allows to hold
/// and use a variety of (intrinsically differently typed) functors at
/// runtime via a common handle which is a unary_functor itself
/// and can be passed to zimt::process and the functions of the
/// transform family.
/// With unary_functors being first-class, copyable objects, this also
/// makes it possible to pass around unary_functors between different
/// TUs where user code can provide new functors at will which can
/// simply be used without having to recompile to make their type
/// known, at the cost of a call through a std::function, which may be
/// higher than when the code is 'visible' to the compiler during
/// compilation, which allows for better optimization.
///
/// Another aspect to grokking is keeping the binary compact. Suppose you
/// have an operation which is polymorphic with several template args.
/// When you instantiate all possible combinations, you end up with 
/// a large number of types, each with it's own binary representation.
/// Type erasure can significantly reduce this effect by bundling
/// a whole set of types into a single type. This is especially welcome
/// when working with functor composition: You can build up a variety of
/// functors addressing a specific aspect of your functionality and
/// 'grok' the result, so you only have one type left, which is much
/// easier to handle: you can pass it around and introduce it's type
/// into other parts of your code without having to deal with the
/// complexity of the original grokked type family. When an actual
/// program is compiled, chances are you'll not exploit the full range
/// of template instantiations, and the compiler will optimize the code
/// as if the 'grok' had not happened, because it can 'see inside' the
/// grok_type object and optimize the grokking overhead away.
///
/// Why not use a base class with virtual 'eval' functions? While this
/// is feasible to implement the evaluation, there are additional aspects
/// to consider: we want the grok_type object to be copyable. So we'd
/// have to implement a mechanism similar to the one we use here: the
/// 'rep' member function. We'd also need a common base class declaring
/// the pure virtual eval member functions, and all unary_functors
/// would need to inherit from it, forcing them to have a virtual
/// function table. I reckon that all in all a proper implementation
/// of class grok_type using virtual eval member functions would end
/// up about as complex as this one, but it would be more intrusive.
/// The method used here is perfectly general and transparent, and it
/// requires no 'coorperation' on the part of the grokkee.

template < typename IN ,       // argument or input type
           typename OUT = IN , // result type
           size_t _vsize = vector_traits < IN > :: size
         >
struct grok_type
: private grok_t ,
  public unary_functor < IN , OUT , _vsize >
{
  typedef unary_functor < IN , OUT , _vsize > base_type ;

  using base_type::vsize ;
  using base_type::dim_in ;
  using base_type::dim_out ;

  using typename base_type::in_type ;
  using typename base_type::out_type ;
  using typename base_type::in_v ;
  using typename base_type::out_v ;

private:

  // 'inner workings' of grok_type, which we keep private

  // define the types for the std::function we will use to
  // wrap the grokkee's evaluation code in. First the eval
  // function for full vectors:

  typedef std::function
    < void ( void * & , const in_v & , out_v & )
    > v_eval_type ;

  // capped evaluation has an additional 'cap' argument, giving the
  // number of 'genuine' lanes, while the rest are duplicates of
  // other lanes to provide valid input and full vectors. The 'cap'
  // value is rarely relevant: only reductions depend on it for
  // correct output, because they need to keepe the duplicated
  // lanes used to fill up the vectors from entring their results.

  typedef std::function
    < void ( void * & , const in_v & , out_v & ,
             const std::size_t & cap )
    > c_eval_type ;

  // these are the class members holding the std::functions:

  v_eval_type _v_ev ;
  c_eval_type _c_ev ;


public:

  /// we provide a default constructor so we can create an empty
  /// grok_type and assign to it later. Calling the empty grok_type's
  /// eval will result in an exception, but we want this to set up
  /// things like arrays of grok_type.

  grok_type() { } ;
  
  /// constructor from 'grokkee' using lambda expressions
  /// to initialize the std::functions above. we enable this if
  /// grokkee_type is a unary_functor. We may relax the
  /// requirement and accept anything that 'fits'.

  template < class grokkee_type ,
             typename std::enable_if
              < std::is_base_of
                < unary_functor_tag < vsize > ,
                  grokkee_type
                > :: value ,
                int
              > :: type = 0
            >
  grok_type ( grokkee_type grokkee )
  : grok_t ( grokkee )
  {
    // use vs_adapter to create a functor with an uncapped and
    // a capped eval function, even if 'grokkee_type' does not
    // have a capped variant: in this case, the uncapped variant
    // will be invoked and the cap value is ignored. Most of the
    // time, 'act' functors don't need a separate 'capped' eval
    // overload - the exception being reductions. With the code
    // to route the call to the capped overload to the uncapped
    // overload if the capped overload is missing, we gain ease
    // of use, but we need to be aware of the fact that reductions
    // must provide a capped overload.
    // The 'context' is stored as a void pointer, and this void
    // pointer, which will be distinct for every copy of the
    // grokked functor, is passed to the lambdas when they are
    // invoked. This ensures the individuality of each copy of
    // a grok_type object, and at the same time it ensures that
    // within a specific grok_type object, all member functions
    // work with the same grokkee. If we were to put p_context
    // 'into' the lambdas right here, all copies would instead
    // refer to the same grokkee - the one we generate here.

    typedef vs_adapter < grokkee_type > g_t ;

    // now initialize the class members holding std::functions
    // with lambdas taking first the context, then more arguments.
    // This is how we delegate to the grokkee: the lambdas
    // static_cast the context pointer to the grokkee's type
    // (which is known in this contructor because the argument
    // is typed so), and then they proceed to delegate the
    // call to the grokkee. Replication and termination use
    // the same pattern. Once the lambdas are stored in the
    // std::functions _e_ev, _c_ev etc. the 'knowledge' of
    // how to delegate is removed from view: the grokkee's type
    // is no longer visible or accessible, hence the term 'type
    // erasure'. Of course the *compiler* can still track what
    // is going on, and if the grokkee's class definition is
    // accessible during compilation of the grok, it can
    // optimize it just as if the grokkee had been used
    // instead, and there is no overhead. On the other hand,
    // if the grokkee's class definition is elsewhere, the
    // mechanism still works, but the context pointer and the
    // delegation won't be optimized away during compilation.
    // Even such constructs may be optimized away during
    // run-time when the code is JITed or submitted to similar
    // tuning mechanisms.

    _v_ev = [] ( void * p_ctx , const in_v & in , out_v & out )
            {
              static_cast<g_t*> ( p_ctx ) -> eval ( in , out ) ;
            } ;

    _c_ev = [] ( void * p_ctx , const in_v & in , out_v & out ,
                 const std::size_t & cap )
            {
              static_cast<g_t*> ( p_ctx ) -> eval ( in , out , cap ) ;
            } ;
  } ;

  // the eval member functions pass the grok_type object's 'own'
  // p_context to the lambdas which are captured in the members
  // _v_ev etc., stored as std::functions.

  // uncapped evaluation member function

  void eval ( const in_v & i , out_v & o )
  {
    _v_ev ( p_context , i , o ) ;
  }

  void eval ( const in_type & i , out_type & o )
  {
    in_v iv ( i ) ;
    out_v ov ;
    eval ( iv , ov ) ;
    o = ov[0] ;
  }

  // capped evaluation function template

  void eval ( const in_v & i , out_v & o , const std::size_t & cap )
  {
    _c_ev ( p_context , i , o , cap ) ;
  }
} ;

/// grok() is the corresponding factory function, wrapping grokkee
/// in a grok_type. Because the grok_type object is quite a
/// mouthful, this nice-to-have: due to ATD, if you have some
/// unary_functor f, all you need is "auto g = zimt::grok(f);"
/// The resulting functor g can then be used instead of f, filling
/// the same syntactic slot, since it's a unary_functor itself,
/// with the same signature.

template < class grokkee_type >
grok_type < typename grokkee_type::in_type ,
                  typename grokkee_type::out_type ,
                  grokkee_type::vsize >
grok ( grokkee_type grokkee )
{
  return grok_type < typename grokkee_type::in_type ,
                           typename grokkee_type::out_type ,
                           grokkee_type::vsize >
                  ( grokkee ) ;
}

/// amplify_type amplifies it's input with a factor. If the data are
/// multi-channel, the factor is multi-channel as well and the channels
/// are amplified by the corresponding elements of the factor.
/// I added this class to make work with integer-valued splines more
/// comfortable - if these splines are prefiltered with 'boost', the
/// effect of the boost has to be reversed at some point, and amplify_type
/// does just that when you use 1/boost as the 'factor'.

template < class _in_type ,
           class _out_type = _in_type ,
           class _math_type = _in_type ,
           size_t _vsize = vector_traits < _in_type > :: vsize >
struct amplify_type
: public unary_functor < _in_type , _out_type , _vsize >
{
  typedef unary_functor < _in_type , _out_type , _vsize > base_type ;
  
  enum { vsize = _vsize } ;
  enum { dimension = base_type::dim_in } ;
  
  // TODO: might assert common dimensionality
  
  using typename base_type::in_type ;
  using typename base_type::out_type ;
  using typename base_type::in_v ;
  using typename base_type::out_v ;
  using typename base_type::in_nd_ele_v ;
  using typename base_type::out_nd_ele_v ;

  typedef _math_type math_type ;
  
  // typedef typename vigra::ExpandElementResult < math_type > :: type
  //   math_ele_type ;
  typedef typename math_type::value_type math_ele_type ;

  typedef zimt::xel_t < math_ele_type , dimension > math_nd_ele_type ;
  
  typedef typename vector_traits < math_ele_type , vsize > :: type
    math_ele_v ;
  
  const math_type factor ;
  
  // constructors initialize factor. If dimension is greater than 1,
  // we have two constructors, one taking a TinyVector, one taking
  // a single value for all dimensions.

  template < typename = std::enable_if < ( dimension > 1 ) > >
  amplify_type ( const math_type & _factor )
  : factor ( _factor )
  { } ;
  
  amplify_type ( const math_ele_type & _factor )
  : factor ( _factor )
  { } ;
  
  void eval ( const in_type & in , out_type & out )
  {
    out = out_type ( math_type ( in ) * factor ) ;
  }

  template < typename = std::enable_if < ( vsize > 1 ) > >
  void eval ( const in_v & in , out_v & out )
  {
    // we take a view to the arguments as TinyVectors, even if
    // the data are 'singular'

    const in_nd_ele_v & _in
      = reinterpret_cast < in_nd_ele_v const & > ( in ) ;

    const math_nd_ele_type & _factor
      = reinterpret_cast < math_nd_ele_type const & > ( factor ) ;

    out_nd_ele_v & _out
      = reinterpret_cast < out_nd_ele_v & > ( out ) ;

    // and perform the application of the factor element-wise

    for ( int i = 0 ; i < dimension ; i++ )
      assign ( _out[i] , math_ele_v ( _in[i] ) * _factor[i] ) ;
  }

} ;

/// flip functor produces it's input with component order reversed.
/// This can be used to deal with situations where coordinates in
/// the 'wrong' order have to be fed to a functor expecting the opposite
/// order and should be a fast way of doing so, since the compiler can
/// likely optimize it well.
/// I added this class to provide simple handling of incoming NumPy
/// coordinates, which are normally in reverse order of vigra coordinates

template < typename _in_type ,
           size_t _vsize = vector_traits < _in_type > :: vsize >
struct flip
: public unary_functor < _in_type , _in_type , _vsize >
{
  typedef unary_functor < _in_type , _in_type , _vsize > base_type ;
                     
  enum { vsize = _vsize } ;
  enum { dimension = base_type::dim_in } ;
  
  using typename base_type::in_type ;
  using typename base_type::out_type ;
  using typename base_type::in_v ;
  using typename base_type::out_v ;
  using typename base_type::in_nd_ele_type ;
  using typename base_type::out_nd_ele_type ;
  using typename base_type::in_nd_ele_v ;
  using typename base_type::out_nd_ele_v ;
  
  void eval ( const in_type & in_ , out_type & out )
  {
    // we need a copy of 'in' in case _in == out
    
    in_type in ( in_ ) ;
    
    // we take a view to the arguments as TinyVectors, even if
    // the data are 'singular'

    const in_nd_ele_type & _in
      = reinterpret_cast < in_nd_ele_type const & > ( in ) ;
      
    out_nd_ele_type & _out
      = reinterpret_cast < out_nd_ele_type & > ( out ) ;
      
    for ( int e = 0 ; e < dimension ; e++ )
      _out [ e ] = _in [ dimension - e - 1 ] ;
  }
  
  template < typename = std::enable_if < ( vsize > 1 ) > >
  void eval ( const in_v & in_ , out_v & out )
  {
    // we need a copy of 'in' in case _in == out
    
    in_v in ( in_ ) ;
    
    // we take a view to the arguments as TinyVectors, even if
    // the data are 'singular'

    const in_nd_ele_v & _in
      = reinterpret_cast < in_nd_ele_v const & > ( in ) ;
      
    out_nd_ele_v & _out
      = reinterpret_cast < out_nd_ele_v & > ( out ) ;
    
    for ( int e = 0 ; e < dimension ; e++ )
      assign ( _out [ e ] , _in [ dimension - e - 1 ] ) ;
  }
  
} ;

/// at times we require reading access to an nD array at given coordinates,
/// as a functor which, receiving the coordinates, produces the values
/// from the array. In the scalar case, this is trivial: if the coordinate
/// is integral, we have a simple indexed access, and if it is real, we
/// can use std::round to produce a nearby discrete coordinate.
/// But for the vectorized case we need a bit more effort: We need to
/// translate the access with a vector of coordinates into a gather
/// operation. We start out with a generalized template class 'yield-type':

template < typename crd_t ,
           typename data_t ,
           size_t _vsize = vector_traits < data_t > :: size ,
           class enable = void
         >
struct yield_type
{ } ;

/// First, we specialize for integral coordinates

template < typename T >
using crd_integral =
  typename std::conditional
           < std::is_fundamental<T>::value ,
             std::is_integral<T> ,
             std::is_integral<typename T::value_type>
           > :: type ;

template < typename crd_t ,
           typename data_t ,
           size_t _vsize >
struct yield_type
   < crd_t , data_t , _vsize ,
     typename std::enable_if
       < crd_integral < crd_t > :: value > :: type >
: public unary_functor < crd_t , data_t , _vsize >
{
  typedef unary_functor < crd_t  ,
                          data_t ,
                         _vsize > base_type ;

  enum { vsize = _vsize } ;
  enum { dimension = base_type::dim_in } ;
  enum { channels = base_type::dim_out } ;
  using typename base_type::in_type ;
  using typename base_type::out_type ;
  using typename base_type::in_ele_type ;
  using typename base_type::out_ele_type ;
  using typename base_type::in_v ;
  using typename base_type::out_v ;

  typedef typename std::integral_constant
                     < bool , ( dimension > 1 ) > :: type is_nd_t ;

  typedef typename std::integral_constant
                     < bool , ( channels > 1 ) > :: type is_mc_t ;

  typedef zimt::view_t < dimension , out_type > array_type ;
  const array_type & data ;

  yield_type ( const array_type & _data )
  : data ( _data )
  { }

private:

  // scalar operation: simple indexed access. This is dispatched to via
  // the third argument, and for the scalar case, the number of dimensions
  // and channels is irrelevant: the array's operator[] knows what to do.

  template < typename d_t , typename c_t >
  void eval ( const in_type & crd , out_type & v ,
              std::true_type ,
              d_t ,
              c_t
            )
  {
    v = data [ crd ] ;
  }

  // the next four overloads deal with vectorized access.
  // first variant: coordinate is 1D, value is single-channel

  void eval ( const in_v & crd , out_v & v ,
              std::false_type ,
              std::false_type ,
              std::false_type )
  {
    auto ofs = crd * int(data.stride(0)) ;

    data_t * p_src = data.data() ;
    v.gather ( p_src , ofs ) ;
  }

  // second variant: coordinate is 1D, value is multi-channel

  void eval ( const in_v & crd , out_v & v ,
              std::false_type ,
              std::false_type ,
              std::true_type )
  {
    auto ofs = crd * int(data.stride(0)) ;
    ofs *= channels ;

    out_ele_type * p_src = (out_ele_type*) ( data.data() ) ;
    for ( int ch = 0 ; ch < channels ; ch++ )
    {
      v[ch].gather ( p_src + ch , ofs ) ;
    }
  }

  // third variant: coordinate is nD, value is single-channel

  void eval ( const in_v & crd , out_v & v ,
              std::false_type ,
              std::true_type ,
              std::false_type )
  {
    auto ofs = crd[0] * int(data.stride(0)) ;
    for ( int d = 1 ; d < dimension ; d++ )
    {
      ofs += crd[d] * int(data.stride(d)) ;
    }

    out_ele_type * p_src = (out_ele_type*) ( data.data() ) ;
    v.gather ( p_src , ofs ) ;
  }

  // fourth variant: coordinate is nD, value is multi-channel

  void eval ( const in_v & crd , out_v & v ,
              std::false_type ,
              std::true_type ,
              std::true_type )
  {
    auto ofs = crd[0] * int(data.stride(0)) ;
    for ( int d = 1 ; d < dimension ; d++ )
    {
      ofs += crd[d] * int(data.stride(d)) ;
    }
    ofs *= channels ;

    out_ele_type * p_src = (out_ele_type*) ( data.data() ) ;
    for ( int ch = 0 ; ch < channels ; ch++ )
    {
      v[ch].gather ( p_src + ch , ofs ) ;
    }
  }

public:

  // this is the dispatching function:

  template < typename in_t , typename out_t >
  void eval ( const in_t & crd , out_t & v )
  {
    typedef typename std::is_same < out_t , out_type > :: type is_scalar_t ;
    eval ( crd , v , is_scalar_t() , is_nd_t() , is_mc_t() ) ; 
  }

} ;
           
/// Next, we specialize for real coordinates

template < typename crd_t ,
           typename data_t ,
           size_t _vsize >
struct yield_type
   < crd_t , data_t , _vsize ,
     typename std::enable_if
       < ! crd_integral < crd_t > :: value > :: type >
: public unary_functor < crd_t , data_t , _vsize >
{
  typedef unary_functor < crd_t  ,
                          data_t ,
                         _vsize > base_type ;

  enum { vsize = _vsize } ;
  enum { dimension = base_type::dim_in } ;
  enum { channels = base_type::dim_out } ;
  using typename base_type::in_type ;
  using typename base_type::out_type ;
  using typename base_type::in_ele_type ;
  using typename base_type::out_ele_type ;
  using typename base_type::in_v ;
  using typename base_type::out_v ;

  typedef typename std::integral_constant
                     < bool , ( dimension > 1 ) > :: type is_nd_t ;

  typedef zimt::view_t < dimension , out_type > array_type ;

  typedef typename std::conditional
                   < dimension == 1 ,
                     int ,
                     zimt::xel_t < int , dimension >
                   > :: type ic_t ;

  typedef yield_type < ic_t , data_t , _vsize > iy_t ;

  const iy_t iy ;
  
  yield_type ( const array_type & _data )
  : iy ( _data )
  { }

private:

  // The implementation is simple: we round the coordinate to int and
  // delegate to the int version

  // scalar operation: simple indexed access, for 1D and nD cases

  void eval ( const in_type & crd , out_type & v ,
              std::true_type ,
              std::false_type )
  {
    iy.eval ( std::round ( crd ) , v ) ;
  }

  void eval ( const in_type & crd , out_type & v ,
              std::true_type ,
              std::true_type )
  {
    zimt::xel_t < int , dimension > icrd ;
    for ( int d = 0 ; d < dimension ; d++ )
      icrd [ d ] = std::round ( crd [ d ] ) ;
    iy.eval ( icrd , v ) ;
  }

  // vectorized operation, for 1D and nD cases

  void eval ( const in_v & crd , out_v & v ,
              std::false_type ,
              std::false_type )
  {
    typename iy_t::in_v icrd ( round ( crd ) ) ;
    iy.eval ( icrd , v ) ;
  }

  void eval ( const in_v & crd , out_v & v ,
              std::false_type ,
              std::true_type )
  {
    typename iy_t::in_v icrd ;
    for ( int d = 0 ; d < dimension ; d++ )
      icrd [ d ] = round ( crd [ d ] ) ;
    iy.eval ( icrd , v ) ;
  }

public:

  template < typename in_t , typename out_t >
  void eval ( const in_t & crd , out_t & v )
  {
    typedef typename std::is_same < out_t , out_type > :: type is_scalar_t ;
    eval ( crd , v , is_scalar_t() , is_nd_t() ) ; 
  }

} ;

/// class uf_adapter 'bends' a unary functor to a unary_functor
/// handling fundamentals or xel_t arguments. For now this is mainly
/// to adapt vspline code and it blindly assumes that the arguments
/// are binary-compatible.
/// the previous implementation produced 'synthetic' in/out types
/// which were all xel_t, but now the switch from fundamentals to
/// xel_t of one T is made in zimt_process, so the in/out types
/// of the functor can be preserved as fundamentals if they come
/// in that way. So this adapter passes the wrappee's in/out
/// types unchanged if they are single-channel, and produce
/// a xel_t if they are multi-channel.

template < typename W >
struct uf_adapter
{
  // set up the type system zimt expects in a unary_functor

  static const int vsize = W::vsize ;
  static const int dim_in = W::dim_in ;
  static const int dim_out = W::dim_out ;

  typedef typename W::in_ele_type in_ele_type ;
  typedef typename W::out_ele_type out_ele_type ;

  typedef zimt::xel_t < in_ele_type , dim_in > in_nd_ele_type ;
  typedef zimt::xel_t < out_ele_type , dim_out > out_nd_ele_type ;

  typedef typename std::conditional
    < dim_in == 1 ,
      typename W::in_type ,
      in_nd_ele_type > :: type in_type ;

  typedef typename std::conditional
    < dim_out == 1 ,
      typename W::out_type ,
      out_nd_ele_type > :: type out_type ;

  typedef typename vector_traits < in_ele_type , vsize >
                     :: ele_v in_ele_v ;

  typedef typename vector_traits < out_ele_type , vsize >
                     :: ele_v out_ele_v ;

  typedef typename vector_traits < in_type , vsize >
                     :: nd_ele_v in_nd_ele_v ;

  typedef typename vector_traits < out_type , vsize >
                     :: nd_ele_v out_nd_ele_v ;

  typedef typename std::conditional
    < dim_in == 1 ,
      typename W::in_v ,
      in_nd_ele_v > :: type in_v ;

  typedef typename std::conditional
    < dim_out == 1 ,
      typename W::out_v ,
      out_nd_ele_v > :: type out_v ;

  // accomodate the wrappee

  W inner ;

  uf_adapter ( const W & _inner )
  : inner ( _inner )
  { }

  // two eval overloads - we might look at the wrappee and only
  // produce the scalar variant if the wrappee contains one, but
  // for now - wrapping vspline code - the scalar variant is taken
  // for granted, because vspline always provides it.

  void eval ( const in_type & in ,
                   out_type & out )
  {
    inner.eval
      ( reinterpret_cast < const typename W::in_type & > ( in ) ,
        reinterpret_cast < typename W::out_type & > ( out ) ) ;
  }

// private:

  // // To detect if a unary functor has a capped eval overload, I use code
  // // adapted from from Valentin Milea's anser to
  // // https://stackoverflow.com/questions/87372/check-if-a-class-has-a-member-function-of-a-given-signature
  // // when providing a capped eval overload of your eval function, please
  // // note that it will only be recognized if one of the three signatures
  // // tested below is matched precisely. Providing a template which would
  // // match the signature does *not* work.
  //
  // template < class C >
  // class has_capped_eval_t
  // {
  //     typedef typename C::in_v in_v ;
  //     typedef typename C::out_v out_v ;
  //
  //     // we allow the cap to be passed as const&, &, and by value,
  //     // the remainder of the signature is 'as usual'
  //
  //     template < class T , class cap_t >
  //     static std::true_type testSignature
  //       (void (T::*)(const in_v&, out_v&, const cap_t&));
  //
  //     template < class T , class cap_t >
  //     static std::true_type testSignature
  //       (void (T::*)(const in_v&, out_v&, cap_t&));
  //
  //     template < class T , class cap_t >
  //     static std::true_type testSignature
  //       (void (T::*)(const in_v&, out_v&, cap_t));
  //
  //     template <class T>
  //     static decltype(testSignature(&T::eval)) test(std::nullptr_t);
  //
  //     template <class T>
  //     static std::false_type test(...);
  //
  // public:
  //     using type = decltype(test<C>(nullptr));
  //     static const bool value = type::value;
  // };
  //
  // does inner_type have a capped eval function? If so, the adapter
  // will dispatch to it, otherwise it will call eval without cap.
  // Most of the time, inner_type won't have a capped variant - this
  // is only useful for reductions, all evaluations inside the
  // wiedling code are coded so that all vectors passed to eval are
  // padded if necessary and therefore 'technically' full and safe
  // to process. The cap is only a hint that some of the lanes were
  // generated by padding wih the last 'genuine' lane, but a reduction
  // needs to be aware of the fact and ignore the lanes filled with
  // padding. An alternative to this method would be the use of
  // masked code, but this would require additional coding effort
  // to handle the masked/capped vectors, and I much prefer to keep
  // the code as simple and straightforward as I can.

  typedef W inner_type ;

  static const bool has_capped_eval = inner_type::has_capped_eval ;

  void _eval ( std::true_type ,
               const in_v & in ,
                    out_v & out ,
               const std::size_t & cap )
  {
    inner.eval
      ( reinterpret_cast < const typename inner_type::in_v & > ( in ) ,
        reinterpret_cast < typename inner_type::out_v & > ( out ) ,
        cap ) ;
  }

  void _eval ( std::false_type ,
               const in_v & in ,
                    out_v & out ,
               const std::size_t & cap )
  {
    inner.eval
      ( reinterpret_cast < const typename inner_type::in_v & > ( in ) ,
        reinterpret_cast < typename inner_type::out_v & > ( out ) ) ;
  }

public:

  // the adapter's capped eval will invoke inner_type's capped eval
  // if it's present, and 'normal' uncapped eval otherwise.

  void eval ( const in_v & in ,
                   out_v & out ,
              const std::size_t & cap )
  {
    _eval ( std::integral_constant < bool , has_capped_eval >() ,
            in , out , cap ) ;
  }

  void eval ( const in_v & in ,
                   out_v & out )
  {
    inner.eval
      ( reinterpret_cast < const typename W::in_v & > ( in ) ,
        reinterpret_cast < typename W::out_v & > ( out ) ) ;
  }
} ;

// pass_through simply copies the input to the output. This is
// a handy functor if all the 'work' is done by the get_t and/or
// put_t object.

template < typename T , std::size_t N ,
           std::size_t L = vector_traits < T > :: vsize >
struct pass_through
: public unary_functor < zimt::xel_t < T , N > ,
                               zimt::xel_t < T , N > ,
                               L >
{
  template < typename I , typename O >
  void eval ( const I & i , O & o , const std::size_t cap = 0 )
  {
    o = i ;
  }
} ;

static bool once = true ;

template < typename T , std::size_t N ,
           std::size_t L = vector_traits < T > :: vsize >
struct echo
: public unary_functor < zimt::xel_t < T , N > ,
                               zimt::xel_t < T , N > ,
                               L >
{
  template < typename I , typename O >
  void eval ( const I & i , O & o , const std::size_t cap = 0 )
  {
    if ( once )
    {
      std::cout << i << std::endl ;
      once = false ;
    }
    o = i ;
  }
} ;

// factory function to 'wrap' a functor 'inner' in a uf_adapter

template < typename W >
uf_adapter < W >
uf_adapt ( const W & inner )
{
  auto adapted = uf_adapter < W > ( inner ) ;
  return adapted ;
}

END_ZIMT_SIMD_NAMESPACE
HWY_AFTER_NAMESPACE() ;

#endif // ZIMT_UNARY_FUNCTOR_H

