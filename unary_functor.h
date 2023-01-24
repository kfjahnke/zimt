/************************************************************************/
/*                                                                      */
/*    zimt - abstraction layer for SIMD programming                     */
/*                                                                      */
/*            Copyright 2023 by Kay F. Jahnke                           */
/*                                                                      */
/*    The git repository for this software is at                        */
/*                                                                      */
/*    https://bitbucket.org/kfj/zimt                                    */
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

    zimt's evaluation and remapping code relies on a unary functor template
    which is used as the base for zimt::evaluator and also constitutes the
    type of object accepted by most of the functions in transform.h.

    This template produces functors which are meant to yield a single output
    for a single input, where both the input and output types may be single
    types or zimt::xel_ts, and their elementary types may be vectorized.
    The functors are expected to provide methods named eval() which are capable
    of performing the required functionality. These eval routines take both
    their input and output by reference - the input is taken by const &, and the
    output as plain &. The result type of the eval routines is void. While
    such unary functors can be hand-coded, the class template 'unary_functor'
    provides services to create such functors in a uniform way, with a specifc
    system of associated types and some convenience code. Using unary_functor
    is meant to facilitate the creation of the unary functors used in zimt.
    
    Using unary_functor generates objects which can be easily combined into
    more complex unary functors, a typical use would be to 'chain' two
    unary_functors, see class template 'chain_type' below, which also provides
    an example for the use of unary_functor.
    
    class unary_functor takes three template arguments:
    
    - the argument type, IN
    
    - the result type, OUT
    
    - the number of fundamentals (float, int etc.) in a vector, _vsize
    
    The vectorized argument and result type are deduced from IN, OUT and
    _vsize by querying zimt::vector_traits. When using Vc (-DUSE_VC),
    these types will be Vc::SimdArrays if the elementary type can be used
    to form a SimdArray. Otherwise zimt provides a fallback type emulating
    vectorization: zimt::simd_type. This fallback type emulates just enough
    of SimdArray's capabilities to function as a replacement inside zimt's
    body of code.
    
    So where is eval() or operator()? Not in class unary_functor. The actual
    functionality is provided by the derived class. There is deliberately no
    code concerning evaluation in class unary_functor. My initial implementation
    had pure virtual functions to define the interface for evaluation, but this
    required explicitly providing the overloads in the derived class. Simply
    omitting any reference to evaluation allows the derived class to accomplish
    evaluation with a template if the code is syntactically the same for vectorized
    and unvectorized operation. To users of concrete functors inheriting from
    unary_functor this makes no difference. The only drawback is that it's not
    possible to perform evaluation via a base class pointer or reference. But
    this is best avoided anyway because it degrades performance. If the need arises
    to have several unary_functors with the same template signature share a common
    type, there's a mechanism to make the internals opaque by 'grokking'.
    grokking provides a wrapper around a unary_functor which hides it's type,
    zimt::grok_type directly inherits from unary_functor and the only template
    arguments are IN, OUT and _vsize. This hurts performance a little - just as
    calling via a base class pointer/reference would, but the code is outside
    class unary_functor and therefore only activated when needed.
    
    Class zimt::evaluator is itself coded as a zimt::unary_functor and can
    serve as another example for the use of the code in this file.
    
    Before the introduction of zimt::simd_type, vectorization was done with
    Vc or not at all. Now zimt::vector_traits will produce Vc types if
    possible and zimt::simd_type otherwise. This breaks code relying on
    the fallback to scalar without Vc, and it also breaks code that assumes that
    Vc is the sole method of vectorization.
    
    Extant code written for use with Vc should function as before as long as
    USE_VC is defined. It may be possible now to use such code even without Vc.
    This depends on how much of Vc::SimdArray's functionality is used. If such
    code runs without Vc, it may still not perform well and possibly even worse
    than scalar code.
*/

#ifndef VSPLINE_UNARY_FUNCTOR_H
#define VSPLINE_UNARY_FUNCTOR_H

#include <functional>
#include "xel.h"

namespace zimt {

/// we derive all zimt::unary_functors from this empty class, to have
/// a common base type for all of them. This enables us to easily check if
/// a type is a zimt::unary_functor without having to wrangle with
/// unary_functor's template arguments.
 
template < size_t _vsize >
struct unary_functor_tag { } ;

/// class unary_functor provides a functor object which offers a system
/// of types for concrete unary functors derived from it. If vectorization
/// isn't used, this is trivial, but with vectorization in use, we get
/// vectorized types derived from plain IN and OUT via query of
/// zimt::vector_traits.
///
/// class unary_functor itself does not provide operator(), this is left to
/// the concrete functors inheriting from unary_functor. It is expected
/// that the derived classes provide evaluation capability, either as a
/// method template or as (overloaded) method(s) 'eval'. eval is to be coded
/// as taking it's first argument as a const&, and writing it's result to
/// it's second argument, which it receives by reference. eval's return type
/// is void. Inside zimt, classes derived from unary_functor do provide
/// operator(), so instances of these objects can be called with function
/// call syntax as well.
///
/// Why not lay down an interface with a pure virtual function eval()
/// which derived classes would need to override? Suppose you had, in
/// unary_functor,
///
/// virtual void eval ( const in_type & in , out_type & out ) = 0 ;
///
/// Then, in a derived class, you'd have to provide an override with this
/// signature. Initially, this seems reasonable enough, but if you want to
/// implement eval() as a member function template in the derived class, you
/// still would have to provide the override (calling an instantiated version
/// of your template), because your template won't be recognized as a viable
/// way to override the pure virtual base class member function. Since
/// providing eval as a template is common (oftentimes vectorized and
/// unvectorized code are the same) I've decided against having virtual eval
/// routines, to avoid the need for explicitly overriding them in derived
/// classes which provide eval() as a template.
///
/// How about providing operator() in unary_functor? We might add the derived
/// class to the template argument list and use unary_functor with CRP. I've
/// decideded against this and instead provide callability as a mixin to be
/// used as needed. This keeps the complexity of unary_functor-derived objects
/// low, adding the extra capability only where it's deemed appropriate. For
/// the mixin, see class 'callable' further down.
///
/// With no virtual member functions, class unary_functor becomes very simple,
/// which is desirable from a design standpoint, and also makes unary_functors
/// smaller, avoiding the creation of the virtual function table.
///
/// The type system used in unary_functor is taken from zimt::vector_traits,
/// additionally prefixing the types with in_ and out_, for input and output
/// types. The other elements of the type names are the same as in
/// vector_traits.

// KFJ 2020-11-15 it does no harm defining that the default for
// OUT should be IN, and this is a very common situation.

template < typename IN ,       // argument or input type
           typename OUT = IN , // result type
           size_t _vsize = zimt::vector_traits < IN > :: size
         >
struct unary_functor
: public unary_functor_tag < _vsize >
{
  // number of fundamentals in simdized data. If vsize is 1, the vectorized
  // types will 'collapse' to the unvectorized types.

  enum { vsize = _vsize } ;

  // number of dimensions. This may well be different for IN and OUT.

  enum { dim_in = zimt::vector_traits < IN > :: dimension } ;
  enum { dim_out = zimt::vector_traits < OUT > :: dimension } ;
  
  // typedefs for incoming (argument) and outgoing (result) type. These two types
  // are non-vectorized types, like zimt::xel_t < float , 2 >. Since such types
  // consist of elements of the same type, the corresponding vectorized type can be
  // easily automatically determined.
  
  typedef IN in_type ;
  typedef OUT out_type ;
  
  // elementary types of same. we rely on zimt::vector_traits to provide
  // these types.
  
  typedef typename zimt::vector_traits < IN > :: ele_type in_ele_type ;
  typedef typename zimt::vector_traits < OUT > :: ele_type out_ele_type ;
  
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
  /// zimt::vector_traits deems appropriate for ele_type - though
  /// both numbers will be the same in most cases.
  
  typedef typename vector_traits < IN , vsize > :: ele_v in_ele_v ;
  typedef typename vector_traits < OUT , vsize > :: ele_v out_ele_v ;
  
  // 'synthetic' types for simdized input and output. These are always
  // TinyVectors, possibly of only one element, of the simdized input
  // and output type.

  typedef typename vector_traits < IN , vsize > :: nd_ele_v in_nd_ele_v ;
  typedef typename vector_traits < OUT , vsize > :: nd_ele_v out_nd_ele_v ;
  
  /// vectorized in_type and out_type. zimt::vector_traits supplies these
  /// types so that multidimensional/multichannel data come as zimt::xel_ts,
  /// while 'singular' data won't be made into TinyVectors of one element.
  
  typedef typename vector_traits < IN , vsize > :: type in_v ;
  typedef typename vector_traits < OUT , vsize > :: type out_v ;

  /// vsize wide vector of ints, used for gather/scatter indexes
  
  typedef typename vector_traits < int , vsize > :: ele_v ic_v ;

} ;

// KFJ 2018-07-14
// To deal with an issue with cppyy, which has trouble processing
// templated operator() into an overloaded callable, I introduce this
// mixin, which specifically provides two distinct operator() overloads.
// This is also a better way to introduce the callable quality, since on
// the side of the derived class it requires only inheriting from the
// mixin, rather than the verbose templated operator() I used previously.
// This is still experimental.

/// mixin 'callable' is used with CRTP: it serves as additional base to
/// unary functors which are meant to provide operator() and takes the
/// derived class as it's first template argument, followed be the
/// argument types and vectorization width, so that the parameter and
/// return type for operator() and - if vsize is greater than 1 - it's
/// vectorized overload can be produced.
/// This formulation has the advantage of not having to rely on the
/// 'out_type_of' mechanism I was using before and provides precisely
/// the operator() overload(s) which are appropriate.

template < class derived_type ,
           typename IN ,
           typename OUT = IN ,
           size_t vsize = zimt::vector_traits < IN > :: size
         >
struct callable
{
  // using a cl_ prefix here for the vectorized types to avoid a name
  // clash with class unary_functor, which also defines in_v, out_v
  
  typedef typename vector_traits < IN , vsize > :: type cl_in_v ;
  typedef typename vector_traits < OUT , vsize > :: type cl_out_v ;
  
  OUT operator() ( const IN & in ) const
  {
    auto self = static_cast < const derived_type * const > ( this ) ;
    OUT out ;
    self->eval ( in , out ) ;
    return out ;
  }
  
  OUT operator() ( const IN & in )
  {
    auto self = static_cast < derived_type * > ( this ) ;
    OUT out ;
    self->eval ( in , out ) ;
    return out ;
  }
  
  template < typename = std::enable_if < ( vsize > 1 ) > >
  cl_out_v operator() ( const cl_in_v & in ) const
  {
    auto self = static_cast < const derived_type * const > ( this ) ;
    cl_out_v out ;
    self->eval ( in , out ) ;
    return out ;
  }
  
  template < typename = std::enable_if < ( vsize > 1 ) > >
  cl_out_v operator() ( const cl_in_v & in )
  {
    auto self = static_cast < derived_type * > ( this ) ;
    cl_out_v out ;
    self->eval ( in , out ) ;
    return out ;
  }
} ;

/// eval_wrap is a helper function template, wrapping an 'ordinary'
/// function which returns some value given some input in a void function
/// taking input as const reference and writing output to a reference,
/// which is the signature used for evaluation in zimt::unary_functors.

template < class IN , class OUT >
std::function < void ( const IN& , OUT& ) >
eval_wrap ( std::function < OUT ( const IN& ) > f )
{
  return [f] ( const IN& in , OUT& out ) { out = f ( in ) ; } ;
}

/// struct broadcast is a mixin providing an 'eval' method to a functor which
/// can process vectorized arguments. This mixin is inherited by the functor
/// missing that capability, using CRTP. Because here, in the providing class,
/// nothing is known (or, even, knowable) about the functor, we need to pass
/// additional template arguments to establish the usual zimt unary functor
/// frame of reference, namely in_type, out_type, vsize etc.
/// The resulting 'vectorized' eval may not be efficient: it has to build
/// individual 'in_type' values from the vectorized input, process them with
/// the derived functor's eval routine, then insert the resulting out_type in
/// the vectorized output. But it's a quick way of getting vectorized evaluation
/// capability without writing the vector code. This is particularly useful when
/// the functor's unvectorized eval() is complex (like, calling into legacy code
/// or even into opaque binary) and 'proper' vectorization is hard to do.
/// And with a bit of luck, the optimizer 'recognizes' what's going on and
/// produces SIMD code anyway.
/// Note that the derived class needs a using declaration for the vectorized
/// eval overload inherited from this base class - see broadcast_type (below)
/// for an example of using this mixin.

template < class derived_type ,
           typename IN ,
           typename OUT = IN ,
           size_t _vsize = zimt::vector_traits < IN > :: size
         >
struct broadcast
: public zimt::unary_functor < IN , OUT , _vsize >
{
  typedef zimt::unary_functor < IN , OUT , _vsize > base_type ;
  
  using base_type::vsize ;
  using base_type::dim_in ;
  using base_type::dim_out ;
  
  using typename base_type::in_type ;
  using typename base_type::out_type ;
  using typename base_type::in_ele_type ;
  using typename base_type::out_ele_type ;
  using typename base_type::in_nd_ele_type ;
  using typename base_type::out_nd_ele_type ;
  
  using typename base_type::in_v ;
  using typename base_type::out_v ;
  using typename base_type::in_ele_v ;
  using typename base_type::out_ele_v ;
  using typename base_type::in_nd_ele_v ;
  using typename base_type::out_nd_ele_v ;

  // provide the vectorized eval which broadcasts the unvectorized one.
  // note that, in the derived functor, you need a using statement, like
  // using broadcast < derived_functor , float , float , 8 > :: eval ;
  // to be able to invoke the vectorized eval, if this is missing, only
  // the unvectorized eval in derived_functor is seen.

  template < typename = std::enable_if < ( vsize > 1 ) > >
  void eval ( const in_v & inv ,
                    out_v & outv ) const
  {
    // to use the derived functor's eval routine, we need:

    auto fp = static_cast < const derived_type * > ( this ) ;

    // we reinterpret input and output as nD types. in_v/out_v are
    // plain SIMD types if in_type/out_type is fundamental; here we
    // want a TinyVector of one element in this case.

    const in_nd_ele_v & iv ( reinterpret_cast < const in_nd_ele_v & > ( inv ) ) ;
    out_nd_ele_v & ov ( reinterpret_cast < out_nd_ele_v & > ( outv ) ) ;

    // we also need a view to an in_type and an out_type object as a nD
    // entity, even if they are single-channel, so that we can us a loop
    // to access them, while we need the plain in_type/out_type objects
    // as arguments to the unvectorized eval routine

    in_nd_ele_type nd_in ;
    out_nd_ele_type nd_out ;

    in_type  & in ( reinterpret_cast < in_type & >  ( nd_in ) ) ;
    out_type & out ( reinterpret_cast < out_type & > ( nd_out ) ) ;

    // now comes the rollout. If either dim_in or dim_out are 1,
    // the compiler should optimize away the loop.

    for ( int e = 0 ; e < vsize ; e++ )
    {
      // extract the eth input value from the simdized input

      for ( int d = 0 ; d < dim_in ; d++ )
        nd_in [ d ] = iv [ d ] [ e ] ;

      // process it with eval, passing the eval-compatible references

      fp->eval ( in , out ) ;

      // now distribute eval's result to the simdized output

      for ( int d = 0 ; d < dim_out ; d++ )
        ov [ d ] [ e ] = nd_out [ d ] ;
    }
  }
} ;

// broadcast_type's constructor takes an unvectorized function and
// 'rolls it out' into a zimt::unary_functor with both unvectorized
// and vectorized eval methods. The 'vectorized' version produces
// unvectorized data from the vector(s), processes them with the
// unvectorized function and writes them to the result vector(s).
// All of this is not very efficient, unless the compiler 'gets it'
// And produces 'real' vector code - not too likely, but maybe more
// so in the future. The code is more for 'quick shots' where vector
// code is not (yet) available or the effort to write it is not
// worth while. Another use scenario is calls to opaque code like
// foreign functions. When using broadcast_type with python code, keep
// in mind that you shouldn't have multithreading enabled, so
// define VSPLINE_SINGLETHREAD.
// The c'tor args allow passing in lambdas, which is extra convenient:
// zimt::broadcast_type < T > bf ( [] ( T x ) { return x + x ; } ) ;

template < typename I ,
           typename O = I ,
           std::size_t S  = zimt::vector_traits < I > :: size >
struct broadcast_type
: public broadcast < broadcast_type < I , O , S > , I , O , S >
{
  typedef std::function < void ( const I & , O & ) > eval_type ;
  typedef std::function < O ( const I & ) > call_type ;

private:

  const eval_type _eval ;

  typedef broadcast < broadcast_type < I , O , S > , I , O , S >
            base_type ;

public:

  // two c'tor variants, taking the unvectorized function in different
  // guise

  broadcast_type ( const call_type & _f )
  : _eval ( eval_wrap ( _f ) )
  { }

  broadcast_type ( const eval_type & _ev )
  : _eval ( _ev )
  { }

  // unvectorized eval delegates to _eval

  void eval ( const I & in , O & out ) const
  {
    _eval ( in , out ) ;
  }

  // we need a using declaration to make the 'vectorized' form of 'eval'
  // from the mixin 'broadcast' available in this (derived) class

  using base_type::eval ;
} ;

/// class chain_type is a helper class to pass one unary functor's result
/// as argument to another one. We rely on T1 and T2 to provide a few of the
/// standard types used in unary functors. Typically, T1 and T2 will both be
/// zimt::unary_functors, but the type requirements could also be fulfilled
/// 'manually'.
///
/// Note how callability is introduced via the mixin 'zimt::callable'.
/// The inheritance definition looks confusing, the template arg list reads as:
/// 'the derived class, followed by the arguments needed to determine the
/// call signature(s)'. See zimt::callable above.

template < typename T1 ,
           typename T2 >
struct chain_type
: public zimt::unary_functor < typename T1::in_type ,
                                  typename T2::out_type ,
                                  T1::vsize > ,
  public zimt::callable
         < chain_type < T1 , T2 > ,
           typename T1::in_type ,
           typename T2::out_type ,
           T1::vsize
         >
{
  // definition base_type
  
  enum { vsize = T1::vsize } ;
  
  typedef zimt::unary_functor < typename T1::in_type ,
                                   typename T2::out_type ,
                                   vsize > base_type ;

  using typename base_type::in_type ;
  using typename base_type::out_type ;
  using typename base_type::in_v ;
  using typename base_type::out_v ;

  // we require both functors to share the same vectorization width
          
  static_assert ( T1::vsize == T2::vsize ,
                  "can only chain unary_functors with the same vector width" ) ;

  static_assert ( std::is_same < typename T1::out_type , typename T2::in_type > :: value ,
                  "chain: output of first functor must match input of second functor" ) ;

  typedef typename T1::out_type intermediate_type ;
  typedef typename T1::out_v intermediate_v ;
  
  // hold the two functors by value

  const T1 t1 ;
  const T2 t2 ;
  
  // the constructor initializes them

  chain_type ( const T1 & _t1 , const T2 & _t2 )
  : t1 ( _t1 ) ,
    t2 ( _t2 )
    { } ;

  // the actual eval needs a bit of trickery to determine the type of
  // the intermediate type from the type of the first argument.

  void eval ( const in_type & argument ,
                    out_type & result ) const
  {
    intermediate_type intermediate ;
    t1.eval ( argument , intermediate ) ; // evaluate first functor into intermediate
    t2.eval ( intermediate , result ) ;   // feed it as input to second functor
  }

  template < typename = std::enable_if < ( vsize > 1 ) > >
  void eval ( const in_v & argument ,
                    out_v & result ) const
  {
    intermediate_v intermediate ;
    t1.eval ( argument , intermediate ) ; // evaluate first functor into intermediate
    t2.eval ( intermediate , result ) ;   // feed it as input to second functor
  }
  
} ;

/// chain is a factory function yielding the result of chaining
/// two unary_functors.

template < class T1 , class T2 >
zimt::chain_type < T1 , T2 >
chain ( const T1 & t1 , const T2 & t2 )
{
  return zimt::chain_type < T1 , T2 > ( t1 , t2 ) ;
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
                  < zimt::unary_functor_tag < T2::vsize > ,
                    T1
                  > :: value
               && std::is_base_of
                  < zimt::unary_functor_tag < T1::vsize > ,
                    T2
                  > :: value
             > :: type
         >
zimt::chain_type < T1 , T2 >
operator+ ( const T1 & t1 , const T2 & t2 )
{
  return zimt::chain ( t1 , t2 ) ;
}

/// class grok_type is a helper class wrapping a zimt::unary_functor
/// so that it's type becomes opaque - a technique called 'type erasure',
/// here applied to zimt::unary_functors with their specific
/// capability of providing both vectorized and unvectorized operation
/// in one common object.
///
/// While 'grokking' a unary_functor may degrade performance slightly,
/// the resulting type is less complex, and when working on complex
/// constructs involving several unary_functors, it can be helpful to
/// wrap the whole bunch into a grok_type for some time to make compiler
/// messages more palatable. I even suspect that the resulting functor,
/// which simply delegates to two std::functions, may optimize better at
/// times than a more complex functor in the 'grokkee'.
///
/// Performance aside, 'grokking' a zimt::unary_functor produces a
/// simple, consistent type that can hold *any* unary_functor with the
/// given input and output type(s), so it allows to hold and use a
/// variety of (intrinsically differently typed) functors at runtime
/// via a common handle which is a zimt::unary_functor itself and
/// can be passed to the transform-type routines. With unary_functors
/// being first-class, copyable objects, this also makes it possible
/// to pass around unary_functors between different TUs where user
/// code can provide new functors at will which can simply be used
/// without having to recompile to make their type known, at the cost
/// of a call through a std::function.
///
/// grok_type also provides a convenient way to introduce functors into
/// zimt. Since the functionality is implemented with std::functions,
/// we allow direct initialization of these std::functions on top of
/// 'grokking' the capabilities of another unary_functor via lambda
/// expressions.
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
/// complexity of the originla grokked type family.
///
/// For grok_type objects where _vsize is greater 1, there are
/// constructor overloads taking only a single function. These
/// constructors broadcast the unvectorized function to process
/// vectorized data, providing a quick way to produce code which
/// runs with vector data, albeit less efficiently than true vector
/// code.
///
/// finally, for convenience, grok_type also provides operator(),
/// to use the grok_type object with function call syntax, and it
/// also provides the common 'eval' routine(s), just like any other
/// unary_functor.

template < typename IN ,       // argument or input type
           typename OUT = IN , // result type
           size_t _vsize = zimt::vector_traits < IN > :: size
         >
struct grok_type
: public zimt::unary_functor < IN , OUT , _vsize > ,
  public zimt::callable < grok_type < IN , OUT , _vsize > ,
                             IN , OUT , _vsize >
{
  typedef zimt::unary_functor < IN , OUT , _vsize > base_type ;

  using base_type::vsize ;
  using base_type::dim_in ;
  using base_type::dim_out ;

  using typename base_type::in_type ;
  using typename base_type::out_type ;
  using typename base_type::in_v ;
  using typename base_type::out_v ;
  
  typedef std::function < void ( const in_type & , out_type & ) > eval_type ;
  typedef std::function < out_type ( const in_type & ) > call_type ;

  eval_type _ev ;

  operator bool()
  {
    return _ev ;
  }

  // given these types, we can define the types for the std::function
  // we will use to wrap the grokkee's evaluation code in.

  typedef std::function < void ( const in_v & , out_v & ) > v_eval_type ;
  
  // this is the class member holding the std::function:

  v_eval_type _v_ev ;

  // we also define a std::function type using 'normal' call/return syntax

  typedef std::function < out_v ( const in_v & ) > v_call_type ;

  /// we provide a default constructor so we can create an empty
  /// grok_type and assign to it later. Calling the empty grok_type's
  /// eval will result in an exception.

  grok_type() { } ;
  
  /// direct initialization of the internal evaluation functions
  /// this overload, with two arguments, specifies the unvectorized
  /// and the vectorized evaluation function explicitly.
  
  grok_type ( const eval_type & fev ,
              const v_eval_type & vfev )
  : _ev ( fev ) ,
    _v_ev ( vfev )
  { } ;
  
  /// constructor taking a call_type and a v_call_type,
  /// initializing the two std::functions _ev and _v_ev
  /// with wrappers around these arguments which provide
  /// the 'standard' zimt evaluation functor signature

  grok_type ( call_type f , v_call_type vf )
  : _ev ( eval_wrap ( f ) )
  , _v_ev ( eval_wrap ( vf ) )
  { } ;
    
  /// constructor from 'grokkee' using lambda expressions
  /// to initialize the std::functions _ev and _v_ev.
  /// we enable this if grokkee_type is a zimt::unary_functor

  template < class grokkee_type ,
             typename std::enable_if
              < std::is_base_of
                < zimt::unary_functor_tag < vsize > ,
                  grokkee_type
                > :: value ,
                int
              > :: type = 0
            >
  grok_type ( grokkee_type grokkee )
  : _ev ( [ grokkee ] ( const IN & in , OUT & out )
            { grokkee.eval ( in , out ) ; } )
  , _v_ev ( [ grokkee ] ( const in_v & in , out_v & out )
            { grokkee.eval ( in , out ) ; } )
  { } ;
    
  /// constructor taking only an unvectorized evaluation function.
  /// this function is broadcast, providing evaluation of SIMD types
  /// with non-vector code, which is less efficient.

  grok_type ( const eval_type & fev )
  : grok_type ( zimt::broadcast_type < IN , OUT , vsize >
                 ( [fev] ( const IN & in )
                   {
                     OUT out ;
                     fev ( in , out ) ;
                     return out ;
                   }
                 )
              )
  { } ;
  
  /// constructor taking only one call_type, which is also broadcast,
  /// since the call_type std::function is wrapped to provide a
  /// std::function with zimt's standard evaluation functor signature
  /// and the result is fed to the single-argument functor above.

  grok_type ( const call_type & f )
  : grok_type ( zimt::broadcast_type < IN , OUT , vsize > ( f ) )
  { } ;
  
  /// unvectorized evaluation. This is delegated to _ev.

  void eval ( const IN & i , OUT & o ) const
  {
    _ev ( i , o ) ;
  }
  
  /// vectorized evaluation function template
  /// the eval overload above will catch calls with (in_type, out_type)
  /// while this overload will catch vectorized evaluations.

  template < typename = std::enable_if < ( vsize > 1 ) > >
  void eval ( const in_v & i , out_v & o ) const
  {
    _v_ev ( i , o ) ;
  }
  
} ;

/// specialization of grok_type for _vsize == 1
/// this is the only possible specialization if vectorization is not used.
/// here we don't use _v_ev but only the unvectorized evaluation.

template < typename IN , // argument or input type
           typename OUT  // result type
         >
struct grok_type < IN , OUT , 1 >
: public zimt::unary_functor < IN , OUT , 1 > ,
  public zimt::callable < grok_type < IN , OUT , 1 > ,
                             IN , OUT , 1 >
{
  typedef zimt::unary_functor < IN , OUT , 1 > base_type ;

  enum { vsize = 1 } ;
  using typename base_type::in_type ;
  using typename base_type::out_type ;
  using typename base_type::in_v ;
  using typename base_type::out_v ;
  
  typedef std::function < void ( const in_type & , out_type & ) > eval_type ;
  typedef std::function < out_type ( const in_type & ) > call_type ;

  eval_type _ev ;

  grok_type() { } ;
  
  template < class grokkee_type ,
             typename std::enable_if
              < std::is_base_of
                < zimt::unary_functor_tag < 1 > ,
                  grokkee_type
                > :: value ,
                int
              > :: type = 0
           >
  grok_type ( grokkee_type grokkee )
  : _ev ( [ grokkee ] ( const IN & in , OUT & out )
            { grokkee.eval ( in , out ) ; } )
  { } ;
  
  grok_type ( const eval_type & fev )
  : _ev ( fev )
  { } ;
  
  grok_type ( call_type f )
  : _ev ( eval_wrap ( f ) )
  { } ;
    
  void eval ( const IN & i , OUT & o ) const
  {
    _ev ( i , o ) ;
  }
  
} ;

/// grok() is the corresponding factory function, wrapping grokkee
/// in a zimt::grok_type.

template < class grokkee_type >
zimt::grok_type < typename grokkee_type::in_type ,
                     typename grokkee_type::out_type ,
                     grokkee_type::vsize >
grok ( const grokkee_type & grokkee )
{
  return zimt::grok_type < typename grokkee_type::in_type ,
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
           size_t _vsize = zimt::vector_traits < _in_type > :: vsize >
struct amplify_type
: public zimt::unary_functor < _in_type , _out_type , _vsize > ,
  public zimt::callable
         < amplify_type < _in_type , _out_type , _math_type , _vsize > ,
           _in_type ,
           _out_type ,
           _vsize
         >
{
  typedef typename
    zimt::unary_functor < _in_type , _out_type , _vsize > base_type ;
  
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
  
  typedef typename zimt::vector_traits < math_ele_type , vsize > :: type
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
  
  void eval ( const in_type & in , out_type & out ) const
  {
    out = out_type ( math_type ( in ) * factor ) ;
  }
  
  template < typename = std::enable_if < ( vsize > 1 ) > >
  void eval ( const in_v & in , out_v & out ) const
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
      zimt::assign ( _out[i] , math_ele_v ( _in[i] ) * _factor[i] ) ;
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
           size_t _vsize = zimt::vector_traits < _in_type > :: vsize >
struct flip
: public zimt::unary_functor < _in_type , _in_type , _vsize > ,
  public zimt::callable
         < flip < _in_type , _vsize > ,
           _in_type ,
           _in_type ,
           _vsize
         >
{
  typedef typename zimt::unary_functor
                     < _in_type , _in_type , _vsize > base_type ;
                     
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
  
  void eval ( const in_type & in_ , out_type & out ) const
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
  void eval ( const in_v & in_ , out_v & out ) const
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
      zimt::assign ( _out [ e ] , _in [ dimension - e - 1 ] ) ;
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
           size_t _vsize = zimt::vector_traits < data_t > :: size ,
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
: public unary_functor < crd_t , data_t , _vsize > ,
  public zimt::callable
         < yield_type < crd_t , data_t , _vsize > ,
           crd_t , data_t , _vsize
         >
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
            ) const
  {
    v = data [ crd ] ;
  }

  // the next four overloads deal with vectorized access.
  // first variant: coordinate is 1D, value is single-channel

  void eval ( const in_v & crd , out_v & v ,
              std::false_type ,
              std::false_type ,
              std::false_type ) const
  {
    auto ofs = crd * int(data.stride(0)) ;

    data_t * p_src = data.data() ;
    v.gather ( p_src , ofs ) ;
  }

  // second variant: coordinate is 1D, value is multi-channel

  void eval ( const in_v & crd , out_v & v ,
              std::false_type ,
              std::false_type ,
              std::true_type ) const
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
              std::false_type ) const
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
              std::true_type ) const
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
  void eval ( const in_t & crd , out_t & v ) const
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
: public unary_functor < crd_t , data_t , _vsize > ,
  public zimt::callable
         < yield_type < crd_t , data_t , _vsize > ,
           crd_t , data_t , _vsize
         >
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
              std::false_type ) const
  {
    iy.eval ( std::round ( crd ) , v ) ;
  }

  void eval ( const in_type & crd , out_type & v ,
              std::true_type ,
              std::true_type ) const
  {
    zimt::xel_t < int , dimension > icrd ;
    for ( int d = 0 ; d < dimension ; d++ )
      icrd [ d ] = std::round ( crd [ d ] ) ;
    iy.eval ( icrd , v ) ;
  }

  // vectorized operation, for 1D and nD cases

  void eval ( const in_v & crd , out_v & v ,
              std::false_type ,
              std::false_type ) const
  {
    typename iy_t::in_v icrd ( round ( crd ) ) ;
    iy.eval ( icrd , v ) ;
  }

  void eval ( const in_v & crd , out_v & v ,
              std::false_type ,
              std::true_type ) const
  {
    typename iy_t::in_v icrd ;
    for ( int d = 0 ; d < dimension ; d++ )
      icrd [ d ] = round ( crd [ d ] ) ;
    iy.eval ( icrd , v ) ;
  }

public:

  template < typename in_t , typename out_t >
  void eval ( const in_t & crd , out_t & v ) const
  {
    typedef typename std::is_same < out_t , out_type > :: type is_scalar_t ;
    eval ( crd , v , is_scalar_t() , is_nd_t() ) ; 
  }

} ;

/// while 'normal' unary_functors are all derived from unary_functor_tag,
/// sink functors will be derived from sink_functor_tag.

template < size_t _vsize >
struct sink_functor_tag { } ;

/// sink_functor is used for functors without an output - e.g. reductors
/// which are used for analytic purposes on data sets. They use the same
/// system of input types, but omit the output types.

template < typename IN , // argument or input type
           size_t _vsize = zimt::vector_traits < IN > :: size
         >
struct sink_functor
: public sink_functor_tag < _vsize >
{
  // number of fundamentals in simdized data. If vsize is 1, the vectorized
  // types will 'collapse' to the unvectorized types.

  enum { vsize = _vsize } ;

  // number of dimensions.

  enum { dim_in = zimt::vector_traits < IN > :: dimension } ;
  
  // typedefs for incoming (argument) type. This type is an
  // unvectorized type, like zimt::xel_t < float , 2 >.
  // Since such types consist of elements of the same type,
  // the corresponding vectorized type can be easily determined.
  
  typedef IN in_type ;
  
  // elementary types of same. we rely on zimt::vector_traits to provide
  // these types.
  
  typedef typename zimt::vector_traits < IN > :: ele_type in_ele_type ;
  
  // 'synthetic' type for input. This is always a TinyVector, possibly
  // of only one element, of the elementary type of in_type.
  // On top of providing a uniform container type (the TinyVector) the
  // synthetic type is also 'unaware' of any specific meaning the 'true'
  // input type may have, and arithmetic operations on the synthetic
  // types won't clash with arithmetic defined for the 'true' types.

  typedef zimt::xel_t < in_ele_type , dim_in > in_nd_ele_type ;
  
  // for vectorized operation, we need a few extra typedefs. I use a _v
  // suffix instead of the _type suffix above to indicate vectorized types.
  // If vsize is 1, the _v types simply collapse to the unvectorized
  // types, having them does no harm, but it's not safe to assume that,
  // for example, in_v and in_type are in fact different types.

  /// a simdized type of the elementary type of result_type,
  /// which is used for coefficients and results. this is fixed via
  /// the traits class vector_traits (in vector.h). Note how we derive
  /// this type using vsize from the template argument, not what
  /// zimt::vector_traits deems appropriate for ele_type - though
  /// both numbers will be the same in most cases.
  
  typedef typename vector_traits < IN , vsize > :: ele_v in_ele_v ;
  
  // 'synthetic' types for simdized input. This is always a
  // TinyVector, possibly of only one element, of the simdized input
  // and output type.

  typedef typename vector_traits < IN , vsize > :: nd_ele_v in_nd_ele_v ;
  
  /// vectorized in_type. zimt::vector_traits supplies this
  /// type so that multidimensional/multichannel data come as
  /// zimt::xel_ts, while 'singular' data won't be made
  /// into TinyVectors of one element.
  
  typedef typename vector_traits < IN , vsize > :: type in_v ;

  /// vsize wide vector of ints, used for gather/scatter indexes
  
  typedef typename vector_traits < int , vsize > :: ele_v ic_v ;

} ;

} ; // end of namespace zimt

#endif // VSPLINE_UNARY_FUNCTOR_H

