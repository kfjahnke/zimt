/************************************************************************/
/*                                                                      */
/*    zimt - a set of generic tools for creation and evaluation      */
/*              of uniform b-splines                                    */
/*                                                                      */
/*            Copyright 2015 - 2023 by Kay F. Jahnke                    */
/*                                                                      */
/*    The git repository for this software is at                        */
/*                                                                      */
/*    https://bitbucket.org/kfj/zimt                                 */
/*                                                                      */
/*    Please direct questions, bug reports, and contributions to        */
/*                                                                      */
/*    kfjahnke+zimt@gmail.com                                        */
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

/*! \file eval.h

    \brief code to evaluate uniform b-splines

    This body of code contains class evaluator and auxilliary classes which are
    needed for it's smooth operation.

    The evaluation is a reasonably straightforward process: A subset of the coefficient
    array, containing coefficients 'near' the point of interest, is picked out, and
    a weighted summation over this subset produces the result of the evaluation.
    The complex bit is to have the right coefficients in the first place
    (this is what prefiltering does), and to use the appropriate weights on
    the coefficient window. For b-splines, there is an efficient method to
    calculate the weights by means of a matrix multiplication, which is easily
    extended to handle b-spline derivatives as well. Since this code lends itself
    to a generic implementation, and it can be parametrized by the spline's order,
    and since the method performs well, I use it here in preference to the code which
    P. Thevenaz uses (which is, for the orders of splines it encompasses, the matrix
    multiplication written out with a few optimizations, like omitting multiplications
    with zero, and slightly more concise calculation of powers). The weight generation
    code is in basis.h.
    
    Evaluation of a b-spline seems to profit more from vectorization than prefiltering,
    especially for float data. On my system, I found single-precision
    operation was about three to four times as fast as unvectorized code (AVX2).
    
    The central class of this file is class evaluator. evaluator objects are set up to
    provide evaluation of a specific b-spline. Once they are set up they don't change and
    effectively become pure functors. The evaluation methods typically take their arguments
    per reference. The details of the evaluation variants, together with explanations of
    specializations used for extra speed, can be found with the individual evaluation
    routines. 'Ordinary' call syntax via operator() is also provided for convenience.
    
    What do I mean by the term 'pure' functor? It's a concept from functional programming.
    It means that calling the functor will not have any effect on the functor itself - it
    can't change once it has been constructed. This has several nice effects: it can
    potentially be optimized very well, it is thread-safe, and it will play well with
    functional programming concepts - and it's conceptually appealing.
    
    Code using class evaluator will probably use it at some core place where it is
    part of some processing pipeline. An example would be an image processing program:
    one might have some outer loop generating arguments (typically simdized types)
    which are processed one after the other to yield a result. The processing will
    typically have several stages, like coordinate generation and transformations,
    then use class evaluator to pick an interpolated intermediate result, which is
    further processed by, say, colour or data type manipulations before finally
    being stored in some target container. The whole processing pipeline can be
    coded to become a single functor, with one of class evaluator's eval-type
    routines embedded somewhere in the middle, and all that's left is code to
    efficiently handle the source and destination to provide arguments to the
    pipeline - like the code in transform.h. And since this code is made to
    provide the data feeding and storing, the only coding needed is for the pipeline,
    which is where the 'interesting' stuff happens.
    
    class evaluator is the 'front' class for evaluation, the implementation
    of the functionality is in class inner_evaluator. User code will typically not
    use class inner_evaluator, which lives in namespace detail.
    
    The code in this file concludes by providing factory functions to obtain
    evaluators for bspline objects. These factory functions produce
    objects which are type-erased (see grok_type) wrappers around the
    evaluators, which hide what's inside and simply provide evaluation of the
    spline at given coordinates. These objects also provide operator(), so that
    they can be used like functions.
    
    Passing vectorized coordinates results in vectorized results, where the specific
    types for the vectorized input and output is gleaned from vector_traits.
    If Vc is used and the fundamental data types can be made into a Vc::SimdArray,
    the data types for vectorized input/output will be Vc::SimdArrays (or
    xel_t of Vc::SimdArrays for multichannel data). Otherwise, zimt's
    SIMD emulation will be used, which replaces Vc::SimdArray with simd_type,
    which is an autovectorization-friendly type with similar performance. Since the
    objects produced by the factory functions are derived from unary_functor,
    they can be fed to the functions in transform.h, like any other
    unary_functor.
    
    If you use transform and relatives, vectorization is done automatically:
    the transform routines will inquire for the functor's vectorized signature which
    is encoded as it's in_v and out_v types, which are - normally - results of querying
    vector_traits. The data are then deinterleaved into the vectorized input
    type, fed to the functor, and the vectorized result is interleaved into target
    memory. class evaluator has all the relevant attributes and capabilites, so using
    it with transform and relatives is easy and you need not be aware of any of the
    'vector magic' going on internally - nor of the automatic multithreading. See
    transform.h for more on the topic.
*/

// #ifndef ZIMT_EVAL_H
// #define ZIMT_EVAL_H

#if defined(ZIMT_EVAL_H) == defined(HWY_TARGET_TOGGLE)
  #ifdef ZIMT_EVAL_H
    #undef ZIMT_EVAL_H
  #else
    #define ZIMT_EVAL_H
  #endif

#include <array>

#include "basis.h"
#include "bspline.h"
#include "unary_functor.h"
#include "map.h"
#include "block.h"

// #if HWY_ONCE

#ifndef CC14_INT_SEQ
#define CC14_INT_SEQ

// While zimt still uses the c++11 dialect, Here I use a c++14 feature.
// I picked the code from https://gist.github.com/ntessore/dc17769676fb3c6daa1f

namespace std14
{
  template<typename T, T... Ints>
  struct integer_sequence
  {
    typedef T value_type;
    static constexpr std::size_t size() { return sizeof...(Ints); }
  };
  
  template<std::size_t... Ints>
  using index_sequence = integer_sequence<std::size_t, Ints...>;
  
  template<typename T, std::size_t N, T... Is>
  struct make_integer_sequence : make_integer_sequence<T, N-1, N-1, Is...> {};
  
  template<typename T, T... Is>
  struct make_integer_sequence<T, 0, Is...> : integer_sequence<T, Is...> {};
  
  template<std::size_t N>
  using make_index_sequence = make_integer_sequence<std::size_t, N>;
  
  template<typename... T>
  using index_sequence_for = make_index_sequence<sizeof...(T)>;
} ;

#endif

HWY_BEFORE_NAMESPACE() ;
BEGIN_ZIMT_SIMD_NAMESPACE(zimt)

/// If there are several differently-typed basis functors to be
/// combined in a multi_bf_type object, we can erase their type,
/// just like grok_type does for unary_functors.
/// grokking a basis functor may cost a little bit of performance
/// but it makes the code to handle multi_bf_types simple: instead
/// of having to cope for several, potentially differently-typed
/// per-axis functors there is only one type - which *may* be a
/// bf_grok_type if the need arises to put differently-typed
/// basis functors into the multi_bf_type.
/// With this mechanism, the code to build evaluators can be kept
/// simple (handling only one uniform type of basis functor used
/// for all axes) and still use different basis functors.

template < typename _delta_et ,
           typename _target_et = _delta_et ,
           std::size_t _vsize
             = vector_traits < _delta_et > :: size >
struct bf_grok_type
{
  static const std::size_t degree ; // degree of the basis function

  typedef _delta_et delta_et ;   // elementary type of delta (like, float)
  typedef _target_et target_et ; // elementary type of weight array
  enum { vsize = _vsize } ;      // lane count

  // erased type of the basis function: a function taking a pointer
  // to target_et, where the weights will be deposited, and a const
  // reference to delta_et, to calculate the weights from.

  typedef std::function
          < void ( target_et * , const delta_et & ) > e_eval_t ;

  // for the erased type of the vectorized basis function, the argument
  // types are generated via alias template simdized_type

  typedef simdized_type < delta_et , vsize > delta_vt ;
  typedef simdized_type < target_et , vsize > target_vt ;
  
  // and this is the erased type of the vectorized basis function:

  typedef std::function
          < void ( target_vt * , const delta_vt & ) > v_eval_t ;

  // bf_grok_type holds two const references, one to the scalar form and
  // one to the vectorized form of the basis function, both type-erased
  // to hide the concrete implementation

  const e_eval_t e_eval ;
  const v_eval_t v_eval ;

  // the 'straightforward' c'tor accepts two suitable std::functions

  bf_grok_type ( const e_eval_t & _e_eval ,
                 const v_eval_t & _v_eval )
  : e_eval ( _e_eval ) ,
    v_eval ( _v_eval )
  { }

  // alternatively, an object may be passed in, which can be called
  // with the signatures the two std::functions would take. This
  // object's corresponding operator() overloads are lambda-captured
  // and used to initialize e_eval and v_eval.

  template < class grokkee_type >
  bf_grok_type ( const grokkee_type & grokkee )
  : e_eval ( [grokkee] ( target_et * p_trg , const delta_et & delta )
             { grokkee ( p_trg , delta ) ; } ) ,
    v_eval ( [grokkee] ( target_vt * p_trg , const delta_vt & delta )
             { grokkee ( p_trg , delta ) ; } )
  { }

  // finally, the two operator() overloads for bf_grok_t: they take
  // the signature of e_eval and v_eval and delegate to them.

  void operator() ( target_et * p_trg , const delta_et & delta ) const
  {
    e_eval ( p_trg , delta ) ;
  }

  void operator() ( target_vt * p_trg , const delta_vt & delta ) const
  {
    v_eval ( p_trg , delta ) ;
  }
} ;

// bf_grok is a factory function to create a bf_grok_type object
// from a 'grokkee' object with suitable operator() overloads.

template < typename _grokkee_t ,
           typename _delta_et = float ,
           typename _target_et = _delta_et ,
           std::size_t _vsize
             = vector_traits < _delta_et > :: size >
bf_grok_type < _delta_et , _target_et , _vsize >
bf_grok ( const _grokkee_t & grokkee )
{
  return bf_grok_type < _delta_et , _target_et , _vsize > ( grokkee ) ;
}

/// When several basis functors have to be passed to an evaluator,
/// it's okay to pass a container type like a std::array or std::vector.
/// All of these basis functors have to be of the same type, but using
/// the method given above ('grokking' basis functors) it's possible
/// to 'slot in' differently-typed basis functors - the functionality
/// required by the evaluation code is provided by the 'grokked'
/// functors and their inner workings remain opaque.

/// For convenience, zimt provides class multi_bf_type, which
/// inherits from std::array and passes the spline degree and,
/// optionally, other arguments, through to the c'tors of the
/// individual basis functors. With this construct, the c'tor
/// signature for 'standard' b-splines (using basis_functor)
/// can be left as it is, and the multi_bf_type holding the
/// basis functors builds the basis functor with the given degree
/// and derivative specification. The rather complex machinations
/// needed to 'roll out' these arguments to the individual per-axis
/// basis functors does not require user interaction.

/// class multi_bf_type holds a set of basis functors, one for each
/// dimension. While most of zimt uses xel_t to hold
/// such n-dimensional objects, here we use std::array, because we
/// need aggregate initialization, while xel_t uses element
/// assignment.
/// The code is a bit tricky: the second c'tor is the one which is
/// called from user code and takes an arbitrary argument sequence.
/// To 'roll out' the initialization into an initializer list, we
/// pass an index_sequence to the private first c'tor and use the
/// expansion of the index sequence as the first value in a comma 
/// separated list of expressions, effectively discarding it.
/// The variable argument list is expanded for each call to the
/// basis functor's c'tor, so the unspecialized template initializes
/// all basis functors with the same values. this is fine for cases
/// where all basis functions are meant to do the same thing, but for
/// b-splines, we have to consider the derivative specification,
/// which is a per-axis argument. So for that case (see below) we
/// need a specialization.

template < typename bf_type , int ndim >
struct multi_bf_type
: public std::array < bf_type , ndim >
{
  public:

  typedef std::array < bf_type , ndim > base_type ;

  const std::size_t degree ;

  private:
  
  template < std::size_t ... indices , typename ... targs >
  multi_bf_type ( std14::index_sequence < indices ... > ,
                  int _degree ,
                  targs ... args )
  : std::array < bf_type , ndim >
      { ( indices , bf_type ( _degree , args ... ) ) ... } ,
    degree ( _degree )
  { }

  public:

  template < typename ... targs >
  multi_bf_type ( int _degree , targs ... args )
  : multi_bf_type ( std14::make_index_sequence<ndim>() ,
                    _degree , args ... )
  { }

  using base_type::base_type ;
} ;

/// homogeneous_mbf_type can be used for cases where all basis functors
/// are the same. The evaluation code uses operator[] to pick the functor
/// for each axis, so here we merely override operator[] to always
/// yield a const reference to the same basis functor.

template < typename bf_type >
struct homogeneous_mbf_type
{
  const bf_type bf ;

  homogeneous_mbf_type ( const bf_type & _bf )
  : bf ( _bf )
  { }

  const bf_type & operator[] ( std::size_t i ) const
  {
    return bf ;
  }
} ;

/// For b-spline processing, we use a multi_bf_type of b-spline basis
/// functors (class basis_functor). We can't use the
/// unspecialized template for this basis functor, because it takes the
/// derivative specification, which is specified per-axis, so we need
/// to 'pick it out' from the derivative specification and pass the
/// per-axis value to the per-axis basis function c'tors.
/// So here, instead of merely using the index_sequence to produce
/// a sequence of basis function c'tor calls and ignoring the indices,
/// we use the indices to pick out the per-axis derivative specs.

template < int ndim , typename math_type >
struct multi_bf_type < basis_functor < math_type > , ndim >
: public std::array < basis_functor < math_type > , ndim >
{
  private:
  
  template < std::size_t ... indices >
  multi_bf_type ( int _degree ,
                  const xel_t < int , ndim > & _dspec ,
                  std14::index_sequence < indices ... > )
  : std::array < basis_functor < math_type > , ndim >
      { basis_functor < math_type >
          ( _degree , _dspec [ indices ] ) ... }
  { }

  public:

  multi_bf_type ( int _degree ,
                  const xel_t < int , ndim > & _dspec )
  : multi_bf_type ( _degree , _dspec ,
                    std14::make_index_sequence<ndim>() )
  { }
} ;

namespace detail
{

/// 'inner_evaluator' implements evaluation of a uniform b-spline,
/// or some other spline-like construct relying on basis functions
/// which can provide sets of weights for given deltas.
/// While class evaluator (below, after namespace detail ends)
/// provides objects derived from unary_functor which
/// are meant to be used by user code, here we have a 'workhorse'
/// object to which 'evaluator' delegates. This code 'rolls out'
/// the per-axis weights the basis functor produces to the set
/// of coefficients relevant to the current evaluation locus
/// (the support window). We rely on a few constraints:
/// - we require all basis functors to provide degree+1 weights
/// - the cross product of the per-axis weights is applied to
///   the coefficient window (the kernel is separable).
///
/// TODO: investigate generalization to variable degree basis
/// functors and nD kernels which aren't separable, like RBFs
///
/// The template arguments are, first, the elementary types (e.t.)
/// of the types involved, then several non-type template arguments
/// fixing aggregate sizes. inner_evaluator only uses 'synthetic'
/// types devoid of any specific meaning, so while class evaluator
/// might accept types like 'std::complex<float>' or 'double'
/// class inner_evaluator would instead accept the synthetic types
/// xel_t<float,2> and xel_t<double,1>.
///
/// Note the name system used for the types. first prefixes:
/// - ic: integral coordinate
/// - rc: real coordinate
/// - ofs: offset in memory
/// - cf: b-spline coefficient
/// - math: used for calculations
/// - trg: type for result ('target') of the evaluation
/// an infix of 'ele' refers to a type's elementary type.
/// suffix 'type' is for unvectorized types, while suffix 'v'
/// is used for simdized types, see below.
///
/// User code will not usually create and handle objects of class
/// inner_evaluator, but to understand the code, it's necessary to
/// know the meaning of the template arguments for class inner_evaluator:
///
/// - _ic_ele_type: e.t. of integral coordinates. integral coordinates occur
///   when incoming real coordinates are split into an integral part and a
///   remainder. Currently, only int is used.
///
/// - _rc_ele_type: e.t. of real coordinates. This is used for incoming real
///   coordinates and the remainder mentioned above
///
/// - _ofs_ele_type: e.t. for offsets (in memory). This is used to encode
///   the location of specific coefficients relative to the coefficients'
///   origin in memory. Currently, only int is used.
///
/// - _cf_ele_type: elementary type of b-spline coefficients. While in most
///   cases, this type will be the same as the elementary type of the knot
///   point data the spline is built over, it may also be different. See
///   class bspline's prefilter method.
///
/// - _math_ele_type: e.t. for mathematical operations. All arithmetic
///   inside class inner_evaluator is done with this elementary type.
///   It's used for weight generation, coefficients are cast to it, and
///   only after the result is complete, it's cast to the 'target type'
///
/// - _trg_ele_type: e.t. of target (result). Since class inner_evaluator
///   normally receives it's 'target' per reference, this template argument
///   fixes the target's type. This way, after the arithmetic is done, the
///   result is cast to the target type and then assigned to the target
///   location.
///
/// - _dimension: number of dimensions of the spline, and therefore the
///   number of components in incoming coordinates. If the spline is 1D,
///   coordinates will still be contained in xel_t, with
///   only one component.
///
/// - _channels: number of channels per coefficient. So, when working on
///   RGB pixels, the number of channels would be three. If the spline
///   is over fundamentals (float, double...), _channels is one and the
///   type used here for coefficients is a xel_t with one
///   element.
///
/// - _specialize: class inner_evaluator has specialized code for
///   degree-0 and degree-1 b-splines, aka nearest neighbour and linear
///   interpolation. This specialized code can be activated by passing
///   0 or 1 here, respectively. All other values will result in the use
///   of general b-spline evaluation, which can handle degree-0 and
///   degree-1 splines as well, but less efficiently.
///
/// - _mbf_type: This defines the basis functors. For b-splines, this
///   is a multi_bf_type of as many basis_functors as the
///   spline's dimensions. Making this type a template argument opens
///   the code up to the use of arbitrary basis functions, as long as all
///   the basis functors are of the same type. To use differently-typed
///   basis functors, erase their type using bf_grok_type (see above)

template < typename _ic_ele_type ,   // e.t. of integral coordinates
           typename _rc_ele_type ,   // e.t. of real coordinates
           typename _ofs_ele_type ,  // e.t. for offsets (in memory)
           
           typename _cf_ele_type ,   // elementary type of coefficients
           typename _math_ele_type , // e.t. for mathematical operations
           typename _trg_ele_type ,  // e.t. of target (result)
           
           unsigned int _dimension , // dimensionality of the spline
           unsigned int _channels ,  // number of channels per coefficient
           int _specialize ,         // specialize for NN, linear, general
           typename _mbf_type
             = multi_bf_type < basis_functor < _math_ele_type > , _dimension >
         >
struct inner_evaluator
{
  // make sure math_ele_type is a floating point type
  
  static_assert ( std::is_floating_point < _math_ele_type > :: value ,
                  "class evaluator requires a floating point math_ele_type" ) ;
                  
  // pull in the template arguments in order to allow other code to
  // inquire about the type system

  typedef _ic_ele_type ic_ele_type ;
  typedef _rc_ele_type rc_ele_type ;
  typedef _ofs_ele_type ofs_ele_type ;
  
  typedef _cf_ele_type cf_ele_type ;
  typedef _math_ele_type math_ele_type ;
  typedef _trg_ele_type trg_ele_type ;

  enum { dimension = _dimension } ;
  enum { level = dimension - 1 } ;
  enum { channels = _channels } ;
  enum { specialize = _specialize } ;
  
  // define the 'synthetic' unvectorized types, which are always
  // xel_t, possibly with only one element. Note how this
  // process of 'synthesizing' the types is in a way the opposite
  // process of what's done in class evaluator, where the template
  // arguments are 'taken apart' to get their elementary types.
  
  typedef xel_t < ic_ele_type , dimension > ic_type ;
  typedef xel_t < rc_ele_type , dimension > rc_type ;
  typedef ofs_ele_type ofs_type ; // TODO: superfluous?
  
  typedef xel_t < cf_ele_type , channels > cf_type ;
  typedef xel_t < math_ele_type , channels > math_type ;
  typedef xel_t < trg_ele_type , channels > trg_type ;
  
  typedef xel_t < std::ptrdiff_t , dimension > shape_type ;
  typedef xel_t < int , dimension > derivative_spec_type ;
  typedef _mbf_type mbf_type ;

private:
  
  // typedef typename view_t < 1 , const cf_ele_type * > :: const_iterator
  //   cf_pointer_iterator ;
  typedef const cf_ele_type * const * cf_pointer_iterator ;
  
  /// Initially I was using a template argument for this flag, but it turned out
  /// that using a const bool set at construction time performs just as well.
  /// Since this makes using class evaluator easier, I have chosen to go this way.

  const bool even_spline_degree ;
  
  /// memory location and layout of the spline's coefficients. Note that the
  /// pointer points to the elementary type, and the stride is given in units
  /// of the elementary type as well (hence the 'e' after the underscore).

  const cf_ele_type * const cf_ebase ;
  const shape_type cf_estride ;
  
  /// cf_pointers holds the sum of cf_ebase and window offsets. This produces a small
  /// performance gain: instead of passing the coefficients' base address (cf_ebase)
  /// and the series of offsets (cf_offsets) into the workhorse evaluation code and 
  /// adding successive offsets to cf_ebase, we do the addition in the constructor
  /// and save the offsetted pointers.
  
  array_t < 1 , std::ptrdiff_t > cf_offsets ;
  array_t < 1 , const cf_ele_type * > cf_pointers ;
  
public:

  // wgt holds a set of weight functors, one for each dimension. This object
  // is passed in via the c'tor, and we're not looking in, just copying it.
  // See the comments with multi_bf_type for a description of this type.

  const mbf_type wgt ;
  
  const int spline_degree ;
  const int spline_order ;
  
  /// size of the window of coefficients contributing to a single evaluation.
  /// This equals 'spline_order' to the power of 'dimension'.
  
  const int window_size ;
  
  /// split function. This function is used to split incoming real coordinates
  /// into an integral and a remainder part, which are used at the core of the
  /// evaluation. selection of even or odd splitting is done via the const bool
  /// flag 'even_spline_degree'. My initial implementation had this flag as a
  /// template argument, but this way it's more flexible and there seems to
  /// be no runtime penalty. This method delegates to the free function templates
  /// even_split and odd_split, respectively, which are defined in basis.h.

  template < class IT , class RT >
  void split ( const RT& input , IT& select , RT& tune ) const
  {
    if ( even_spline_degree )
      even_split ( input , select , tune ) ;
    else
      odd_split ( input , select , tune ) ;
  }

  const int & get_order() const
  {
    return spline_order ;
  }

  const int & get_degree() const
  {
    return spline_degree ;
  }

  const shape_type & get_estride() const
  {
    return cf_estride ;
  }

  /// inner_evaluator only has a single constructor, which takes these arguments:
  ///
  /// - _cf_ebase: pointer to the origin of the coefficient array, expressed
  ///   as a pointer to the coefficients' elementary type. 'origin' here means
  ///   the memory location coinciding with the origin of the knot point data,
  ///   which coincides with a bspline object's 'core', not the origin of a
  ///   bspline object's 'container'. Nevertheless, the data have to be suitably
  ///   'braced' - evaluation may well fail (spectacularly) if the brace is
  ///   absent, please refer to class bspline's documentation.
  ///
  /// - _cf_estride: the stride(s) of the coefficient array, expressed in units
  ///   of the coefficients' elementary type.
  ///
  /// - _spline_degree: the degree of the b-spline. this can be up to 45
  ///   currently. See the remarks on 'shifting' in the documentation of class
  ///   evaluator below.
  ///

  inner_evaluator ( const cf_ele_type * const _cf_ebase ,
                    const shape_type & _cf_estride ,
                    int _spline_degree ,
                    const mbf_type & _wgt
                  )
  : cf_ebase ( _cf_ebase ) ,
    cf_estride ( _cf_estride ) ,
    spline_degree ( _spline_degree ) ,
    wgt ( _wgt ) ,
    even_spline_degree ( ! ( _spline_degree & 1 ) ) ,
    spline_order ( _spline_degree + 1 ) ,
    window_size ( std::pow ( _spline_degree + 1 , int(dimension) ) )
  {
    // The evaluation forms a weighted sum over a window of the coefficient array.
    // The sequence of offsets we calculate here is the set of pointer differences
    // from the central element in that window to each element in the window. It's
    // another way of coding this window, where all index calculations have already
    // been done beforehand rather than performing them during the traversal of the
    // window by means of stride/shape arithmetic.
    
    // we want to iterate over all nD indexes in a window which has equal extent
    // of spline_order in all directions (like the reconstruction filter's kernel),
    // relative to a point which is spline_degree/2 away from the origin along
    // every axis. This sounds complicated but is really quite simple: For a cubic
    // b-spline over 2D data we'd get
    
    // (-1,-1) , (0,-1), (1,-1), (2,-1)
    // (-1, 0) , (0, 0), (1, 0), (2, 0)
    // (-1, 1) , (0, 1), (1, 1), (2, 1)
    // (-1, 2) , (0, 2), (1, 2), (2, 2)
    
    // for the indexes, which are subsequently multiplied with the strides and
    // summed up to obtain 1D offsets instead of nD coordinates. So if the coefficient
    // array has strides (10,100) and the coefficients are single-channel, the sequence
    // of offsets generated is
    
    // -110, -100, -90, -80,  // which is -1 * 10 + -1 * 100, 0 * 10 + -1 * 100 ...
    //  -10,    0,  10,  20,
    //   90,  100, 110, 120,
    //  190,  200, 210, 220
  
    shape_type window_shape ( spline_order ) ;
    mcs_t<dimension> mcs ( window_shape ) ;
    
    // cf_pointers will hold the sums of cf_ebase and the offsets into the window
    // of participating coefficients. Now there is only one more information
    // that's needed to localize the coefficient access during evaluation,
    // namely an additional offset specific to the locus of the evaluation. This is
    // generated from the integral part of the incoming coordinates during the
    // evaluation and varies with each evaluation - the DDA (data defined access).
    // This locus-specific offset originates as the integral part of the coordinate,
    // (which is an nD integral coordinate) 'condensed' into an offset by multiplying
    // it with coefficient array's stride and summimg up. During the evaluation,
    // the coefficients in the window relevant to the current evaluation can now
    // be accessed by combining two values: a pointer from 'cf_pointers' and the
    // locus-specific offset. So rather than using a pointer and a set of indexes
    // we use a set of pointers and an index to the same effect. Why do it this way?
    // because of vectorization. If we want to process a vector of loci, this is the
    // most efficient way of coding the operation, since all calculations which do not
    // depend on the locus are already done here in the constructor, and the vector
    // of offsets generated from the vector of loci can be used directly as a gather
    // operand for accessing the coefficients. This gather operand can remain in a
    // register throughout the entire evaluation, only the base pointer this gather
    // operand is used with changes in the course of the evaluation. The problem
    // with this approach is the fact that the vector of offsets is not regular in any
    // predictable way and may well access memory locations which are far apart.
    // Luckily this is the exception, and oftentimes access will be to near memory,
    // which is in cache already.
    
    cf_pointers = array_t < 1 , const cf_ele_type * > ( window_size ) ;
    cf_offsets = array_t < 1 , std::ptrdiff_t > ( window_size ) ;
    
    auto ofs_target = &(cf_offsets[0]) ; // cf_offsets.begin() ;
    auto target = &(cf_pointers[0]) ; // cf_pointers.begin() ;
    
    for ( int i = 0 ; i < window_size ; i++ )
    {
      // offsets are calculated by multiplying indexes with the coefficient array's
      // strides and summing up. So now we have offsets instead of nD indices.
      // By performing this addition now rather than passing in both the pointer and
      // the offsets, we can save a few cycles and reduce register pressure.
      
      // Note how we subtract spline_degree/2 to obtain indexes which are relative
      // to the window's center. Annoying aside: the subtraction of spline_degree/2
      // from *mci yields double (vigra's result type for the right scalar
      // subtraction which accepts only double as the scalar), so the product with
      // cf_estride and the result of 'sum' are also double. Hence I can't code
      // 'auto offset'. We keep a record of the offset and of it's sum with cf_ebase,
      // so we can choose which 'flavour' we want 'further down the line'
      
      std::ptrdiff_t offset = ( ( mcs() - spline_degree / 2 ) * cf_estride ).sum() ;
      *ofs_target = offset ;
      *target = cf_ebase + offset ;
      
      // increment the iterators
      
      // ++mci ;
      ++ofs_target ;
      ++target ;
    }
  }

  /// obtain_weights calculates the weights to be applied to a section
  /// of the coefficients from  the fractional parts of the split coordinates.
  /// What is calculated here is the evaluation of the spline's basis function
  /// at dx, dx+/-1 ... but doing it naively is computationally expensive,
  /// as the evaluation of the spline's basis function at arbitrary values has
  /// to look at the value, find out the right interval, and then calculate
  /// the value with the appropriate function. But we always have to calculate
  /// the basis function for *all* intervals anyway, and the method used here
  /// performs this tasks efficiently using a vector/matrix multiplication.
  /// If the spline is more than 1-dimensional, we need a set of weights for
  /// every dimension. The weights are accessed through a 2D view_t.
  /// For every dimension, there are spline_order weights. Note that this
  /// code will process unvectorized and vectorized data alike - hence the
  /// template arguments.
  /// note that wgt[axis] contains a 'basis_functor' object (see basis.h)
  /// which encodes the generation of the set of weights.
  
  template < typename nd_rc_type , typename weight_type >
  void obtain_weights ( block_t < weight_type , 2 > & weight ,
                        const nd_rc_type & c ) const
  {
    const auto * ci = &(c[0]) ; // c.cbegin() ;
    for ( int axis = 0 ; axis < dimension ; ++ci , ++axis )
      wgt[axis] ( weight.data() + axis * spline_order , *ci ) ;
  }

  template < typename weight_type >
  void obtain_weights ( block_t < weight_type , 2 > & weight ) const
  {
    for ( int axis = 0 ; axis < dimension ; ++axis )
      wgt[axis] ( weight.data() + axis * spline_order ) ;
  }

  /// obtain weight for a single axis

  template < typename rc_type , typename weight_type >
  void obtain_weights ( weight_type * p_weight ,
                        const int & axis ,
                        const rc_type & c ) const
  {
    wgt[axis] ( p_weight , c ) ;
  }

  template < typename weight_type >
  void obtain_weights ( weight_type * p_weight ,
                        const int & axis ) const
  {
    wgt[axis] ( p_weight ) ;
  }

private:
  
  // next we have collateral code which we keep private.
  // TODO some of the above could also be private.
  
  // to be able to use the same code to access the coefficients, no matter
  // if the operation is vectorized or not, we provide 'load' functions
  // which encapsulate the memory access. This allows us to uniformly handle
  // vectorized and unvectorized data: the remainder of the code processes
  // unvectorized and vectorized data alike, and only when it comes to
  // fetching the coefficients from memory we need specialized code for
  // the memory access.
  
  // KFJ 2018-03-15 changed the load functions to expect a pointer to
  // cf_ele_type, access memory via this pointer, then convert to some
  // given target type, rather than load to some type derived from
  // cf_ele_type. We know the coefficients are always cf_ele_type, but
  // we usually want math_type for processing. And we don't want the
  // calling code having to be 'aware' of what cf_ele_type is at all.
  // With this change, the template argument introducing the coefficient
  // type could go from the eval code, and ATD via the arguments works.

  // Note that since inner_evaluator uniformly handles data as xel_t,
  // 'target' in the load functions below is always a xel_t, possibly
  // with only one element: we're using 'synthetic' types.
  
  /// load function for xel_t of fundamental T
  
  template < typename T , std::size_t N >
  static inline void load ( xel_t < T , N > & target ,
                            const cf_ele_type * const mem ,
                            const int & index )
  {
    for ( int i = 0 ; i < N ; i++ )
      target[i] = T ( mem [ index + i ] ) ;
  }
  
  // KFJ 2018-05-08 with the automatic use of vectorization the
  // distinction whether cf_ele_type is 'vectorizable' or not
  // is no longer needed: simdized_type will be a Vc::SimdArray
  // if possible, a simd_type otherwise.
  
  // dispatch, depending on whether cf_ele_type is the same as
  // what target contains. Usually the target will hold
  // 'math_ele_type', but for degree-0 splines, where the result
  // is directly derived from the coefficients, target holds
  // 'trg_ele_type'. We have the distinct cases first and the
  // dispatching routine below.
  
  template < typename target_t , typename index_t >
  static void load ( target_t & target ,
                     const cf_ele_type * const mem ,
                     const index_t & indexes ,
                     std::true_type
                   )
  {
    // static const size_t sz = index_t::size() ;
    static const size_t sz = get_ele_t < target_t > :: size ;
    for ( int e = 0 ; e < sz ; e++ )
    {
      // directly gather to 'target'
      target[e].gather ( mem + e , indexes ) ;
    }
  }
  
  template < typename target_t , typename index_t >
  static void load ( target_t & target ,
                     const cf_ele_type * const mem ,
                     const index_t & indexes ,
                     std::false_type
                   )
  {
    // static const size_t sz = index_t::size() ;
    static const size_t sz = get_ele_t < index_t > :: size ;
    simdized_type < cf_ele_type , sz > help ;
    static const size_t tsz = get_ele_t < target_t > :: size ;
    for ( int e = 0 ; e < tsz ; e++ )
    {
      // gather to 'help' and 'assign' to target, which affects
      // the necessary type transformation
      help.gather ( mem + e , indexes ) ;
      assign ( target[e] , help ) ;
    }
  }
  
  /// dispatch function for vectorized loads. We look at one criterion:
  /// - is cf_ele_type the same type as what the target object contains?
  
  template < typename target_t , typename index_t >
  static void inline load
  ( target_t & target ,
    const cf_ele_type * const mem ,
    const index_t & indexes )
  {
    // typedef typename target_t::value_type::value_type target_ele_type ;
    typedef typename get_ele_t < target_t > :: type target_ele_type ;
    
    load ( target , mem , indexes ,
           std::integral_constant
                < bool ,
                  std::is_same < cf_ele_type , target_ele_type > :: value
                > ()          
         ) ;
  }
  
  /// _eval is the workhorse routine and implements the recursive arithmetic
  /// needed to evaluate the spline. First the weights for the current dimension
  /// are obtained from the weights object passed in. Once the weights are known,
  /// they are successively multiplied with the results of recursively calling
  /// _eval for the next lower dimension and the products are summed up to produce
  /// the result value. The scheme of using a recursive evaluation has several
  /// benefits: it needs no explicit intermediate storage of partial sums
  /// (uses the stack instead) and it makes the process dimension-agnostic in an
  /// elegant way. Therefore, the code is also thread-safe. note that this routine
  /// is used for operation on braced splines, with the sequence of offsets to be
  /// visited fixed at the evaluator's construction.
  
  template < int level , class math1_type , class offset_type >
  struct _eval
  {
    inline
    void operator() ( const offset_type & locus ,
                      cf_pointer_iterator & cfp_iter ,
                      const block_t < math1_type , 2 > & weight ,
                      xel_t < math1_type , channels > & sum
                    ) const
    {
      const math1_type w ( weight [ { 0 , level } ] ) ;
      
      // recursively call _eval for the next lower level, receiving
      // the result in 'sum', and apply the first weight to it

      _eval < level - 1 , math1_type , offset_type >()
            ( locus , cfp_iter , weight , sum ) ;
      
      for ( int d = 0 ; d < channels ; d++ )
        sum[d] *= w ;

      // to pick up the result of further recursive calls:
      
      xel_t < math1_type , channels > subsum ;

      // now keep calling _eval for the next lower level, receiving
      // the result in 'subsum', and apply the corresponding weight.
      // Then add the weighted subsum to 'sum'.
      
      for ( int i = 1 ; i < weight.shape [ 0 ] ; i++ )
      {
        const math1_type w ( weight [ { i , level } ] ) ;

        _eval < level - 1 , math1_type , offset_type >()
              ( locus , cfp_iter , weight , subsum ) ;
        
      // KFJ 2019-02-12 tentative use of fma

#ifdef USE_FMA
        for ( int d = 0 ; d < channels ; d++ )
          sum[d] = fma ( subsum[d] , w , sum[d] ) ;
#else
        for ( int d = 0 ; d < channels ; d++ )
          subsum[d] *= w ;

        sum += subsum ;
#endif
      }
    }
  } ;

  /// at level 0 the recursion ends, now we finally apply the weights for axis 0
  /// to the window of coefficients. Note how cfp_iter is passed in per reference.
  /// This looks wrong, but it's necessary: When, in the course of the recursion,
  /// the level 0 routine is called again, it needs to access the next bunch of
  /// spline_order coefficients.
  ///
  /// Just incrementing the reference saves us incrementing higher up.
  /// This is the point where we access the spline's coefficients. Since _eval works
  /// for vectorized and unvectorized data alike, this access is coded as a call to
  /// 'load' which provides a uniform syntax for the memory access. The implementation
  /// of the load routines used here is just above.
  ///
  /// The access to the coefficients is a bit difficult to spot: they are accessed
  /// via cfp_iter. cfp_iter iterates over an array of readily offsetted pointers.
  /// These pointers point to all elements in a window of coefficients centered at
  /// the coefficients' base address. By adding 'locus' to one of these pointers,
  /// the resulting pointer now points to the element of a specific window of
  /// coefficients, namely that window where the coefficient subset for the current
  /// evaluation is located. locus may be a SIMD type, in which case it refers to
  /// several windows. 'locus' is the offset produced from the integral part of the
  /// coordinate(s) passed in, so it is the datum which provides the localization
  /// of the DDA, while the pointers coming from cfp_iter are constant throughout
  /// the evaluator's lifetime.
  
  template < class math1_type , class offset_type >
  struct _eval < 0 , math1_type , offset_type >
  {
    inline
    void operator() ( const offset_type & locus ,
                      cf_pointer_iterator & cfp_iter ,
                      const block_t < math1_type , 2 > & weight ,
                      xel_t < math1_type , channels > & sum
                    ) const
    {
      typedef xel_t < math1_type , channels > math_type ;
      
      const math1_type w ( weight [ { 0 , 0 } ] ) ;
      
      // initialize 'sum' by 'loading' a coefficient (or a set of
      // coefficients if we're running vector code) - then apply
      // the first (set of) weight(s).
  
      load ( sum , *cfp_iter , locus ) ;
      
      for ( int d = 0 ; d < channels ; d++ )
        sum[d] *= w ;
      
      ++cfp_iter ;

      // now keep on loading coefficients, apply corresponding weights
      // and add the weighted coefficients to 'sum'
      
      for ( int i = 1 ; i < weight.shape [ 0 ] ; i++ )
      {
        const math1_type w ( weight [ { i , 0 } ] ) ;
        
        math_type help ;
        load ( help , *cfp_iter , locus ) ;
        ++cfp_iter ;

      // KFJ 2019-02-12 tentative use of fma

#ifdef USE_FMA
        for ( int d = 0 ; d < channels ; d++ )
          sum[d] = fma ( help[d] , w , sum[d] ) ;
#else
        for ( int d = 0 ; d < channels ; d++ )
          help[d] *= w ;

        sum += help ;
#endif
      }
    }

  } ;

  /// specialized code for degree-1 b-splines, aka linear interpolation.
  /// here, there is no gain to be had from working with precomputed
  /// per-axis weights, the weight generation is trivial. So the
  /// specialization given here is faster than using the general _eval
  /// sequence, which otherwise works just as well for degree-1 splines.
  
  template < int level , class math1_type , class offset_type >
  struct _eval_linear
  {
    inline
    void operator() ( const offset_type& locus ,
                      cf_pointer_iterator & cfp_iter ,
                      const xel_t < math1_type , dimension > & tune ,
                      xel_t < math1_type , channels > & sum
                    ) const
    {
      const math1_type wl ( math1_type(1) - tune [ level ] ) ;
      const math1_type wr ( tune [ level ] ) ;

      _eval_linear < level - 1 , math1_type , offset_type >()
                   ( locus , cfp_iter , tune , sum ) ;

      for ( int d = 0 ; d < channels ; d++ )
        sum[d] *= wl ;
      
      xel_t < math1_type , channels > subsum ;
      
      _eval_linear < level - 1 , math1_type , offset_type >()
                   ( locus , cfp_iter , tune , subsum ) ;
      
      // KFJ 2019-02-12 tentative use of fma

#ifdef USE_FMA
      for ( int d = 0 ; d < channels ; d++ )
        sum[d] = fma ( subsum[d] , wr , sum[d] ) ;
#else
      for ( int d = 0 ; d < channels ; d++ )
        subsum[d] *= wr ;
      
      sum += subsum ;
#endif
    }  
  } ;

  /// again, level 0 terminates the recursion, again accessing the spline's
  /// coefficients with the 'load' function defined above.
  
  template < class math1_type , class offset_type >
  struct _eval_linear < 0 , math1_type , offset_type >
  {
    inline
    void operator() ( const offset_type & locus ,
                      cf_pointer_iterator & cfp_iter ,
                      const xel_t < math1_type , dimension > & tune ,
                      xel_t < math1_type , channels > & sum
                    ) const
    {
      const math1_type wl ( math1_type(1) - tune [ 0 ] ) ;
      const math1_type wr ( tune [ 0 ] ) ;
      
      load ( sum , *cfp_iter , locus ) ;
      ++cfp_iter ;
      
      for ( int d = 0 ; d < channels ; d++ )
        sum[d] *= wl ;
      
      xel_t < math1_type , channels > help ;
      
      load ( help , *cfp_iter , locus ) ;
      ++cfp_iter ;
      
      // KFJ 2019-02-12 tentative use of fma

#ifdef USE_FMA
      for ( int d = 0 ; d < channels ; d++ )
        sum[d] = fma ( help[d] , wr , sum[d] ) ;
#else
      for ( int d = 0 ; d < channels ; d++ )
        help[d] *= wr ;
      
      sum += help ;
#endif
    }
  } ;

public:
  
  // next we have the code which is called from 'outside'. In this section,
  // incoming coordinates are split into their integral and remainder part.
  // The remainder part is used to obtain the weights to apply to the spline
  // coefficients. The resulting data are then fed to the workhorse code above.
  // We have several specializations here depending on the degree of the spline.
  
  /// the 'innermost' eval routine is called with offset(s) and weights. This
  /// routine is public because it is used from outside (namely by grid_eval).
  /// In this final delegate we call the workhorse code in class _eval 
  
  // TODO: what if the spline is degree 0 or 1? for these cases, grid_eval
  // should not pick this general-purpose routine
  
  template < class result_type , class math1_type , class offset_type >
  inline
  void eval ( const offset_type& select ,
              const block_t < math1_type , 2 > & weight ,
              result_type & result
            ) const
  {
    // we need an *instance* of this iterator because it's passed into _eval
    // by reference and manipulated by the code there:
    
    cf_pointer_iterator cfp_iter = &(cf_pointers[0]) ;
    
    // now we can call the recursive _eval routine yielding the result
    // as math_type.
   
    typedef xel_t < math1_type , channels > math_type ;
    math_type _result ;
    
    _eval < level , math1_type , offset_type >()
          ( select , cfp_iter , weight , _result ) ;
    
    // finally, we assign to result, casting to 'result_type'. If _result
    // and result are of the same type, the compiler will optimize the
    // intermediate _result away.

    assign ( result , _result ) ;
  }
  
private:

  /// 'penultimate' eval starts from an offset to a coefficient window; here
  /// the nD integral index to the coefficient window has already been
  /// 'condensed' into a 1D offset into the coefficient array's memory.
  /// Here we have the specializations affected by the template argument 'specialize'
  /// which activates more efficient code for degree 0 (nearest neighbour) and degree 1
  /// (linear interpolation) splines. I draw the line here; one might add further
  /// specializations, but from degree 2 onwards the weights are reused several times
  /// so looking them up in a small table (as the general-purpose code for unspecialized
  /// operation does) should be more efficient (TODO test).
  ///
  /// we have three variants, depending on 'specialize'. first is the specialization
  /// for nearest-neighbour interpolation, which doesn't delegate further, since the
  /// result can be obtained directly by gathering from the coefficients:
  ///
  /// dispatch for nearest-neighbour interpolation (degree 0)
  /// this is trivial: we pick the coefficient(s) at position 'select' and directly
  /// convert to result_type, since the coefficient 'window' in this case has width 1
  /// in every direction, the 'weight' to apply is 1. The general-purpose code below
  /// would iterate over this width-1 window and apply the weight, which makes it
  /// slower since both is futile here.
  
  template < class result_type , class math1_type , class offset_type >
  inline
  void eval ( const offset_type& select ,
              const xel_t < math1_type , dimension > & tune ,
              result_type & result ,
              std::integral_constant < int , 0 >
            ) const
  {
    load ( result , cf_ebase , select ) ;
  }

  /// eval dispatch for linear interpolation (degree 1)
  /// again, we might use the general-purpose code below for this situation,
  /// but since the weights for linear interpolation are trivially computable,
  /// being 'tune' and 1 - 'tune', we use specialized workhorse code in
  /// _eval_linear, see there.
  
  template < class result_type , class math1_type , class offset_type >
  inline
  void eval ( const offset_type & select ,
              const xel_t < math1_type , dimension > & tune ,
              result_type & result ,
              std::integral_constant < int , 1 > ) const
  {
    cf_pointer_iterator cfp_iter = &(cf_pointers[0]) ;
    typedef xel_t < math1_type , channels > math_type ;
    
    math_type _result ;
    
    _eval_linear < level , math1_type , offset_type >()
                 ( select , cfp_iter , tune , _result ) ;
    
    assign ( result , _result ) ;
  }

  /// eval dispatch for arbitrary spline degrees
  /// here we have the general-purpose routine which works for
  /// arbitrary spline degrees (as long as the code in basis.h 
  /// can provide, which is currently up to degree 45). Here,
  /// the weights are calculated by accessing the b-spline basis
  /// function. With the weights at hand, we delegate to an
  /// overload of 'eval' accepting weights, see there.
  
  template < class result_type ,
             class math1_type ,
             class offset_type ,
             int arbitrary_spline_degree >
  inline
  void eval ( const offset_type& select ,
              const xel_t < math1_type , dimension > & tune ,
              result_type & result ,
              std::integral_constant < int , arbitrary_spline_degree > ) const
  {
    // 'weight' is a 2D block_t of math1_type:

    block_t < math1_type , 2 >
      weight ( { std::size_t ( spline_order ) ,
                 std::size_t ( dimension ) } ) ;

    // obtain_weights fills the block of weights:

    obtain_weights ( weight , tune ) ;

    // now we can proceed to 'eval proper'

    eval ( select , weight , result ) ;
  }

public:
  
  /// while class evaluator accepts the argument signature of a
  /// unary_functor, class inner_evaluator uses 'synthetic'
  /// types, which are always xel_t - possibly of just one element.
  /// This simplifies the code, since the 'singular' arguments don't have
  /// to be treated separately. The data are just the same in memory and
  /// class evaluator simply reinterpret_casts the arguments it receives
  /// to the 'synthetic' types. Another effect of moving to the 'synthetic'
  /// types is to 'erase' their type: any 'meaning' they may have, like
  /// std::complex etc., is removed - they are treated as 'bunches' of
  /// a fundamental type or a vector. The synthetic types are built using
  /// a combination of two template template arguments: 'bunch' and 'vector':
  /// - 'bunch', which forms an aggregate of several of a given
  ///   type, like a xel_t, which is currently the
  ///   only template used for the purpose.
  /// - 'vector', which represents an aggregate of several fundamentals
  ///   of equal type which will be processed with SIMD logic. Currently,
  ///   the templates used for the purpose are simd_type
  ///   (simulating SIMD operations with ordinary scalar code),
  ///   Vc::SimdArray,  which is a 'proper' SIMD type, and
  ///   scalar, which is used for unvectorized data.
  /// Note how 'vsize' is a template argument to this function, and
  /// not a template argument to class inner_evaluator. This is more
  /// flexible, the calling code can process any vsize with the same
  /// inner_evaluator.
  /// the last template argument is used to differentiate between
  /// 'normal' operation with real coordinates and access with discrete
  /// coordinates. Weight generation for discrete coordinates is easier,
  /// because when they are split into an integral part and a remainder,
  /// the remainder is always zero. From degree 2 up we still need to
  /// calculate weights for the evaluation, but we can use a simplified
  /// method for obtaining the weights. For degree 0 and 1, we need no
  /// weights at all, and so we can directly load and return the spline
  /// coefficients, which is just like the code for the nearest-neighbour
  /// evaluation variants, minus the coordinate splitting.
  /// First in line is the overload taking real coordinates:

  template < template < typename , int > class bunch ,
             template < typename , size_t > class vector ,
             size_t vsize >
  inline
  void eval ( const bunch < vector < rc_ele_type , vsize > , dimension >
                    & coordinate ,
                    bunch < vector < trg_ele_type , vsize > , channels >
                    & result ,
              std::false_type // not a discrete coordinate
            ) const
  {
    // derive the 'vectorized' types, depending on 'vector' and the
    // elementary types

    typedef vector < ic_ele_type , vsize > ic_ele_v ;
    typedef vector < rc_ele_type , vsize > rc_ele_v ;
    typedef vector < math_ele_type , vsize > math_ele_v ;
    typedef vector < ofs_ele_type , vsize > ofs_ele_v ;
    typedef vector < trg_ele_type , vsize > trg_ele_v ;
    
    // perform the coordinate split
    
    bunch < ic_ele_v , dimension > select ;
    bunch < rc_ele_v , dimension > _tune ;
    
    split ( coordinate , select , _tune ) ;
    
    // convert the remainders to math_type
    
    bunch < math_ele_v , dimension > tune ;
    assign ( tune , _tune ) ;
    
    // 'condense' the discrete nD coordinates into offsets

    // let's assume the spline is distributed to several tiles.
    // let's also assume that cf_pointers holds pointers suitable
    // for a tile. We split 'select' into two parts: the part
    // referring to a tile (by right-shifting the select values)
    // and the part referring to the in-tile position (by and-ing
    // with a mask). The first part can be used as an index into
    // the array of tiles, and there we must check whether the
    // tiles in question are currently in-memory or need to be
    // loaded first (note that select is vectorized), in which case
    // that has to be done first. With all needed tiles in-memory,
    // each of the tiles has a specific partial offset from the
    // reference (ex cf_pointers), let's call that ofs_tile,
    // and  there is a second partial offset ofs_in_tile, which
    // is derived from the second part of select and the strides
    // of the tile. Adding these two partial offsets yields a
    // combined value which can be passed to the code 'further
    // down the line' which can remain unchanged, as long as the
    // tiles have enough frame to cover the support of the spline
    // for all in-tile locations.
    // The most time-critical process here would be the test for
    // availability and possible loading from disk of tiles, and
    // this would need careful crafting to avoid frequent discards
    // and reloads. Overhead for the frames could be kept quite low
    // by using rather large tiles.

    ofs_ele_v origin = select[0] * ic_ele_type ( cf_estride [ 0 ] ) ;
    for ( int d = 1 ; d < dimension ; d++ )
      origin += select[d] * ic_ele_type ( cf_estride [ d ] ) ;
    
    // delegate, dispatching on 'specialize'
    
    eval ( origin , tune , result ,
           std::integral_constant < int , specialize > () ) ;
  }

  /// first ieval overload taking discrete coordinates, implying a
  /// 'delta' of zero.
  /// this overload is for 'uspecialized' evaluation. Here we use
  /// the call to obtain_weights without a delta, which simply takes
  /// only the first line of the weight matrix, which is precisely
  /// what obtain_weights would produce with a delta of zero.

  template < template < typename , int > class bunch ,
             template < typename , size_t > class vector ,
             size_t vsize >
  inline
  void ieval ( const bunch < vector < ic_ele_type , vsize > , dimension >
                    & select ,
                    bunch < vector < trg_ele_type , vsize > , channels >
                    & result ,
              std::false_type
            ) const
  {
    typedef vector < math_ele_type , vsize > math_ele_v ;

    block_t < math_ele_v , 2 >
      weight ( Shape2 ( spline_order , dimension ) ) ;

    obtain_weights ( weight ) ;

    typedef vector < ofs_ele_type , vsize > ofs_ele_v ;

    ofs_ele_v origin = select[0] * ic_ele_type ( cf_estride [ 0 ] ) ;
    for ( int d = 1 ; d < dimension ; d++ )
      origin += select[d] * ic_ele_type ( cf_estride [ d ] ) ;

    eval ( origin , weight , result ) ;
  }

  /// second ieval overload taking discrete coordinates, implying a
  /// 'delta' of zero.
  /// this overload is for evaluators specialized to 0 or 1, where we
  /// can simply load the coefficients and don't need weights: the
  /// support is so narrow that we needn't consider neighbouring
  /// coefficients

  template < template < typename , int > class bunch ,
             template < typename , size_t > class vector ,
             size_t vsize >
  inline
  void ieval ( const bunch < vector < ic_ele_type , vsize > , dimension >
                    & select ,
                    bunch < vector < trg_ele_type , vsize > , channels >
                    & result ,
              std::true_type
            ) const
  {
    typedef vector < ofs_ele_type , vsize > ofs_ele_v ;

    // 'condense' the discrete nD coordinates into offsets

    ofs_ele_v origin = select[0] * ic_ele_type ( cf_estride [ 0 ] ) ;
    for ( int d = 1 ; d < dimension ; d++ )
      origin += select[d] * ic_ele_type ( cf_estride [ d ] ) ;

    // load the data

    load ( result , cf_ebase , origin ) ;
  }

  /// this overload is taken for discrete coordinates, and dispatches
  /// again, depending on the evaluator's 'specialize' template argument,
  /// to one of two variants of 'ieval', above

  template < template < typename , int > class bunch ,
             template < typename , size_t > class vector ,
             size_t vsize >
  inline
  void eval ( const bunch < vector < ic_ele_type , vsize > , dimension >
                    & select ,
                    bunch < vector < trg_ele_type , vsize > , channels >
                    & result ,
              std::true_type
            ) const
  {
    static const bool just_load = ( specialize == 0 || specialize == 1 ) ;

    ieval < bunch , vector , vsize > ( select , result ,
                       std::integral_constant < bool , just_load > () ) ;
  }

  /// initial dispatch on whether incoming coordinates are discrete or real

  template < template < typename , int > class bunch ,
             template < typename , size_t > class vector ,
             size_t vsize >
  inline
  void eval ( const bunch < vector < rc_ele_type , vsize > , dimension >
                    & select ,
                    bunch < vector < trg_ele_type , vsize > , channels >
                    & result
            ) const
  {
    // check the coordinate's elementary type. Is it discrete?
    // dispatch accordingly.

    static const bool take_discrete_coordinates
      = std::is_integral < rc_ele_type > :: value ;

    eval < bunch , vector , vsize > ( select , result ,
           std::integral_constant < bool , take_discrete_coordinates > () ) ;
  }

} ; // end of inner_evaluator

} ; // namespace detail

// we define the default fundamental type used for mathematical operations.
// starting out with the 'better' of coordinate_type's and value_type's
// elementary type, we take this type's RealPromote type to make sure we're
// operating in some real type. With a real math_type we can operate on
// integral coefficients/values and only suffer from quantization errors,
// so provided the dynamic range of the integral values is 'sufficiently'
// large, this becomes an option - as opposed to operating in an integral
// type which is clearly not an option with the weights being in [0..1].

template < typename coordinate_type ,
           typename value_type
         >
// using default_math_type =
// typename NumericTraits
//   < typename PromoteTraits
//     < typename get_ele_t < coordinate_type > :: type ,
//       typename get_ele_t < value_type > :: type
//     > :: Promote
//   > :: RealPromote ;
using default_math_type
  = PROMOTE ( typename get_ele_t < coordinate_type > :: type ,
              typename get_ele_t < value_type > :: type ) ;

/// tag class used to identify all evaluator instantiations

struct bspline_evaluator_tag { } ;

/// class evaluator encodes evaluation of a spline-like object. This is a
/// generalization of b-spline evaluation, which is the default if no other
/// basis functions are specified. Technically, a evaluator is a
/// unary_functor, which has the specific capability of evaluating
/// a specific spline-like object. This makes it a candidate to be passed
/// to the functions in transform.h, like remap() and transform(), and it
/// also makes it suitable for zimt's functional constructs like chaining,
/// mapping, etc.
///
/// While creating and using evaluators is simple enough, especially from
/// bspline objects, there are also factory functions producing objects capable
/// of evaluating a b-spline. These objects are wrappers around a evaluator,
/// please see the factory functions make_evaluator() and make_safe_evaluator() at the
/// end of this file.
///
/// If you don't want to concern yourself with the details, the easiest way is to
/// have a bspline object handy and use one of the factory functions, assigning the
/// resulting functor to an auto variable:
///
/// // given a bspline object 'bspl'
/// // create an object (a functor) which can evaluate the spline
/// auto ev = make_safe_evaluator ( bspl ) ;
/// // which can be used like this:
/// auto value = ev ( real_coordinate ) ;
///
/// The evaluation relies on 'braced' coefficients, as they are provided by
/// a bspline object. While the most general constructor will accept
/// a raw pointer to coefficients (enclosed in the necessary 'brace'), this will rarely
/// be used, and an evaluator will be constructed from a bspline object. To create an
/// evaluator directly, the specific type of evaluator has to be established by providing
/// the relevant template arguments. We need at least two types: the 'coordinate type'
/// and the 'value type':
///
/// - The coordinate type is encoded as a xel_t of some real data type -
/// doing image processing, the typical type would be a xel_t < float , 2 >.
/// fundamental real types are also accepted (for 1D splines). There is also specialized
/// code for incoming discrete coordinates, which can be evaluated more quickly, So if
/// the caller only evaluates at discrete locations, it's good to actually pass the
/// discrete coordinates in, rather than converting them to some real type and passing
/// the real equivalent.
///
/// - The value type will usually be either a fundamental real data type such as 'float',
/// or a xel_t of such an elementary type. Other data types which can be
/// handled by vigra's ExpandElementResult mechanism should also work. When processing
/// colour images, your value type would typically be a xel_t < float , 3 >.
/// You can also process integer-valued data, in which case you may suffer from
/// quantization errors, so you should make sure your data cover the dynamic range of
/// the integer type used as best as possible (like by using the 'boost' parameter when
/// prefiltering the b-spline). Processing of integer-valued data is done using floating
/// point arithmetics internally, so the quantization error occurs when the result is
/// ready and assigned to an integer target type; if the target data type is real, the
/// result is precise (within arithmetic precision).
///
/// you can choose the data type which is used internally to do computations. per default,
/// this will be a real type 'appropriate' to the operation, but you're free to pick some
/// other type. Note that picking integer types for the purpose is *not* allowed.
/// The relevant template argument is 'math_ele_type'.
///
/// Note that class evaluator operates with 'native' spline coordinates, which run with
/// the coefficient array's core shape, so typically from 0 to M-1 for a 1D spline over
/// M values. Access with different coordinates is most easily done by 'chaining' class
/// evaluator objects to other unary_functor objects providing coordinate
/// translation, see unary_functor.h, map.h and domain.h.
///
/// The 'native' coordinates can be thought of as an extension of the discrete coordinates
/// used to index the spline's knot points. Let's assume you have a 1D spline over knot
/// points in an array 'a'. While you can index 'a' with discrete coordinates like 0, 1, 2...
/// you can evaluate the spline at real coordinates like 0.0, 1.5, 7.8. If a real coordinate
/// has no fractional part, evaluation of the spline at this coordinate will produce the
/// knot point value at the index which is equal to the real coordinate, so the interpolation
/// criterion is fulfilled. zimt does not (currently) provide code for non-uniform
/// splines. For such splines, the control points are not unit-spaced, and possibly also
/// not equally spaced. To evaluate such splines, it's necessary to perform a binary search
/// in the control point vector to locate the interval containing the coordinate in question,
/// subtract the interval's left hand side, then divide by the interval's length. This value
/// constitutes the 'fractional part' or 'delta' used for evaluation, while the interval's
/// ordinal number yields the integral part. Evaluation can then proceed as for uniform
/// splines. A popular subtype of such non-uniform splines are NURBS (non-uniform rational
/// b-splines), which have the constraints that they map a k-D manifold to a k+1-D space,
/// apply weights to the control points and allow coinciding control points. At the core
/// of a NURBS evaluation there is the same evaluation of a uniform spline, but all the
/// operations going on around that are considerable - especially the binary search to
/// locate the correct knot point interval (with it's many conditionals and memory
/// accesses) is very time-consuming.
///
/// While the template arguments specify coordinate and value type for unvectorized
/// operation, the types for vectorized operation are inferred from them, using zimt's
/// vector_traits mechanism. The width of SIMD vectors to be used can be chosen explicitly.
/// This is not mandatory - if omitted, a default value is picked.
///
/// With the evaluator's type established, an evaluator of this type can be constructed by
/// passing a bspline object to it's constructor. Usually, the bspline object will
/// contain data of the same value type, and the spline has to have the same number of
/// dimensions as the coordinate type. Alternatively, coefficients can be passed in as a
/// pointer into a field of suitably braced coefficients. It's okay for the spline to hold
/// coefficients of a different type: they will be cast to math_type during the evaluation.
/// 
/// I have already hinted at the evaluation process used, but here it is again in a nutshell:
/// The coordinate at which the spline is to be evaluated is split into it's integral part
/// and a remaining fraction. The integral part defines the location where a window from the
/// coefficient array is taken, and the fractional part defines the weights to use in calculating
/// a weighted sum over this window. This weighted sum represents the result of the evaluation.
/// Coordinate splitting is done with the method split(), which picks the appropriate variant
/// (different code is used for odd and even splines)
///
/// The generation of the weights to be applied to the window of coefficients is performed
/// by employing weight functors from basis.h. What's left to do is to bring all the components
/// together, which happens in class inner_evaluator. The workhorse code in the subclass _eval
/// takes care of performing the necessary operations recursively over the dimensions of the
/// spline.
///
/// a evaluator is technically a unary_functor. This way, it can be directly
/// used by constructs like chain and has a consistent interface which allows code
/// using evaluators to query it's specifics. Since evaluation uses no conditionals on the
/// data path, the whole process can be formulated as a set of templated member functions
/// using vectorized types or unvectorized types, so the code itself is vector-agnostic.
/// This makes for a nicely compact body of code inside class inner_evaluator, at the cost of
/// having to provide a bit of collateral code to make data access syntactically uniform,
/// which is done with inner_evaluator's 'load' method.
///
/// The evaluation strategy is to have all dependencies of the evaluation except for the actual
/// coordinates taken care of by the constructor - and immutable for the evaluator's lifetime.
/// The resulting object has no state which is modified after construction, making it thread-safe.
/// It also constitutes a 'pure' functor in a functional-programming sense, because it has
/// no mutable state and no side-effects, as can be seen by the fact that the 'eval' methods
/// are all marked const.
///
/// By providing the evaluation in this way, it becomes easy for calling code to integrate
/// the evaluation into more complex functors. Consider, for example, code which generates
/// coordinates with a functor, then evaluates a b-spline at these coordinates,
/// and finally subjects the resultant values to some postprocessing. All these processing
/// steps can be bound into a single functor, and the calling code can be reduced to polling
/// this functor until it has provided the desired number of output values. zimt has a
/// large body of code to this effect: the family of 'transform' functions. These functions
/// accept functors and 'wield' them over arrays of data. The functors *may* be of class
/// evaluator, but it's much nicer to have a generalized function which can use any
/// functor with the right signature, because it widens the scope of the transform faimily
/// to deal with any functor producing an m-dimensional output from an n-dimensional input.
/// zimt's companion program, pv, uses this method to create complex pixel pipelines
/// by 'chaining' functors, and then, finally, uses functions from zimt's transform
/// family to 'roll out' the pixel pipeline code over large arrays of data.
///
/// An aside: unary_functor, from which class evaluator inherits, provides
/// convenience code to use unary_functor objects with 'normal' function syntax. So if
/// you have an evaluator e, you can write code like y = e ( x ), which is equivalent
/// to the notation I tend to use: e ( x , y ) - this two-argument form, where the first
/// argumemt is a const reference to the input and the second one a reference to the output
/// has some technical advantages (easier ATD).
///
/// While the 'unspecialized' evaluator will try and do 'the right thing' by using general
/// purpose code fit for all eventualities, for time-critical operation there are
/// specializations which can be used to make the code faster:
///
/// - template argument 'specialize' can be set to 0 to forcibly use (more efficient) nearest
/// neighbour interpolation, which has the same effect as simply running with degree 0 but avoids
/// code which isn't needed for nearest neighbour interpolation (like the application of weights,
/// which is futile under the circumstances, the weight always being 1.0).
/// specialize can also be set to 1 for explicit n-linear interpolation. There, the weight
/// generation is very simple: the 1D case considers only two coefficients, which have weights
/// w and 1-w, respectively. This is better coded explicitly - the 'general' weight generation
/// code produces just the same weights, but not quite as quickly.
/// Any other value for 'specialize' will result in the general-purpose code being used.
///
/// Note how the default number of vector elements is fixed by picking the value
/// which vector_traits considers appropriate. There should rarely be a need to
/// choose a different number of vector elements: evaluation will often be the most
/// computationally intensive part of a processing chain, and therefore this choice is
/// sensible. But it's not mandatory. Just keep in mind that when building processing
/// pipelines with zimt, all their elements must use the *same* vectorization width.
/// When you leave it to the defaults to set 'vsize', you may get functors which differ in
/// vsize, and when you try to chain them, the code won't compile. So keep this in mind:
/// When building complex functors, pass an explicit value for vsize to all component
/// functors.
///
/// So here are the template arguments to class evaluator again, where the first two
/// are mandatory, while the remainder have defaults:
///
/// - _coordinate_type: type of a coordinate, where the spline is to be
///   evaluated. This can be either a fundamental type like float or double,
///   or a xel_t of as many elements as the spline has dimensions.
///   discrete coordinates are okay and produce specialized, faster code.
///
/// - _trg_type: this is the data type the evaluation will produce as it's
///   result. While internally all arithmetic is done in 'math_type', the
///   internal result is cast to _trg_type when it's ready. _trg_type may be
///   a fundamental type or any type known to vigra's ExpandElementResult
///   mechanism, like xel_t. It has to have as many channels as
///   the coefficients of the spline (or the knot point data).
///
/// - _vsize: width of SIMD vectors to use for vectorized operation.
///   While class inner_evaluator takes this datum as a template argument to
///   it's eval routine, here it's a template argument to the evaluator class.
///   This is so because class evaluator inherits from unary_functor,
///   which also requires a specific vector size, because otherwise type
///   erasure using std::functions would not be possible. While you may
///   choose arbitrary _vsize, only small multiples of the hardware vector
///   width of the target machine will produce most efficient code. Passing
///   1 here will result in unvectorized code.
///
/// - specialize can be used to employ more efficient code for degree-0 and
///   degree-1 splines (aka nearest-neighbour and linear interpolation). Pass
///   0 for degree 0, 1 for degree 1 and -1 for any other degree.
///
/// - _math_ele_type: elementary type to use for arithemtics in class
///   inner_evaluator. While in most cases default_math_type will be just right,
///   the default may be overridden. _math_ele_type must be a real data type.
///
/// - cf_type: data type of the coefficients of the spline. Normally this
///   will be the same as _trg_type above, but this is not mandatory. The
///   coefficients will be converted to math_type once they have been loaded
///   from memory, and all arithmetic is done in math_type, until finally the
///   result is converted to _trg_type.
///
/// - _mbf_type: This defines the basis functors. For b-splines, this
///   is a multi_bf_type of as many basis_functors as the
///   spline's dimensions. Making this type a template argument opens
///   the code up to the use of arbitrary basis functions, as long as all
///   the basis functors are of the same type. To use differently-typed
///   basis functors, erase their type using bf_grok_type (see above)

template < typename _coordinate_type ,
           typename _trg_type ,
           size_t _vsize = vector_traits < _trg_type > :: size ,
           int _specialize = -1 ,
           typename _math_ele_type
             = default_math_type < _coordinate_type , _trg_type > ,
           typename _cf_type = _trg_type ,
           typename _mbf_type
             = multi_bf_type < basis_functor < _math_ele_type > ,
                               get_ele_t<_coordinate_type>::size
                             >
         >
class evaluator
: public bspline_evaluator_tag ,
  public unary_functor < _coordinate_type , _trg_type , _vsize >
  // , public callable
         // < evaluator < _coordinate_type , _trg_type , _vsize ,
         //               _specialize , _math_ele_type , _cf_type , _mbf_type > ,
         //   _coordinate_type ,
         //   _trg_type ,
         //   _vsize
         // >
{

public:
  
  // pull in the template arguments

  typedef _coordinate_type coordinate_type ;
  typedef _cf_type cf_type ;
  typedef _math_ele_type math_ele_type ;
  typedef _trg_type trg_type ;
  typedef _mbf_type mbf_type ;
  
  // we figure out the elementary types and some enums which we'll
  // use to specify the type of 'inner_evaluator' we'll use. This is
  // the 'analytic' part of dealing with the types, inner_evaluator
  // does the 'synthetic' part.

  typedef int ic_ele_type ;
  typedef int ofs_ele_type ;
  
  typedef ET < coordinate_type > rc_ele_type ;
  typedef ET < cf_type > cf_ele_type ;
  typedef ET < trg_type > trg_ele_type ;
                    
  enum { vsize = _vsize } ;
  
  // we want to access facilities of the base class (unary_functor)
  // so we use a typedef for the base class. class evaluator's property of
  // being derived from unary_functor provides it's 'face' to
  // calling code, while it's inner_evaluator provides the implementation of
  // it's capabilities.

  typedef unary_functor < coordinate_type , trg_type , vsize > base_type ;

  enum { dimension = base_type::dim_in }  ;
  enum { level = dimension - 1 }  ;
  enum { channels = base_type::dim_out } ;
  enum { specialize = _specialize } ;
  
  // now we can define the type of the 'inner' evaluator.
  // we pass all elementary types we intend to use, plus the number of
  // dimensions and channels, and the 'specialize' parameter which activates
  // specialized code for degree-0 and -1 splines.

  typedef detail::inner_evaluator < ic_ele_type ,
                                    rc_ele_type ,
                                    ofs_ele_type ,
                                    cf_ele_type ,
                                    math_ele_type ,
                                    trg_ele_type ,
                                    dimension ,
                                    channels ,
                                    specialize ,
                                    mbf_type > inner_type ;

  // class evaluator has an object of this type as it's sole member. Note that
  // this member is 'const': it's created when the evaluator object is created,
  // and immutable afterwards. This allows the compiler to optimize well, and
  // it makes class evaluator a 'pure' functor in a functional-programming sense.

  const inner_type inner ;

private:

  /// 'feeder' function. This is private, since it performs potentially
  /// dangerous reinterpret_casts which aren't meant for 'the public',
  /// but only for use by the 'eval' methods below, which provide the
  /// interface expected of a unary_functor.
  /// The cast reinterprets the arguments as the corresponding
  /// 'synthetic' types, using the templates 'bunch' and 'vector'.
  /// The reinterpreted data are fed to 'inner'.

  template < template < typename , int > class bunch ,
             template < typename , size_t > class vector ,
             size_t VSZ ,
             typename in_type ,
             typename out_type >
  inline
  void feed ( const in_type & _coordinate ,
              out_type & _result ) const
  {
    typedef bunch < vector < rc_ele_type , VSZ > , dimension > rc_t ;

    typedef bunch < vector < trg_ele_type , VSZ > , channels > trg_t ;

    auto const & coordinate
      = reinterpret_cast < rc_t const & >
        ( _coordinate ) ;

    auto & result
      = reinterpret_cast < trg_t & >
        ( _result ) ;

    inner.template eval < bunch , vector , VSZ >
      ( coordinate , result ) ;
  }

public:
  
  /// unvectorized evaluation function. this is delegated to 'feed'
  /// above, which reinterprets the arguments as the 'synthetic' types
  /// used by class inner_evaluator.
  
  inline
  void eval ( const typename base_type::in_type & _coordinate ,
              typename base_type::out_type & _result ) const
  {
    feed < xel_t , scalar , 1 >
      ( _coordinate , _result ) ;
  }
  
  /// vectorized evaluation function. This is enabled only if vsize > 1
  /// to guard against cases where vsize is 1. Without the enable_if, we'd
  /// end up with two overloads with the same signature if vsize is 1.
  /// Again we delegate to 'feed' to reinterpret the arguments, this time
  /// passing simdized_type for 'vector'.

  template < typename = std::enable_if < ( vsize > 1 ) > >
  inline
  void eval ( const typename base_type::in_v & _coordinate ,
              typename base_type::out_v & _result ) const
  {
    feed < xel_t , simdized_type , vsize >
      ( _coordinate , _result ) ;
  }

  typedef xel_t < std::ptrdiff_t , dimension > shape_type ;
  typedef xel_t < int , dimension > derivative_spec_type ;

  /// class evaluator's constructors are used to initialize 'inner'.
  /// This first constructor overload will rarely be used by calling
  /// code; the commonly used overload is the next one down taking a
  /// bspline object.
  /// we create the 'multi_bf_type' object here in the c'tor, and
  /// pass it on to inner's c'tor. This way we can declare the copy
  /// in inner_evaluator const. Earlier versions of this code did pass
  /// the derivative specification through  to inner_evaluator, but with
  /// the generalization to process arbitrary basis functions, this does
  /// not make sense anymore, so now the multi_bf_type is built and passed
  /// here.
  
  evaluator ( const cf_ele_type * const cf_ebase ,
              const shape_type & cf_estride ,
              int spline_degree ,
              derivative_spec_type derivative  = derivative_spec_type ( 0 )
            )
  : inner ( cf_ebase ,
            cf_estride ,
            spline_degree ,
            mbf_type ( spline_degree , derivative )
          )
  { } ;

  /// This c'tor overload takes a const reference to a multi_bf_type
  /// object providing the basis functions.
  // TODO: infer spline degree from mbf

  evaluator ( const cf_ele_type * const cf_ebase ,
              const shape_type & cf_estride ,
              int spline_degree ,
              const mbf_type & mbf )
  : inner ( cf_ebase ,
            cf_estride ,
            spline_degree ,
            mbf
          )
  { } ;

  /// constructor taking a bspline object, and, optionally,
  /// a specification for derivatives of the spline and 'shift'.
  /// derivative: pass values other than zero here for an axis for which you
  /// want the derivative. Note that passing non-zero values for several axes
  /// at the same time will likely not give you the result you intend: The
  /// evaluation proceeds from the highest dimension to the lowest (z..y..x),
  /// obtaining the weights for the given axis by calling the basis_functor
  /// assigned to that axis. If any of these basis_functor objects provides
  /// weights to calculate a derivative, subsequent processing for another
  /// axis with a basis_functor yielding weights for derivatives would calculate
  /// the derivative of the derivative, which is not what you would normally want.
  /// So for multidimensional data, use a derivative specification only for one
  /// axis. If necessary, calculate several derivatives separately (each with
  /// their own evaluator), then multiply. See gsm.cc for an example.
  
  evaluator ( const bspline < cf_type , dimension > & bspl ,
              derivative_spec_type derivative = derivative_spec_type ( 0 ) ,
              int shift = 0
            )
  : evaluator ( (cf_ele_type*) ( bspl.core.data() ) ,
                channels * bspl.core.strides ,
                bspl.spline_degree + shift ,
                derivative
              )
  {
    // while the general constructor above has already been called,
    // we haven't yet made certain that a requested shift has resulted
    // in a valid evaluator. We check this now and throw an exception
    // if the shift was illegal.

    if ( ! bspl.shiftable ( shift ) )
      throw not_supported
       ( "insufficient frame size. the requested shift can not be performed." ) ;
  } ;
  
  /// This c'tor overload takes a const reference to a multi_bf_type
  /// object providing the basis functions.

  evaluator ( const bspline < cf_type , dimension > & bspl ,
              const mbf_type & mbf ,
              int shift = 0
            )
  : evaluator ( (cf_ele_type*) ( bspl.core.data() ) ,
                channels * bspl.core.stride() ,
                bspl.spline_degree + shift ,
                mbf
              )
  {
    // while the general constructor above has already been called,
    // we haven't yet made certain that a requested shift has resulted
    // in a valid evaluator. We check this now and throw an exception
    // if the shift was illegal.

    if ( ! bspl.shiftable ( shift ) )
      throw not_supported
       ( "insufficient frame size. the requested shift can not be performed." ) ;
  } ;
} ; // end of class evaluator
 
/// alias template to make the declaration of non-bspline evaluators
/// easier. Note that we fix the 'specialze' template parameter at -1
/// (no specialization) and pass the mbf type argument in third position.
/// 'abf' stands for 'alternative basis function'. One example of such
/// an alternative basis function is area_basis_functor - see
/// basis.h.

template < typename _coordinate_type ,
           typename _trg_type ,
           typename _mbf_type ,
           size_t _vsize = vector_traits < _trg_type > :: size ,
           typename _math_ele_type
             = default_math_type < _coordinate_type , _trg_type > ,
           typename _cf_type = _trg_type
         >
using abf_evaluator = evaluator < _coordinate_type ,
                                  _trg_type ,
                                  _vsize ,
                                  -1 ,
                                  _math_ele_type ,
                                  _cf_type ,
                                  _mbf_type
                                > ;

// in the next section we have the collateral code needed to implement
// the factory functions make_evaluator() and make_safe_evaluator().
// This code uses class evaluator as a unary_functor, so the
// objects which are produced by the factory functions can only handle
// a fixed vsize, in contrast to class inner_evaluator, which can
// process 'synthetic' arguments with a wider spectrum.

namespace detail
{

/// helper object to create a type-erased evaluator for
/// a given bspline object. The evaluator is specialized to the
/// spline's degree, so that degree-0 splines are evaluated with
/// nearest neighbour interpolation, degree-1 splines with linear
/// interpolation, and all other splines with general b-spline
/// evaluation. The resulting evaluator is 'grokked' to
/// erase it's type to make it easier to handle on the receiving
/// side: build_ev will always return a grok_type, not
/// one of the several possible evaluators which it produces
/// initially. Why the type erasure? Because a function can only
/// return one distinct type. With specialization for degree-0,
/// degre-1 and arbitrary spline degrees, there are three distinct
/// types of evaluator to take care of. If they are to be returned
/// as a common type, type erasure is the only way.

// KFJ 2019-07-11 bug fix:
// changed conditionals in build_ev and build_safe_ev to consider
// the sum of bspl.spline_degree and shift, instead of merely
// bspl.spline_degree. The specialization has to take the shift
// into account.

template < typename spline_type ,
           typename rc_type ,
           size_t _vsize ,
           typename math_ele_type ,
           typename result_type
         >
struct build_ev
{
  grok_type < bspl_coordinate_type < spline_type , rc_type > ,
                       result_type ,
                       _vsize >
  operator() ( const spline_type & bspl ,
               xel_t<int,spline_type::dimension> dspec
                = xel_t<int,spline_type::dimension> ( 0 ) ,
               int shift = 0
             )
  {
    typedef bspl_coordinate_type < spline_type , rc_type > crd_t ;
    typedef bspl_value_type < spline_type > value_type ;

    if ( bspl.spline_degree + shift == 0 )    
      return grok
            ( evaluator
                < crd_t , result_type , _vsize , 0 ,
                  math_ele_type , value_type >
                  ( bspl , dspec , shift )
            ) ;
    else if ( bspl.spline_degree + shift == 1 )    
      return grok
            ( evaluator
                < crd_t , result_type , _vsize , 1 ,
                  math_ele_type , value_type >
                  ( bspl , dspec , shift )
            ) ;
    else  
      return grok
            ( evaluator
                < crd_t , result_type , _vsize , -1 ,
                  math_ele_type , value_type >
                  ( bspl , dspec , shift )
            ) ;
  }
} ;

/// helper object to create a mapper object with gate types
/// matching a bspline's boundary conditions and extents matching the
/// spline's lower and upper limits. Please note that these limits
/// depend on the boundary conditions and are not always simply
/// 0 and N-1, as they are for, say, mirror boundary conditions.
/// see lower_limit() and upper_limit() in bspline.
///
/// gate types are inferred from boundary conditions like this:
///
/// PERIODIC -> periodic_gate
/// MIRROR, REFLECT -> mirror_gate
/// all other boundary conditions -> clamp_gate
///
/// The mapper object is chained to an evaluator, resulting in
/// a functor providing safe access to the evaluator. The functor
/// is subsequently 'grokked' to produce a uniform return type.
///
/// Please note that this is only one possible way of dealing with
/// out-of-bounds coordinates: they are mapped into the defined range
/// in a way that is coherent with the boundary conditions. If you
/// need other methods you'll have to build your own functional
/// construct.
///
/// While build_ev (above) had three distinct types to deal with,
/// here, the number of potential types is even larger: every distinct
/// boundary condition along every distinct axis will result in a specfic
/// type of 'gate' object. So again we use type erasure to provide a
/// common return type, namely grok_type.

template < int level ,
           typename spline_type ,
           typename rc_type ,
           size_t _vsize ,
           typename math_ele_type ,
           typename result_type ,
           class ... gate_types >
struct build_safe_ev
{
  grok_type < bspl_coordinate_type < spline_type , rc_type > ,
                       result_type ,
                       _vsize >
  operator() ( const spline_type & bspl ,
               gate_types ... gates ,
               xel_t<int,spline_type::dimension> dspec
                = xel_t<int,spline_type::dimension> ( 0 ) ,
               int shift = 0
             )
  {
    // find out the spline's lower and upper limit for the current level

    rc_type lower ( bspl.lower_limit ( level ) ) ;
    rc_type upper ( bspl.upper_limit ( level ) ) ;

    // depending on the spline's boundary condition for the current
    // level, construct an appropriate gate object and recurse to
    // the next level. If the core's shape along this axis (level)
    // is 1, always clamp to zero. Note how for BC NATURAL the coordinate
    // is also clamped, because we can't produce the point-mirrored
    // continuation of the signal with a coordinate manipulation.

    auto bc = bspl.bcv [ level ] ;
    
    if ( bspl.core.shape [ level ] == 1 )
    {
      bc = CONSTANT ;
      lower = upper = rc_type ( 0 ) ;
    }

    switch ( bc )
    {
      case PERIODIC:
      {
        auto gt = periodic < rc_type , _vsize >
                  ( lower , upper ) ;
        return build_safe_ev < level - 1 , spline_type , rc_type , _vsize ,
                               math_ele_type , result_type ,
                               decltype ( gt ) , gate_types ... >()
              ( bspl , gt , gates ... , dspec , shift ) ; 
        break ;
      }
      case MIRROR:
      case REFLECT:
      {
        auto gt = mirror < rc_type , _vsize >
                  ( lower , upper ) ;
        return build_safe_ev < level - 1 , spline_type , rc_type , _vsize ,
                               math_ele_type , result_type ,
                               decltype ( gt ) , gate_types ... >()
              ( bspl , gt , gates ... , dspec , shift ) ; 
        break ;
      }
      default:
      {
        auto gt = clamp < rc_type , _vsize >
                  ( lower , upper , lower , upper ) ;
        return build_safe_ev < level - 1 , spline_type , rc_type , _vsize ,
                               math_ele_type , result_type ,
                               decltype ( gt ) , gate_types ... >()
              ( bspl , gt , gates ... , dspec , shift ) ; 
        break ;
      }
    }
  } 
} ;

/// at level -1, there are no more axes to deal with, here the recursion
/// ends and the actual mapper object is created. Specializing on the
/// spline's degree (0, 1, or indeterminate), an evaluator is created
/// and chained to the mapper object. The resulting functor is grokked
/// to produce a uniform return type, which is returned to the caller.

template < typename spline_type ,
           typename rc_type ,
           size_t _vsize ,
           typename math_ele_type ,
           typename result_type ,
           class ... gate_types >
struct build_safe_ev < -1 , spline_type , rc_type , _vsize ,
                       math_ele_type , result_type , gate_types ... >
{
  grok_type < bspl_coordinate_type < spline_type , rc_type > ,
                       result_type ,
                       _vsize >
  operator() ( const spline_type & bspl ,
               gate_types ... gates ,
               xel_t<int,spline_type::dimension> dspec
                = xel_t<int,spline_type::dimension> ( 0 ) ,
               int shift = 0
             )
  {
    typedef bspl_coordinate_type < spline_type , rc_type > crd_t ;
    typedef bspl_value_type < spline_type > value_type ;

    if ( bspl.spline_degree + shift == 0 )    
      return grok
            (   mapper
                  < crd_t , _vsize , gate_types... > ( gates ... )
              + evaluator
                  < crd_t , result_type , _vsize , 0 ,
                    math_ele_type , value_type >
                    ( bspl , dspec , shift )
            ) ;
    else if ( bspl.spline_degree + shift == 1 )    
      return grok
            (   mapper
                  < crd_t , _vsize , gate_types... > ( gates ... )
              + evaluator
                  < crd_t , result_type , _vsize , 1 ,
                    math_ele_type , value_type >
                    ( bspl , dspec , shift )
            ) ;
    else  
      return grok
            (   mapper
                  < crd_t , _vsize , gate_types... > ( gates ... )
              + evaluator
                  < crd_t , result_type , _vsize , -1 ,
                    math_ele_type , value_type >
                    ( bspl , dspec , shift )
            ) ;
  }
} ;

} ; // namespace detail

/// make_evaluator is a factory function, producing a functor
/// which provides access to an evaluator object. Evaluation
/// using the resulting object is *not* intrinsically safe,
/// it's the user's responsibility not to pass coordinates
/// which are outside the spline's defined range. If you need
/// safe access, see 'make_safe_evaluator' below. 'Not safe'
/// in this context means that evaluation at out-of-bounds
/// locations may result in a memory fault or produce wrong or
/// undefined results. Note that zimt's bspline objects
/// are set up as to allow evaluation at the lower and upper
/// limit of the spline and will also tolerate values 'just
/// outside' the bounds to guard against quantization errors.
/// see bspline for details.
///
/// The evaluator will be specialized to the spline's degree:
/// degree 0 splines will be evaluated with nearest neighbour,
/// degree 1 splines with linear interpolation, all other splines
/// will use general b-spline evaluation.
///
/// This function returns the evaluator wrapped in an object which
/// hides it's type. This object only 'knows' what coordinates it
/// can take and what values it will produce. The extra level of
/// indirection may cost a bit of performance, but having a common type
/// simplifies handling. The wrapped evaluator also provides operator().
///
/// So, if you have a bspline object 'bspl', you can use this
/// factory function like this:
///
/// auto ev = make_evaluator ( bspl ) ;
/// typedef typename decltype(ev)::in_type coordinate_type ;
/// coordinate_type c ;
/// auto result = ev ( c ) ;
///
/// make_evaluator requires one template argument: spline_type, the
/// type of the bspline object you want to have evaluated.
/// Optionally, you can specify the elementary type for coordinates
/// (use either float or double) and the vectorization width. The
/// latter will only have an effect if vectorization is used and
/// the spline's data type can be vectorized. Per default, the
/// vectorization width will be inferred from the spline's value_type
/// by querying vector_traits, which tries to provide a
/// 'good' choice. Note that a specific evaluator will only be capable
/// of processing vectorized coordinates of precisely the _vsize it
/// has been created with. A recent addition to the template argument
/// list is 'math_ele_type', which allows you to pick a different type
/// for internal processing than the default. The default is a real
/// type 'appropriate' to the data in the spline.
///
/// Note that the object created by this factory function will
/// only work properly if you evaluate coordinates of the specific
/// 'rc_type' used. If you create it with the default rc_type, which
/// is float (and usually sufficiently precise for a coordinate), you
/// can't evaluate double precision coordinates with it.
///
/// On top of the bspline object, you can optionally pass a derivative
/// specification and a shift value, which are simply passed through
/// to the evaluator's constructor, see there for the meaning of these
/// optional parameters.
///
/// While the declaration of this function looks frightfully complex,
/// using it is simple: in most cases it's simply
///
/// auto ev = make_evaluator ( bspl ) ;
///
/// For an explanation of the template arguments, please see
/// make_safe_evaluator() below, which takes the same template args.

// KFJ 2019-02-03 modified rc_type's default to be the real promote
// of the spline's elementary type

template < class spline_type ,
           typename rc_type = float ,
           size_t _vsize = vector_traits
                         < typename spline_type::value_type > :: size ,
           typename math_ele_type
             = default_math_type < typename spline_type::value_type ,
                                   rc_type > ,
           typename result_type = typename spline_type::value_type >
grok_type < bspl_coordinate_type < spline_type , rc_type > ,
                     result_type ,
                     _vsize >
make_evaluator ( const spline_type & bspl ,
                 xel_t<int,spline_type::dimension> dspec
                  = xel_t<int,spline_type::dimension> ( 0 ) ,
                 int shift = 0
               )
{
  typedef typename spline_type::value_type value_type ;
  typedef typename get_ele_t < value_type > :: type ele_type ;
  enum { vsize = _vsize } ;
  return detail::build_ev < spline_type , rc_type , _vsize ,
                            math_ele_type , result_type >()
    ( bspl , dspec , shift ) ;
}

/// make_safe_evaluator is a factory function, producing a functor
/// which provides safe access to an evaluator object. This functor
/// will map incoming coordinates into the spline's defined range,
/// as given by the spline with it's lower_limit and upper_limit
/// methods, honoring the bspline objects's boundary conditions.
/// So if, for example, the spline is periodic, all incoming
/// coordinates are valid and will be mapped to the first period.
/// Note the use of lower_limit and upper_limit. These values
/// also depend on the spline's boundary conditions, please see
/// class bspline for details. If there is no way to
/// meaningfully fold a coordinate into the defined range, the
/// coordinate is clamped to the nearest limit.
///
/// The evaluator will be specialized to the spline's degree:
/// degree 0 splines will be evaluated with nearest neighbour,
/// degree 1 splines with linear interpolation, all other splines
/// will use general b-spline evaluation.
///
/// This function returns the functor wrapped in an object which
/// hides it's type. This object only 'knows' what coordinates it
/// can take and what values it will produce. The extra level of
/// indirection costs a bit of performance, but having a common type
/// simplifies handling: the type returned by this function only
/// depends on the spline's data type, the coordinate type and
/// the vectorization width.
///
/// Also note that the object created by this factory function will
/// only work properly if you evaluate coordinates of the specific
/// 'rc_type' used. If you create it with the default rc_type, which
/// is float (and usually sufficiently precise for a coordinate), you
/// can't evaluate double precision coordinates with it.
///
/// On top of the bspline object, you can optionally pass a derivative
/// specification and a shift value, which are simply passed through
/// to the evlauator's constructor, see there for the meaning of these
/// optional parameters.
///
/// While the declaration of this function looks frightfully complex,
/// using it is simple: in most cases it's simply
///
/// auto ev = make_safe_evaluator ( bspl ) ;
///
/// The first template argument, spline_type, is the type of a
/// bspline object. This template argument has no default,
/// since it determines the dimensionality and the coefficient type.
/// But since the first argument to this factory function is of
/// this type, spline_type can be fixed via ATD, so it can be
/// omitted.
///
/// The second template argument, rc_type, can be used to pick a
/// different elementary type for the coordinates the evaluator will
/// accept. In most cases the default, float, will be sufficient.
///
/// The next template argument, _vsize, fixes the vectorization width.
/// Per default, this will be what zimt deems appropriate for the
/// spline's coefficient type.
///
/// math_ele_type can be used to specify a different fundamental type
/// to be used for arithemtic operations during evaluation. The default
/// used here is a real type of at least the precision of the coordinates
/// or the spline's coefficients, but you may want to raise precision
/// here, for example by passing 'double' while your data are all float.
///
/// Finally you can specify a result type. Per default the result will
/// be of the same type as the spline's coefficients, but you may want
/// a different value here - a typical example would be a spline with
/// integral coefficients, where you might prefer to get the result in,
/// say, float to avoid quantization errors on the conversion from the
/// 'internal' result (which is in math_type) to the output.

template < class spline_type ,
           typename rc_type = float ,
           size_t _vsize = vector_traits
                         < typename spline_type::value_type > :: size ,
           typename math_ele_type
             = default_math_type < typename spline_type::value_type ,
                                   rc_type > ,
           typename result_type = typename spline_type::value_type
         >
grok_type < bspl_coordinate_type < spline_type , rc_type > ,
                     result_type ,
                     _vsize >
make_safe_evaluator ( const spline_type & bspl ,
                      xel_t<int,spline_type::dimension> dspec
                       = xel_t<int,spline_type::dimension> ( 0 ) ,
                      int shift = 0
                    )
{
  typedef typename spline_type::value_type value_type ;
  // typedef typename ExpandElementResult < value_type > :: type ele_type ;
  typedef get_ele_t < value_type > ele_type ;
  enum { vsize = _vsize } ;
  return detail::build_safe_ev < spline_type::dimension - 1 ,
                                 spline_type ,
                                 rc_type ,
                                 _vsize ,
                                 math_ele_type ,
                                 result_type >() ( bspl , dspec , shift ) ;
} ;

END_ZIMT_SIMD_NAMESPACE
HWY_AFTER_NAMESPACE() ;

#endif // sentinel
