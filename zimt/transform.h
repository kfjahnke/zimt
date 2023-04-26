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

/*! \file transform.h
   
    \brief set of generic transform and apply functions
   
    zimt::transform is similar to std::transform, but uses different
    arguments: the data are passed as zimt::view_t and the functor
    to process the data is a zimt::unary_functor. Given:
   
    - c, an array containing input values
    - t, the target array, with the same shape as c
    - f, a unary functor converting input to output values
    - j, a coordinate into c and t
   
    transform performs the operation
   
    t[j] = f(c[j]) for all j
   
    There is also a variant of 'transform' which doesn't take an input
    array. Instead, for every target location, the location's discrete
    coordinates are passed to the unary_functor type object. Given:
   
    - f, a unary functor converting discrete coordinates to output values
    - j, a discrete coordinate into t
    - t, the target array
   
    the 'index-based' transform performs the operation
   
    t[j] = f(j) for all j

    Another variation is zimt::apply, which is a transform where input
    and output array are the same. The functor is applied to every value
    in the array. Given:

    - c, an array containing values
    - f, a unary functor converting input to output values
    - j, a coordinate into c

    apply performs the operation

    c[j] = f(c[j]) for all j

    Some specifics of the transformation can be modified by the parameter
    'bill', see it's source code for information.

    the transform family of functions can also be used for reductions;
    this requires additional coding effort in the zimt::unary_functor:
    it has to gather it's reduction result in the functor and have a
    way to communicate this (partial) result to the caling code in a
    tread-safe manner when it's destructed. functors for reduction
    also need to provide a 'capped eval' overload, see wielding.h for
    more about this topic.
*/

#ifndef ZIMT_TRANSFORM_H
#define ZIMT_TRANSFORM_H

// The bulk of the implementation of zimt's two 'transform' functions
// is now in wielding.h:

#include "wielding.h"

namespace zimt
{
template < typename view_type ,
           typename = std::enable_if
             < std::is_base_of < view_flag , view_type > :: value > >
view_type sort_strides ( const view_type & rhs , bool ascending = true )
{
  static const size_t dimension = view_type::dimension ;
  if ( dimension <= 1 )
    return rhs ;

  auto strides = view_type::strides ;
  auto shape = view_type::shape ;

  bool ordered = true ;

  // strategy: large strides wil 'trickle upwards', every iteration
  // of the outer loop will move the largest stride to the last
  // position (dmax-1). The next iteration will deal with the remaining
  // values. If no swaps were performed, the order is correct already.

  auto cmp =   ascending
             ? [] ( long a , long b ) { return a > b ; }
             : [] ( long a , long b ) { return a < b ; } ;

  for ( std::size_t dmax = dimension ; dmax > 1 ; dmax-- )
  {
    for ( std::size_t d = 1 ; d < dmax ; d++ )
    {
      if ( cmp ( strides [ d-1 ] , strides [ d ] ) )
      {
        std::swap ( strides [ d-1 ] , strides [ d ] ) ;
        std::swap ( shape [ d-1 ] , shape [ d ] ) ;
        ordered = false ;
      }
    }
    if ( ordered )
      break ;
  }
  view_type ( rhs , strides , shape ) ;
}

/// implementation of two-array transform using zimt::process.
///
/// 'array-based' transform takes two template arguments:
///
/// - the dimensionality of the input and output array
///
/// - 'unary_functor_type', which is a class satisfying the interface
///   laid down in unary_functor.h. Typically, this would be a type 
///   inheriting from zimt::unary_functor, but any type will do as
///   long as it provides the required typedefs and an the relevant
///   eval() routines.
///
/// this overload of transform takes at least three parameters:
///
/// - a reference to a const unary_functor_type object providing the
///   functionality needed to generate values from arguments.
///
/// - a zimt::view_t referring to input values. It has to have the same
///   shape as the target array and contain data of the unary_functor's
///   'in_type'.
///
/// - a zimt::view_t referring to output values. This is where the
///   resulting data are put, so it has to contain data of the
///   unary_functor's 'out_type'.
///
///   If you use zimt::view_t objects to pass the data, you can omit
///   the template arguments and rely on ATD. If you pass initializers,
///   you will need to pass the dimension.
///
/// trailing the argument list: all transform overloads have an additional
/// parameter 'bill' which serves to fine-tune the operation - see the
/// documentation of struct bill_t.

// TODO: the data might be tested for their layout im memory, and if
// they are contiguous - or subdimensional slices are contiguous - the
// view(s) might be reshaped to make processing more efficient.

template < std::size_t dimension , typename act_t >
void transform ( const act_t & act ,
                 const view_t < dimension ,
                                typename act_t::in_type
                              > input ,
                 view_t < dimension ,
                          typename act_t::out_type
                        > output ,
                 bill_t bill = bill_t()
               )
{
  // check shape compatibility
  
  if ( output.shape != input.shape )
  {
    throw zimt::shape_mismatch
     ( "transform: the shapes of the input and output array do not match" ) ;
  }

  auto shape = input.shape ;

  typedef typename act_t::in_ele_type in_ele_type ;
  typedef typename act_t::out_ele_type out_ele_type ;
  static const std::size_t chn_in = act_t::dim_in ;
  static const std::size_t chn_out = act_t::dim_out ;
  static const std::size_t vsize = act_t::vsize ;

  // confine the bill to sensible values

  if (    bill.segment_size <= 0
       || bill.segment_size > shape [ bill.axis ])
    bill.segment_size = shape [ bill.axis ] ;

  if ( bill.njobs <= 1 )
    bill.njobs = 1 ;

  if ( bill.njobs > zimt::default_njobs )
    bill.njobs = zimt::default_njobs ;

  // create a storer object to accept output from the act functor

  storer < out_ele_type , chn_out , dimension , vsize >
    put ( output , bill ) ;

  // input is provided with a loader or unstrided_loader object

  if ( input.strides [ bill.axis ] == 1 )
  {
    unstrided_loader < in_ele_type , chn_in , dimension , vsize >
      get ( input , bill ) ;

    process ( shape , get , act , put , bill ) ;
  }
  else
  {
    loader < in_ele_type , chn_in , dimension , vsize >
      get ( input , bill ) ;

    process ( shape , get , act , put , bill ) ;
  }
}

/// implementation of index-based transform using process
///
/// this overload of transform() is very similar to the first one,
/// but instead of picking input from an array, it feeds the discrete
/// coordinates of the successive places data should be rendered to
/// as input to the unary_functor_type object.

/// This sounds complicated, but is really quite simple. Let's assume
/// you have a 2X3 output array to fill with data. When this array is
/// passed to transform, the functor will be called with every
/// coordinate pair in turn, and the result the functor produces is
/// written to the output array. So for the example given,
/// with 'ev' being the functor, we have this set of operations:
///
/// output [ ( 0 , 0 ) ] = ev ( ( 0 , 0 ) ) ;
///
/// output [ ( 1 , 0 ) ] = ev ( ( 1 , 0 ) ) ;
///
/// output [ ( 2 , 0 ) ] = ev ( ( 2 , 0 ) ) ;
///
/// output [ ( 0 , 1 ) ] = ev ( ( 0 , 1 ) ) ;
///
/// output [ ( 1 , 1 ) ] = ev ( ( 1 , 1 ) ) ;
///
/// output [ ( 2 , 1 ) ] = ev ( ( 2 , 1 ) ) ;
///
/// this transform overload takes two template arguments:
///
/// - the dimensionality of the view/array
///
/// - 'act_t', which is a class satisfying the interface laid
///   down in unary_functor.h. This is an object which can provide values
///   given *discrete* coordinates, like class evaluator, but generalized
///   to allow for arbitrary ways of achieving it's goal. The unary functor's
///   'in_type' determines the number of dimensions of the coordinates - since
///   they are coordinates into the target array, the functor's input type
///   has to have the same number of channels as the target has dimensions.
///   The functor's 'out_type' has to be the same as the data type of the
///   target array, since the target array stores the results of calling the
///   functor.
///
/// this transform overload takes three parameters:
///
/// - a reference to a const unary_functor_type object providing the
///   functionality needed to generate values from discrete coordinates
///
/// - a reference to a zimt::view_t to use as a target. This is where the
///   resulting data are put.
///
/// - optionally, a bill_t object with additional parameters
///
/// Please note that zimt holds with vigra's coordinate handling convention,
/// which puts the fastest-changing index first. In a 2D, image processing,
/// context, this is the column index, or the x coordinate. C and C++ do
/// instead put this index last when using multidimensional array access code.
/// I think of the index ordering as 'latin book order': character follows
/// character in a line, line follows line in a page and page follows page
/// in a book... and you write coordinates as (column, line, page, ...)
/// Of course, if you invert the strides from their 'standard' order
/// (ascending in size) to, e.g, descending in size, the meaning of the
/// coordinates is different, but still the first component will use the
/// first stride etc.. view_t has the 'offset' function which transforms
/// a multi-dimensional coordinate into an offset. This is simply coded
/// as (coordinate * strides).sum()
///
/// transform can be used without template arguments, they will be inferred
/// by ATD from the arguments if that is possible: if you actually pass
/// a view_t, the dimensionality can be inferred, but if you pass a
/// std::initilaizer_sequence to create a view_t 'on the fly', you must
/// pass the dimensionality explicitly.

template < std::size_t dimension , class act_t >
void transform ( const act_t & act ,
                 view_t < dimension ,
                          typename act_t::out_type
                        > output ,
                 bill_t bill = bill_t() )
{
  auto shape = output.shape ;

  typedef typename act_t::in_ele_type in_ele_type ;
  typedef typename act_t::out_ele_type out_ele_type ;
  static const std::size_t chn_in = act_t::dim_in ;
  static const std::size_t chn_out = act_t::dim_out ;
  static const std::size_t vsize = act_t::vsize ;

  // confine the bill to sensible values

  if (    bill.segment_size <= 0
       || bill.segment_size > shape [ bill.axis ])
    bill.segment_size = shape [ bill.axis ] ;

  if ( bill.njobs <= 1 )
    bill.njobs = 1 ;

  if ( bill.njobs > zimt::default_njobs )
    bill.njobs = zimt::default_njobs ;

  // now delegate to the wielding code. We need two additional
  // parameters: 'get' generates coordinates as input to 'act',
  // and 'put' disposes of the result by storing to 'trg'.
  // The coordinate-generation produces discrete coordinates,
  // in the type act takes as input. So, act can take, e.g.
  // float coordinates and 'get' will provide.

  get_crd < in_ele_type , chn_in , dimension , vsize >
    get ( bill ) ;

  storer < out_ele_type , chn_out , dimension , vsize >
    put ( output , bill ) ;

  process ( shape , get , act , put , bill ) ;
}

// for 1D index-based transforms, we add an overload taking a naked
// pointer, stride and length. Then we delegate to the version above.

template < class act_t >
void transform ( const act_t & act ,
                 typename act_t::out_type * trg ,
                 long stride ,
                 std::size_t length ,
                 bill_t bill = bill_t() )
{
  typedef zimt::view_t < 1 , typename act_t::out_type > v_t ;
  v_t v ( trg , { stride } , { length } ) ;

  transform ( act , v , bill ) ;
}

/// we code 'apply' as a special variant of 'transform' where the output
/// is also used as input, so the effect is to feed the unary functor with
/// each 'output' value in turn, let it process it and store the result
/// back to the same location. While this looks like a rather roundabout
/// way of performing an apply, it has the advantage of using the same
/// type of functor (namely one with const input and writable output),
/// rather than a different functor type which modifies it's argument
/// in-place. While, at this level, using such a functor looks like a
/// feasible idea, It would require specialized code 'further down the
/// line' when complex functors are built with zimt's functional
/// programming tools: the 'apply-capable' functors would need to read
/// the output values first and write them back after anyway, resulting
/// in the same sequence of loads and stores as we get with the current
/// 'fake apply' implementation.
/// From the direct delegation to the two-array overload of 'transform',
/// you can see that this is merely syntactic sugar.

template < std::size_t dimension ,
           typename unary_functor_type >
void apply ( const unary_functor_type & ev ,
             view_t < dimension ,
                      typename unary_functor_type::out_type >
                    output ,
             bill_t bill = bill_t()
           )
{
  // make sure the functor's input and output type are the same

  static_assert ( std::is_same < typename unary_functor_type::in_type ,
                                 typename unary_functor_type::out_type > :: value ,
                  "apply: functor's input and output type must be the same" ) ;

  // delegate to transform

  transform ( ev , output , output , bill ) ;
}

} ; // end of namespace zimt

#endif // ZIMT_TRANSFORM_H
