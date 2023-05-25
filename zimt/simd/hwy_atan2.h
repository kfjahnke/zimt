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
/************************************************************************/

#ifndef HWY_ATAN2_H

// This file can go once the highway implementation of atan2 'trickles
// down'.

// This file has a port af Vc's atan2 implementation to highway.
// The implementation of atan2 which I have ported to highway was originally
// found in https://github.com/VcDevel/Vc/blob/1.4/src/trigonometric.cpp
// That code is licensensed like this:

/*  This file is part of the Vc library. {{{
Copyright © 2012-2015 Matthias Kretz <kretz@kde.org>

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the names of contributing organizations nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

}}}*/

// some code irrelevant to the port was deleted here.

/* this is the original code of Vc's atan2 implementation, begin quote

template <>
template <>
Vc::float_v Trigonometric<Vc::Detail::TrigonometricImplementation<
    Vc::CurrentImplementation::current()>>::atan2(const Vc::float_v &y,
                                                  const Vc::float_v &x)
{
    using V = Vc::float_v;
    typedef Const<float, V::abi> C;
    typedef V::Mask M;

    const M xZero = x == V::Zero();
    const M yZero = y == V::Zero();
    const M xMinusZero = xZero && isnegative(x);
    const M yNeg = y < V::Zero();
    const M xInf = !isfinite(x);
    const M yInf = !isfinite(y);

    V a = copysign(C::_pi(), y);
    a.setZero(x >= V::Zero());

    // setting x to any finite value will have atan(y/x) return sign(y/x)*pi/2, just in case x is inf
    V _x = x;
    _x(yInf) = copysign(V::One(), x);

    a += atan(y / _x);

    // if x is +0 and y is +/-0 the result is +0
    a.setZero(xZero && yZero);

    // for x = -0 we add/subtract pi to get the correct result
    a(xMinusZero) += copysign(C::_pi(), y);

    // atan2(-Y, +/-0) = -pi/2
    a(xZero && yNeg) = -C::_pi_2();

    // if both inputs are inf the output is +/- (3)pi/4
    a(xInf && yInf) += copysign(C::_pi_4(), x ^ ~y);

    // correct the sign of y if the result is 0
    a(a == V::Zero()) = copysign(a, y);

    // any NaN input will lead to NaN output
    a.setQnan(isnan(y) || isnan(x));

    return a;
}
template<> template<> Vc::double_v Trigonometric<Vc::Detail::TrigonometricImplementation<Vc::CurrentImplementation::current()>>::atan2 (const Vc::double_v &y, const Vc::double_v &x) {
    typedef Vc::double_v V;
    typedef Const<double, V::abi> C;
    typedef V::Mask M;

    const M xZero = x == V::Zero();
    const M yZero = y == V::Zero();
    const M xMinusZero = xZero && isnegative(x);
    const M yNeg = y < V::Zero();
    const M xInf = !isfinite(x);
    const M yInf = !isfinite(y);

    V a = copysign(V(C::_pi()), y);
    a.setZero(x >= V::Zero());

    // setting x to any finite value will have atan(y/x) return sign(y/x)*pi/2, just in case x is inf
    V _x = x;
    _x(yInf) = copysign(V::One(), x);

    a += atan(y / _x);

    // if x is +0 and y is +/-0 the result is +0
    a.setZero(xZero && yZero);

    // for x = -0 we add/subtract pi to get the correct result
    a(xMinusZero) += copysign(C::_pi(), y);

    // atan2(-Y, +/-0) = -pi/2
    a(xZero && yNeg) = -C::_pi_2();

    // if both inputs are inf the output is +/- (3)pi/4
    a(xInf && yInf) += copysign(C::_pi_4(), x ^ ~y);

    // correct the sign of y if the result is 0
    a(a == V::Zero()) = copysign(a, y);

    // any NaN input will lead to NaN output
    a.setQnan(isnan(y) || isnan(x));

    return a;
}

End quote original Vc code */

// To affect the port, I have copied and pasted infrastructure code from
// https://github.com/google/highway/blob/master/hwy/contrib/math/math-inl.h
// in order to make the code 'slot into' highway. The highway code I have
// copied and pasted from is licensed like this:

// Copyright 2020 Google LLC
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

// Begin code adapted from math-inl.h

#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

/**
 * Highway SIMD version of std::atan2(x).
 *
 * Valid Lane Types: float32, float64
 *        Max Error: ULP = 3
 *      Valid Range: float32[-FLT_MAX, +FLT_MAX], float64[-DBL_MAX, +DBL_MAX]
 * @return atan2 of 'y', 'x'
 */
template <class D, class V>
HWY_INLINE V Atan(const D d, V y, V x);
template <class D, class V>
HWY_NOINLINE V CallAtan2(const D d, VecArg<V> y, VecArg<V> x) {
  return Atan2(d, y, x);
}

template <class D, class V>
HWY_INLINE V Atan2(const D d, V y, V x) {
  using T = TFromD<D>;
  typedef Mask < D > M ;

// End of code adapted from math-inl.h

// Begin port to highway. This derived work was made by
// Kay F. Jahnke. Any aspect of this derived work which is not
// covered by the licenses of either of the the original files
// given above is copyrighted and licensed like this:
//
// Copyright Kay F. Jahnke 2022 - 2023
//
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

  const M xZero = Eq ( x , Set ( d , 0.0 ) ) ;
  const M yZero = Eq ( y , Set ( d , 0.0 ) ) ;
  const M xNeg = Lt ( x , Set ( d , 0.0 ) ) ;
  const M xMinusZero = And ( xZero , xNeg ) ;
  const M yNeg = Lt ( y , Set ( d , 0.0 ) ) ;
  const M xInf = IsInf ( x ) ;
  const M yInf = IsInf ( y ) ;
//   auto qnan = std::numeric_limits<T>::quiet_NaN() ;

  V cpsy = CopySign ( Set ( d , M_PI ) , y ) ;
  V a = IfThenElseZero ( xNeg , cpsy ) ;
  V cpsx = CopySign ( Set ( d , 1.0 ) , x ) ;
  // setting x to any finite value will have atan(y/x) return sign(y/x)*pi/2,
  // just in case x is inf
  V _x = IfThenElse ( yInf , cpsx , x ) ;
  a = Add ( a , Atan ( d , Div ( y , _x ) ) ) ;
  // if x is +0 and y is +/-0 the result is +0
  a = IfThenZeroElse ( And ( xZero , yZero ) , a ) ;
  // for x = -0 we add/subtract pi to get the correct result
  a = IfThenElse ( xMinusZero , Add ( a , cpsy ) , a ) ;
  // atan2(-Y, +/-0) = -pi/2
  a = IfThenElse ( And ( xZero , yNeg ) , Set ( d , - M_PI_2 ) , a ) ;
  V cpsp4 = CopySign ( Set ( d , M_PI_4 ) , Xor ( x , Not ( y ) ) ) ;
  // if both inputs are inf the output is +/- (3)pi/4
  a = IfThenElse ( And ( xInf , yInf ) , Add ( a , cpsp4 ) , a ) ;
  M azero = Eq ( a , Set ( d , 0.0 ) ) ;
  // correct the sign of y if the result is 0
  a = IfThenElse ( azero , CopySign ( a , y ) , a ) ;
  // any NaN input will lead to NaN output
  a = IfThenElse ( Or ( IsNaN ( y ) , IsNaN ( x ) ) , NaN ( d ) , a ) ;
  return a ;
}

}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#define HWY_ATAN2_H
#endif // #ifndef HWY_ATAN2_H
