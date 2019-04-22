/*!
 *  Copyright (c) 2017 by Contributors
 * \file elemwise.cc
 * \brief Elemenwise operators
 */
#include <cvm/op.h>
#include <cvm/node.h>
#include <cvm/op_attr_types.h>
#include <cvm/compiler/op_attr_types.h>
#include <cvm/compiler/util.h>
#include <cvm/top/tensor.h>
#include <cmath>
#include "../op_common.h"
#include "../elemwise_op_common.h"

namespace cvm {
namespace top {

// undefined op
NNVM_REGISTER_ELEMWISE_UNARY_OP(__undef__)
.describe(R"code(undefined op.

Used to produce invalide node during optimization.

)code" NNVM_ADD_FILELINE)
.set_num_outputs(1)
.set_num_inputs(0);

// floor
NNVM_REGISTER_ELEMWISE_UNARY_OP(floor)
.describe(R"code(Take floor input array, computed element-wise.
)code" NNVM_ADD_FILELINE)
.set_support_level(3);

// ceil
NNVM_REGISTER_ELEMWISE_UNARY_OP(ceil)
.describe(R"code(Take ceil input array, computed element-wise.
)code" NNVM_ADD_FILELINE)
.set_support_level(3);

// trunc
NNVM_REGISTER_ELEMWISE_UNARY_OP(trunc)
.describe(R"code(Take truncated value of the input, element-wise.
)code" NNVM_ADD_FILELINE)
.set_support_level(3);

// round
NNVM_REGISTER_ELEMWISE_UNARY_OP(round)
.describe(R"code(Round elements of the input to nearest integer.
)code" NNVM_ADD_FILELINE)
.set_support_level(3);

// abs
NNVM_REGISTER_ELEMWISE_UNARY_OP(abs)
.describe(R"code(Take absolute value of elements of the input.
)code" NNVM_ADD_FILELINE)
.set_support_level(3);

// sigmoid
NNVM_REGISTER_ELEMWISE_UNARY_OP(sigmoid)
.describe(R"code(Computes sigmoid.

.. math::
  Y = 1 / (1 + exp(-X))

)code" NNVM_ADD_FILELINE)
.set_support_level(1);

// tanh
NNVM_REGISTER_ELEMWISE_UNARY_OP(tanh)
.describe(R"code(Computes hyperbolic tangent.

.. math::
   Y = sinh(X) / cosh(X)

)code" NNVM_ADD_FILELINE)
.set_support_level(1);

// exp
NNVM_REGISTER_ELEMWISE_UNARY_OP(exp)
.describe(R"code(Returns the exp input array, computed element-wise.

.. math::
   exp(x)

)code" NNVM_ADD_FILELINE)
.set_support_level(1);

// log2
NNVM_REGISTER_ELEMWISE_UNARY_OP(log2)
.describe(R"code(Returns the log input array, computed element-wise.

.. math::
   log2(x)

)code" NNVM_ADD_FILELINE)
.set_support_level(1);

// log
NNVM_REGISTER_ELEMWISE_UNARY_OP(log)
.describe(R"code(Returns the log input array, computed element-wise.

.. math::
   log(x)

)code" NNVM_ADD_FILELINE)
.set_support_level(1);

// sqrt
NNVM_REGISTER_ELEMWISE_UNARY_OP(sqrt)
.describe(R"code(Returns the sqrt input array, computed element-wise.

.. math::
   \sqrt(x)

)code" NNVM_ADD_FILELINE)
.set_support_level(1);

// binary ops

NNVM_REGISTER_ELEMWISE_BINARY_OP(elemwise_add)
.describe(R"code(Element-wise add

)code")
.set_support_level(1);

NNVM_REGISTER_ELEMWISE_BINARY_OP(elemwise_sub)
.describe(R"code(Element-wise substraction

)code"  NNVM_ADD_FILELINE)
.set_support_level(1);

NNVM_REGISTER_ELEMWISE_BINARY_OP(elemwise_mul)
.describe(R"code(Element-wise multiplication

)code"  NNVM_ADD_FILELINE)
.set_support_level(1);

NNVM_REGISTER_ELEMWISE_BINARY_OP(elemwise_div)
.describe(R"code(Element-wise division

)code"  NNVM_ADD_FILELINE)
.set_support_level(1);

NNVM_REGISTER_ELEMWISE_BINARY_OP(elemwise_mod)
  .describe(R"code(Element-wise modulo

)code" NNVM_ADD_FILELINE)
.set_support_level(1);

NNVM_REGISTER_ELEMWISE_BINARY_OP(elemwise_pow)
  .describe(R"code(Element-wise power

)code" NNVM_ADD_FILELINE)
.set_support_level(1);

// logical
NNVM_REGISTER_ELEMWISE_BINARY_OP(logical_and)
.describe(R"code(Elementwise compute the logical AND

)code")
.set_support_level(4);

NNVM_REGISTER_ELEMWISE_BINARY_OP(logical_or)
.describe(R"code(Elementwise compute the logical OR

)code")
.set_support_level(4);

// negative
NNVM_REGISTER_ELEMWISE_UNARY_OP(negative)
.describe(R"code(Elemenwise numeric negative

)code"  NNVM_ADD_FILELINE)
.set_support_level(3);

// logical NOT
NNVM_REGISTER_ELEMWISE_UNARY_OP(logical_not)
.describe(R"code(Elementwise compute the logical NOT

)code"  NNVM_ADD_FILELINE)
.set_support_level(4);

// copy
NNVM_REGISTER_ELEMWISE_UNARY_OP(copy)
.describe(R"code(Copy tensor to another one.

)code"  NNVM_ADD_FILELINE)
.set_support_level(3);

DMLC_REGISTER_PARAMETER(InitOpParam);
DMLC_REGISTER_PARAMETER(InitOpWithScalarParam);
DMLC_REGISTER_PARAMETER(FillValueParam);

// full
NNVM_REGISTER_INIT_OP(full)
.describe(R"code(Fill array with scalar value

)code"  NNVM_ADD_FILELINE)
.set_attr_parser(ParamParser<InitOpWithScalarParam>)
.set_attr<FGetAttrDict>(
  "FGetAttrDict", ParamGetAttrDict<InitOpWithScalarParam>)
.add_arguments(InitOpWithScalarParam::__FIELDS__())
.set_attr<FInferShape>("FInferShape", ZeroShape<InitOpWithScalarParam>)
.set_attr<FInferType>("FInferType", ZeroType<InitOpWithScalarParam>)
.set_attr<FCorrectLayout>("FCorrectLayout", ZeroLayout)
.set_support_level(4);

NNVM_REGISTER_INIT_OP(zeros)
.describe(R"code(Fill target with zeros

)code"  NNVM_ADD_FILELINE)
.set_attr_parser(ParamParser<InitOpParam>)
.set_attr<FGetAttrDict>(
  "FGetAttrDict", ParamGetAttrDict<InitOpParam>)
.add_arguments(InitOpParam::__FIELDS__())
.set_attr<FInferShape>("FInferShape", ZeroShape<InitOpParam>)
.set_attr<FInferType>("FInferType", ZeroType<InitOpParam>)
.set_attr<FCorrectLayout>("FCorrectLayout", ZeroLayout)
.set_support_level(4);

NNVM_REGISTER_INIT_OP(ones)
.describe(R"code(Fill target with ones

)code"  NNVM_ADD_FILELINE)
.set_attr_parser(ParamParser<InitOpParam>)
.set_attr<FGetAttrDict>(
  "FGetAttrDict", ParamGetAttrDict<InitOpParam>)
.add_arguments(InitOpParam::__FIELDS__())
.set_attr<FInferShape>("FInferShape", ZeroShape<InitOpParam>)
.set_attr<FInferType>("FInferType", ZeroType<InitOpParam>)
.set_attr<FCorrectLayout>("FCorrectLayout", ZeroLayout)
.set_support_level(4);

// full_like
NNVM_REGISTER_INIT_LIKE_OP(full_like)
.describe(R"code(Return an scalar value array with the same shape and type
as the input array

)code"  NNVM_ADD_FILELINE)
.add_arguments(FillValueParam::__FIELDS__())
.set_attr_parser(ParamParser<FillValueParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<FillValueParam>)
.set_support_level(4);

NNVM_REGISTER_INIT_LIKE_OP(zeros_like)
.describe(R"code(Return an array of zeros with the same shape and type
as the input array.

)code")
.set_support_level(4);

NNVM_REGISTER_INIT_LIKE_OP(ones_like)
.describe(R"code(Return an array of ones with the same shape and type
as the input array.

)code")
.set_support_level(4);

// unary scalar op
DMLC_REGISTER_PARAMETER(ScalarParam);

#define NNVM_REGISTER_ELEMWISE_BINARY_SCALAR(op)                        \
  NNVM_REGISTER_ELEMWISE_UNARY_OP(op)                                   \
  .add_arguments(ScalarParam::__FIELDS__())                             \
  .set_attr_parser(ParamParser<ScalarParam>)                            \
  .set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<ScalarParam>)

NNVM_REGISTER_ELEMWISE_BINARY_SCALAR(__add_scalar__)
.describe(R"code(Tensor add scalar

)code"  NNVM_ADD_FILELINE)
.set_support_level(3);

NNVM_REGISTER_ELEMWISE_BINARY_SCALAR(__sub_scalar__)
.describe(R"code(Tensor substract scalar

)code"  NNVM_ADD_FILELINE)
.set_support_level(3);

NNVM_REGISTER_ELEMWISE_BINARY_SCALAR(__rsub_scalar__)
.describe(R"code(scalar substract Tensor

)code"  NNVM_ADD_FILELINE)
.set_support_level(3);

NNVM_REGISTER_ELEMWISE_BINARY_SCALAR(__lshift_scalar__)
.describe(R"code(Tensor left shift by scalar

)code"  NNVM_ADD_FILELINE)
.set_support_level(3);

NNVM_REGISTER_ELEMWISE_BINARY_SCALAR(__rshift_scalar__)
.describe(R"code(Tensor right shift by scalar

)code"  NNVM_ADD_FILELINE)
.set_support_level(3);

NNVM_REGISTER_ELEMWISE_BINARY_SCALAR(__mul_scalar__)
.describe(R"code(Tensor multiplies scalar

)code"  NNVM_ADD_FILELINE)
.set_support_level(3);

NNVM_REGISTER_ELEMWISE_BINARY_SCALAR(__div_scalar__)
.describe(R"code(Tensor divides scalar

)code"  NNVM_ADD_FILELINE)
.set_support_level(3);

NNVM_REGISTER_ELEMWISE_BINARY_SCALAR(__rdiv_scalar__)
.describe(R"code(scalar divides Tensor

)code"  NNVM_ADD_FILELINE)
.set_support_level(3);

NNVM_REGISTER_ELEMWISE_BINARY_SCALAR(__pow_scalar__)
.describe(R"code(Tensor power scalar

)code"  NNVM_ADD_FILELINE)
.set_support_level(3);

NNVM_REGISTER_ELEMWISE_BINARY_SCALAR(__rpow_scalar__)
.describe(R"code(scalar power Tensor

)code"  NNVM_ADD_FILELINE)
.set_support_level(3);

DMLC_REGISTER_PARAMETER(ElementWiseReduceParam);

NNVM_REGISTER_ELEMWISE_REDUCE_OP(elemwise_sum)
.describe(R"code(Adds all input arguments element-wise.

)code"  NNVM_ADD_FILELINE)
.set_support_level(4);

NNVM_REGISTER_ELEMWISE_UNARY_OP(block_grad)
.describe(R"code(Blocks gradient computation for input.

)code" NNVM_ADD_FILELINE)
.set_attr<cvm::FInplaceIdentity>(
  "FInplaceIdentity", [](const NodeAttrs& attrs){
    return std::vector<bool>{true};
})
.set_support_level(4);

DMLC_REGISTER_PARAMETER(IndicatorParam);

// indicator function
NNVM_REGISTER_INDICATOR_OP(greater)
.describe(R"code(Greater function that returns a mask tensor
with 1.0 if (left > right), otherwise 0.0 element-wise.

)code" NNVM_ADD_FILELINE)
.add_argument("lhs", "Tensor", "First input")
.add_argument("rhs", "Tensor", "Second input")
.set_num_inputs(2)
.set_attr<cvm::FInferShape>("FInferShape", ElemwiseShape<2, 1>)
.set_support_level(4);


NNVM_REGISTER_INDICATOR_OP(less)
  .describe(R"code(Less function that returns a mask tensor
with 1.0 if (left < right), otherwise 0.0 element-wise.

)code" NNVM_ADD_FILELINE)
.add_argument("lhs", "Tensor", "First input")
.add_argument("rhs", "Tensor", "Second input")
.set_num_inputs(2)
.set_attr<cvm::FInferShape>("FInferShape", ElemwiseShape<2, 1>)
.set_support_level(4);

NNVM_REGISTER_INDICATOR_OP(_max_mask)
  .describe(R"code(Function that returns a mask tensor
with 1.0 if the value is maximum over given axes, otherwise 0.0 element-wise.

)code" NNVM_ADD_FILELINE)
.add_argument("data", "Tensor", "Input")
.set_num_inputs(1)
.add_arguments(IndicatorParam::__FIELDS__())
.set_attr_parser(ParamParser<IndicatorParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<IndicatorParam>)
.set_attr<cvm::FInferShape>("FInferShape", ElemwiseShape<1, 1>)
.set_support_level(1);

NNVM_REGISTER_INDICATOR_OP(_min_mask)
  .describe(R"code(Function that returns a mask tensor
with 1.0 if the value is minimum over given axes, otherwise 0.0 element-wise.

)code" NNVM_ADD_FILELINE)
.add_argument("data", "Tensor", "Input")
.set_num_inputs(1)
.add_arguments(IndicatorParam::__FIELDS__())
.set_attr_parser(ParamParser<IndicatorParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<IndicatorParam>)
.set_attr<cvm::FInferShape>("FInferShape", ElemwiseShape<1, 1>)
.set_support_level(1);


DMLC_REGISTER_PARAMETER(ClipParam);

NNVM_REGISTER_OP(clip)
.describe(R"doc(Clips (limits) the values in an array.
Given an interval, values outside the interval are clipped to the interval edges.
Clipping ``x`` between `a_min` and `a_x` would be::
   clip(x, a_min, a_max) = max(min(x, a_max), a_min))
Example::
    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    clip(x,1,8) = [ 1.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  8.]
)doc" NNVM_ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<ClipParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<ClipParam>)
.set_attr<cvm::FInferShape>("FInferShape", ElemwiseShape<1, 1>)
.set_attr<cvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<cvm::FCorrectLayout>("FCorrectLayout", ElemwiseFixedLayoutUnknownOut<1, 1>)
.add_argument("data", "NDArray-or-Symbol", "Input array.")
.add_arguments(ClipParam::__FIELDS__())
.set_support_level(4);

}  // namespace top
}  // namespace cvm
