/*!
 *  Copyright (c) 2018 by Contributors
 * \file state_op.cc
 * \brief Experimental operators
 *   Currently we only support assign
 */
#include <nnvm/op.h>
#include <nnvm/node.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/compiler/op_attr_types.h>
#include <nnvm/top/cvm_op.h>
#include <topi/elemwise.h>
#include "../op_common.h"
#include "../elemwise_op_common.h"
#include "topi/broadcast.h"

namespace nnvm {
namespace top {

using namespace tvm;
using namespace nnvm::compiler;

DMLC_REGISTER_PARAMETER(CVMClipParam);

NNVM_REGISTER_OP(cvm_clip)
.describe(R"doc(CVM clip input with precision.

.. math::
	range = 2 ** (precision - (is_sign ? 1 : 0)) - 1
	a_min = is_sign ? -range : 0
	a_max = range
	Y = clip(X, a_min=a_min, a_max=a_max)

Example::

	data = [275, 157, -23, -168, -275]

	cvm_clip(data, precision=8, is_sign=True)
	[127, 127, -23, -127, -127]

	cvm_clip(data, precision=8, is_sign=False)
	[255, 157, 0, 0, 0]
)doc" NNVM_ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<CVMClipParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<CVMClipParam>)
.set_attr<nnvm::FInferShape>("FInferShape", ElemwiseShape<1, 1>)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<nnvm::FCorrectLayout>("FCorrectLayout", ElemwiseFixedLayoutUnknownOut<1, 1>)
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const NodeAttrs& attrs,
                    const Array<Tensor>& inputs,
                    const Array<Tensor>& out_info) {
    const CVMClipParam params = get<CVMClipParam>(attrs.parsed);
		int64_t a_max, a_min;
		if (params.is_sign) {
			a_max = (1 << (params.precision-1)) - 1;
			a_min = -a_max;
		} else {
			a_max = (1 << params.precision) - 1;
			a_min = 0;
		}
    return Array<Tensor>{
      topi::clip(inputs[0], tvm::make_const(tvm::Int(64), a_min),
                 tvm::make_const(tvm::Int(64), a_max)) };
  }, 11)
.add_argument("data", "Tensor", "input")
.add_arguments(CVMClipParam::__FIELDS__())
.set_support_level(4);

DMLC_REGISTER_PARAMETER(CVMLeftShiftParam);

NNVM_REGISTER_OP(cvm_left_shift)
.describe(R"code(CVM left shift with precision-aware clip.

.. math::
	assert shift_bit > 0
	tmp = X << shift_bit
	Y = cvm_clip(tmp, precision)
)code" NNVM_ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<CVMLeftShiftParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<CVMLeftShiftParam>)
.set_attr<nnvm::FInferShape>("FInferShape", ElemwiseShape<1, 1>)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<nnvm::FCorrectLayout>("FCorrectLayout", ElemwiseFixedLayoutUnknownOut<1, 1>)
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const NodeAttrs& attrs,
                    const Array<Tensor>& inputs,
                    const Array<Tensor>& out_info) {
    const CVMLeftShiftParam params = get<CVMLeftShiftParam>(attrs.parsed);
		int64_t a_max, a_min;
		if (params.is_sign) {
			a_max = (1 << (params.precision-1)) - 1;
			a_min = -a_max;
		} else {
			a_max = (1 << params.precision) - 1;
			a_min = 0;
		}
		const Tensor& tmp = topi::left_shift(inputs[0],
				tvm::make_const(tvm::Int(64), params.shift_bit));
    return Array<Tensor>{
      topi::clip(inputs[0], tvm::make_const(tvm::Int(64), a_min),
                 tvm::make_const(tvm::Int(64), a_max)) };
  }, 11)
.add_argument("data", "Tensor", "input")
.add_arguments(CVMLeftShiftParam::__FIELDS__())
.set_support_level(4);

DMLC_REGISTER_PARAMETER(CVMRightShiftParam);

NNVM_REGISTER_OP(cvm_right_shift)
.describe(R"code(CVM right shift with precision-aware clip.

The right shift is equal to float number round divide operator,
which means to implement via tricky equation.

.. math::
	assert shift_bit > 0
	tmp = X >> (shift_bit - 1)
	tmp = tmp + 1
	tmp = tmp >> 1
	Y = cvm_clip(tmp, precision)
)code" NNVM_ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<CVMRightShiftParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<CVMRightShiftParam>)
.set_attr<nnvm::FInferShape>("FInferShape", ElemwiseShape<1, 1>)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<nnvm::FCorrectLayout>("FCorrectLayout", ElemwiseFixedLayoutUnknownOut<1, 1>)
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const NodeAttrs& attrs,
                    const Array<Tensor>& inputs,
                    const Array<Tensor>& out_info) {
    const CVMRightShiftParam params = get<CVMRightShiftParam>(attrs.parsed);
		int64_t a_max, a_min;
		if (params.is_sign) {
			a_max = (1 << (params.precision-1)) - 1;
			a_min = -a_max;
		} else {
			a_max = (1 << params.precision) - 1;
			a_min = 0;
		}
		const Tensor& tmp = topi::right_shift(inputs[0],
				tvm::make_const(tvm::Int(64), params.shift_bit));
    return Array<Tensor>{
      topi::clip(tmp, tvm::make_const(tvm::Int(64), a_min),
                 tvm::make_const(tvm::Int(64), a_max)) };
  }, 11)
.add_argument("data", "Tensor", "input")
.add_arguments(CVMRightShiftParam::__FIELDS__())
.set_support_level(4);

}  // namespace top
}  // namespace nnvm
