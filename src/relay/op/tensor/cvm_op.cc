#include <tvm/relay/expr.h>
#include <tvm/relay/op.h>
#include <tvm/relay/attrs/cvm_op.h>
#include "topi/broadcast.h"
#include <topi/elemwise.h>
#include <topi/transform.h>
#include "../type_relations.h"
#include "../op_common.h"

namespace tvm {
namespace relay {
#define INT_PREC 32

TVM_REGISTER_NODE_TYPE(CVMLUTAttrs);

bool CVMLUTReal(const Array<Type>& types,
                int num_inputs,
                const Attrs& attrs,
                const TypeReporter& reporter) {
  // `types` contains: [data, table, result]
  CHECK_EQ(types.size(), 3);
  const auto* data = types[0].as<TensorTypeNode>();
  CHECK(data != nullptr);
  const auto* table = types[1].as<TensorTypeNode>();
  CHECK(table != nullptr);
  const auto param = attrs.as<CVMLUTAttrs>();
  CHECK(param != nullptr);

  std::vector<IndexExpr>&& oshape = AsVector(data->shape);
  reporter->Assign(types[2], TensorTypeNode::make(oshape, table->dtype));
  return true;
}

TVM_REGISTER_API("relay.op._make.cvm_lut")
.set_body_typed<Expr(Expr, Expr, int)>([](Expr data, Expr table, int in_dim) {
  auto attrs = make_node<CVMLUTAttrs>();
  attrs->in_dim = in_dim;
  static const Op& op = Op::Get("cvm_lut");

  return CallNode::make(op, {data, table}, Attrs(attrs), {});
});

RELAY_REGISTER_OP("cvm_lut")
.describe(R"doc(CVMLUT look up input with table.
)doc" TVM_ADD_FILELINE)
.set_num_inputs(2)
.add_argument("data", "Tensor", "input")
.add_argument("table", "Tensor", "The table to lookup")
.add_type_rel("CVMLUT", CVMLUTReal)
.set_attr<TOpPattern>("TOpPattern", kInjective)
.set_support_level(4)
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const Attrs& attrs,
                    const Array<Tensor>& inputs,
                    const Type& out_type,
                    const Target& target) {
    return Array<Tensor>{
      topi::take(inputs[1],
          topi::cast(inputs[0], tvm::Int(INT_PREC)))
    };
});

TVM_REGISTER_NODE_TYPE(CVMClipAttrs);

TVM_REGISTER_API("relay.op._make.cvm_clip")
.set_body_typed<Expr(Expr, int)>([](Expr a, int precision) {
    auto attrs = make_node<CVMClipAttrs>();
    attrs->precision = precision;
    static const Op& op = Op::Get("cvm_clip");
  return CallNode::make(op, {a}, Attrs(attrs), {});
});

RELAY_REGISTER_OP("cvm_clip")
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
)doc" TVM_ADD_FILELINE)
.set_num_inputs(1)
.add_argument("data", "Tensor", "The input tensor.")
.add_type_rel("CVMClip", IdentityRel)
.set_attr<TOpPattern>("TOpPattern", kElemWise)
.set_attr<TOpIsStateful>("TOpIsStateful", false)
.set_attr<FInferCorrectLayout>("FInferCorrectLayout", ElemwiseArbitraryLayout)
.set_attrs_type_key("relay.attrs.CVMClipAttrs")
.set_support_level(3)
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const Attrs& attrs,
                    const Array<Tensor>& inputs,
                    const Type& out_type,
                    const Target& target) {
    const CVMClipAttrs *param = attrs.as<CVMClipAttrs>();
    int a_max = (1 << (param->precision - 1)) - 1;
    int a_min = -a_max;
    return Array<Tensor>{
      topi::clip(inputs[0], tvm::make_const(tvm::Int(INT_PREC), a_min),
                 tvm::make_const(tvm::Int(INT_PREC), a_max))
    };
});

TVM_REGISTER_NODE_TYPE(CVMLeftShiftAttrs);

TVM_REGISTER_API("relay.op._make.cvm_left_shift")
.set_body_typed<Expr(Expr, int, int)>([](Expr a, int precision, int shift_bit) {
    auto attrs = make_node<CVMLeftShiftAttrs>();
    attrs->precision = precision;
    attrs->shift_bit = shift_bit;
    static const Op& op = Op::Get("cvm_left_shift");
  return CallNode::make(op, {a}, Attrs(attrs), {});
});

RELAY_REGISTER_OP("cvm_left_shift")
.describe(R"code(CVM left shift with precision-aware clip.

.. math::
	assert shift_bit > 0
	tmp = X << shift_bit
	Y = cvm_clip(tmp, precision)
)code" TVM_ADD_FILELINE)
.set_num_inputs(1)
.add_argument("data", "Tensor", "The input tensor.")
.add_type_rel("CVMLeftShift", IdentityRel)
.set_attr<TOpPattern>("TOpPattern", kElemWise)
.set_attr<TOpIsStateful>("TOpIsStateful", false)
.set_attr<FInferCorrectLayout>("FInferCorrectLayout", ElemwiseArbitraryLayout)
.set_attrs_type_key("relay.attrs.CVMLeftShiftAttrs")
.set_support_level(3)
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const Attrs& attrs,
                    const Array<Tensor>& inputs,
                    const Type& out_type,
                    const Target& target) {
    const CVMLeftShiftAttrs *param = attrs.as<CVMLeftShiftAttrs>();
    int a_max = (1 << (param->precision - 1)) - 1;
    int a_min = -a_max;
    const Tensor& tmp = topi::left_shift(inputs[0],
        tvm::make_const(tvm::Int(INT_PREC), param->shift_bit));
    return Array<Tensor>{
      topi::clip(tmp, tvm::make_const(tvm::Int(INT_PREC), a_min),
                 tvm::make_const(tvm::Int(INT_PREC), a_max))
    };
});

TVM_REGISTER_NODE_TYPE(CVMRightShiftAttrs);

TVM_REGISTER_API("relay.op._make.cvm_right_shift")
.set_body_typed<Expr(Expr, int, int)>([](Expr a, int precision, int shift_bit) {
    auto attrs = make_node<CVMRightShiftAttrs>();
    attrs->precision = precision;
    attrs->shift_bit = shift_bit;
    static const Op& op = Op::Get("cvm_right_shift");
  return CallNode::make(op, {a}, Attrs(attrs), {});
});

RELAY_REGISTER_OP("cvm_right_shift")
.describe(R"code(CVM right shift with precision-aware clip.

The right shift is equal to float number round divide operator,
which means to implement via tricky equation.

.. math::
	assert shift_bit > 0
	tmp = X >> (shift_bit - 1)
	tmp = tmp + 1
	tmp = tmp >> 1
	Y = cvm_clip(tmp, precision)
)code" TVM_ADD_FILELINE)
.set_num_inputs(1)
.add_argument("data", "Tensor", "The input tensor.")
.add_type_rel("CVMRightShift", IdentityRel)
.set_attr<TOpPattern>("TOpPattern", kElemWise)
.set_attr<TOpIsStateful>("TOpIsStateful", false)
.set_attr<FInferCorrectLayout>("FInferCorrectLayout", ElemwiseArbitraryLayout)
.set_attrs_type_key("relay.attrs.CVMRightShiftAttrs")
.set_support_level(3)
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const Attrs& attrs,
                    const Array<Tensor>& inputs,
                    const Type& out_type,
                    const Target& target) {
    const CVMRightShiftAttrs *param = attrs.as<CVMRightShiftAttrs>();
    int a_max = (1 << (param->precision - 1)) - 1;
    int a_min = -a_max;
    const Tensor& tmp = topi::right_shift(inputs[0],
        tvm::make_const(tvm::Int(INT_PREC), param->shift_bit));
    return Array<Tensor>{
      topi::clip(tmp, tvm::make_const(tvm::Int(INT_PREC), a_min),
                 tvm::make_const(tvm::Int(INT_PREC), a_max))
    };
});

}  // namespace relay
}  // namespace tvm
