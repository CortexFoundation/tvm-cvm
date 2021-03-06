/*!
 *  Copyright (c) 2017 by Contributors
 * \file nn.cc
 * \brief Property def of nn operators.
 */
#include <tvm/tvm.h>
#include <tvm/expr.h>
#include <tvm/packed_func_ext.h>
#include <cvm/op.h>
#include <cvm/node.h>
#include <cvm/layout.h>
#include <cvm/op_attr_types.h>
#include <cvm/compiler/op_attr_types.h>
#include <cvm/top/nn.h>
#include "nn_common.h"
#include "../op_common.h"
#include "../elemwise_op_common.h"

namespace cvm {
namespace top {

using tvm::Var;
using tvm::Expr;
using tvm::Tensor;
using tvm::Array;

// dense
DMLC_REGISTER_PARAMETER(DenseParam);

inline bool DenseInferShape(const cvm::NodeAttrs& attrs,
                            std::vector<TShape>* in_shape,
                            std::vector<TShape>* out_shape) {
  const DenseParam& param = cvm::get<DenseParam>(attrs.parsed);
  if (param.use_bias) {
    CHECK_EQ(in_shape->size(), 3U) << "Input:[data, weight, bias]";
  } else {
    CHECK_EQ(in_shape->size(), 2U) << "Input:[data, weight]";
  }
  CHECK_EQ(out_shape->size(), 1U);
  // reverse infer
  if ((*out_shape)[0].ndim() != 0) {
    TShape dshape = (*out_shape)[0];
    dshape[dshape.ndim() - 1] = 0;
    NNVM_ASSIGN_INPUT_SHAPE(attrs, *in_shape, DenseParam::kData, dshape);
  }
  dim_t num_inputs = 0;
  if ((*in_shape)[DenseParam::kData].ndim() != 0) {
    TShape oshape = (*in_shape)[DenseParam::kData];
    num_inputs = oshape[oshape.ndim() - 1];
    oshape[oshape.ndim() - 1] = param.units;
    NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_shape, 0, oshape);
  }
  NNVM_ASSIGN_INPUT_SHAPE(attrs, *in_shape, DenseParam::kWeight,
                          TShape({param.units, num_inputs}));
  if (param.use_bias) {
    NNVM_ASSIGN_INPUT_SHAPE(attrs, *in_shape, DenseParam::kBias, TShape({param.units}));
  }
  return true;
}

NNVM_REGISTER_OP(dense)
.describe(R"code(Applies a linear transformation: :math:`Y = XW^T + b`.

- **data**: `(x1, x2, ..., xn, input_dim)`
- **weight**: `(units, input_dim)`
- **bias**: `(units,)`
- **out**: `(x1, x2, ..., xn, units)`

The learnable parameters include both ``weight`` and ``bias``.

If ``use_bias`` is set to be false, then the ``bias`` term is ignored.

)code" NNVM_ADD_FILELINE)
.add_argument("data", "nD Tensor", "Input data.")
.add_argument("weight", "2D Tensor", "Weight matrix.")
.add_argument("bias", "1D Tensor", "Bias parameter.")
.add_arguments(DenseParam::__FIELDS__())
.set_attr_parser(ParamParser<DenseParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<DenseParam>)
.set_num_outputs(1)
.set_num_inputs(UseBiasNumInputs<DenseParam>)
.set_attr<FListInputNames>("FListInputNames", UseBiasListInputNames<DenseParam>)
.set_attr<FInferShape>("FInferShape", DenseInferShape)
.set_attr<FInferType>("FInferType", ElemwiseType<-1, 1>)
// leave weight & bias layout undefined
.set_attr<FCorrectLayout>("FCorrectLayout", ElemwiseFixedLayoutCopyToOut<1, 1>)
.set_support_level(1);

// relu
NNVM_REGISTER_ELEMWISE_UNARY_OP(relu)
.describe(R"code(Computes rectified linear.

.. math::
   max(input, 0)

)code" NNVM_ADD_FILELINE)
.set_support_level(1);

/*
// dropout
DMLC_REGISTER_PARAMETER(DropoutParam);

NNVM_REGISTER_OP(dropout)
.describe(R"(Applies dropout operation to input array.

- During training, each element of the input is set to zero with probability p.
  The whole array is rescaled by :math:`1/(1-p)` to keep the expected
  sum of the input unchanged.

)" NNVM_ADD_FILELINE)
.add_argument("data", "Tensor", "Input to which dropout will be applied")
.add_arguments(DropoutParam::__FIELDS__())
.set_attr_parser(ParamParser<DropoutParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<DropoutParam>)
.set_num_inputs(1)
.set_num_outputs(2)
.set_attr<FInferShape>("FInferShape", ElemwiseShape<1, 2>)
.set_attr<FInferType>("FInferType", ElemwiseType<1, 2>)
.set_attr<FCorrectLayout>("FCorrectLayout", ElemwiseArbitraryLayout<1, 1>)
.set_attr<FNumVisibleOutputs>("FNumVisibleOutputs", [](const NodeAttrs& attrs) {
    return 1;
  })
.set_attr<FListOutputNames>("FListOutputNames", [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"output", "mask"};
  })
.set_support_level(1);
*/

/*
// batchnorm
DMLC_REGISTER_PARAMETER(BatchNormParam);

inline bool BatchNormInferShape(const cvm::NodeAttrs& attrs,
                                std::vector<TShape>* in_shape,
                                std::vector<TShape>* out_shape) {
  const BatchNormParam& param = cvm::get<BatchNormParam>(attrs.parsed);
  CHECK_EQ(in_shape->size(), 5U)
      << "Input:[data, gamma, beta, moving_mean, moving_var]";
  CHECK_EQ(out_shape->size(), 3U);
  const TShape &dshape = in_shape->at(0);
  if (dshape.ndim() == 0) return false;
  CHECK((size_t)param.axis < dshape.Size());

  TShape bshape({dshape[param.axis]});
  if (in_shape->at(1).ndim() == 0) in_shape->at(1) = bshape;
  if (in_shape->at(2).ndim() == 0) in_shape->at(2) = bshape;
  if (in_shape->at(3).ndim() == 0) in_shape->at(3) = bshape;
  if (in_shape->at(4).ndim() == 0) in_shape->at(4) = bshape;
  NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_shape, 0, dshape);
  out_shape->at(1) = in_shape->at(3);
  out_shape->at(2) = in_shape->at(4);
  return true;
}

inline bool BatchNormCorrectLayout(const NodeAttrs& attrs,
                                   std::vector<Layout> *in_layouts,
                                   const std::vector<Layout> *last_in_layouts,
                                   std::vector<Layout> *out_layouts) {
  const BatchNormParam& param = cvm::get<BatchNormParam>(attrs.parsed);
  CHECK_EQ(in_layouts->size(), 5U);
  CHECK_EQ(last_in_layouts->size(), 5U);
  CHECK_EQ(out_layouts->size(), 3U);

  Layout data_layout = in_layouts->at(0);
  const Layout& origin_data_layout = last_in_layouts->at(0);
  Layout param_layout("C");
  if (data_layout.defined()) {
    if (data_layout.indexof('C') != param.axis) {
      CHECK(origin_data_layout.defined())
        << "Channel in data layout " << data_layout
        << " is not at index " << param.axis;
      // convert it to the original one.
      data_layout = origin_data_layout;
      NNVM_ASSIGN_LAYOUT(*in_layouts, 0, origin_data_layout);
    } else if (data_layout.indexof('c') >= 0 &&
               static_cast<uint32_t>(data_layout.indexof('c')) != (data_layout.ndim()-1)) {
      CHECK(origin_data_layout.defined())
        << "sub-channel c in data layout " << data_layout
        << " does not at the final dimension";
      // convert it to the original one.
      data_layout = origin_data_layout;
      NNVM_ASSIGN_LAYOUT(*in_layouts, 0, origin_data_layout);
    } else {
      for (Layout::LayoutDim axis : data_layout) {
        if (Layout::is_subdim(axis) && axis != 'c') {
          CHECK(origin_data_layout.defined())
            << "sub-axis other than c appears in data layout " << data_layout;
          // convert it to the original one.
          data_layout = origin_data_layout;
          NNVM_ASSIGN_LAYOUT(*in_layouts, 0, origin_data_layout);
          break;
        }
      }
    }

    // decide the param layout
    if (data_layout.defined()) {
      auto channel_block = data_layout.subsizeof('C');
      if (channel_block > 0) {
        param_layout = param_layout.split('C', 1, channel_block);
      }
    }
  }

  NNVM_ASSIGN_LAYOUT(*in_layouts, 0, data_layout);
  NNVM_ASSIGN_LAYOUT(*in_layouts, 1, param_layout);
  NNVM_ASSIGN_LAYOUT(*in_layouts, 2, param_layout);
  NNVM_ASSIGN_LAYOUT(*in_layouts, 3, param_layout);
  NNVM_ASSIGN_LAYOUT(*in_layouts, 4, param_layout);

  NNVM_ASSIGN_LAYOUT(*out_layouts, 0, data_layout);
  NNVM_ASSIGN_LAYOUT(*out_layouts, 1, param_layout);
  NNVM_ASSIGN_LAYOUT(*out_layouts, 2, param_layout);
  return true;
}
*/
// softmax
/*
DMLC_REGISTER_PARAMETER(SoftmaxParam);

NNVM_REGISTER_OP(softmax)
.describe(R"code(Computes softmax.

.. math:: \text{softmax}(x)_i = \frac{exp(x_i)}{\sum_j exp(x_j)}

.. note::
    This operator can be optimized away for inference.
)code" NNVM_ADD_FILELINE)
.add_argument("data", "Tensor", "Input data.")
.add_arguments(SoftmaxParam::__FIELDS__())
.set_attr_parser(ParamParser<SoftmaxParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<SoftmaxParam>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<FInferShape>("FInferShape", ElemwiseShape<1, 1>)
.set_attr<FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<FCorrectLayout>("FCorrectLayout", ElemwiseFixedLayoutCopyToOut<1, 1>)
.set_support_level(1);
*/
/*
// log_softmax
NNVM_REGISTER_OP(log_softmax)
.describe(R"code(Computes log softmax.

.. math:: \text{log_softmax}(x)_i = \log \frac{exp(x_i)}{\sum_j exp(x_j)}

.. note::
    This operator can be optimized away for inference.
)code" NNVM_ADD_FILELINE)
.add_argument("data", "Tensor", "Input data.")
.add_arguments(SoftmaxParam::__FIELDS__())
.set_attr_parser(ParamParser<SoftmaxParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<SoftmaxParam>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<FInferShape>("FInferShape", ElemwiseShape<1, 1>)
.set_attr<FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<FCorrectLayout>("FCorrectLayout", ElemwiseFixedLayoutCopyToOut<1, 1>)
.set_support_level(1);

// leaky_relu
DMLC_REGISTER_PARAMETER(LeakyReLUParam);

NNVM_REGISTER_OP(leaky_relu)
.describe(R"code(Leaky version of a Rectified Linear Unit.

`y = x > 0 ? x : alpha * x`

)code" NNVM_ADD_FILELINE)
.add_argument("data", "Tensor", "Input data.")
.add_arguments(LeakyReLUParam::__FIELDS__())
.set_attr_parser(ParamParser<LeakyReLUParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<LeakyReLUParam>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<FInferShape>("FInferShape", ElemwiseShape<1, 1>)
.set_attr<FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<FCorrectLayout>("FCorrectLayout", ElemwiseArbitraryLayout<1, 1>)
.set_support_level(1);

// prelu
DMLC_REGISTER_PARAMETER(PReLUParam);

inline bool PReluInferShape(const cvm::NodeAttrs &attrs,
                            std::vector<TShape> *in_shape,
                            std::vector<TShape> *out_shape) {
  const PReLUParam &param = cvm::get<PReLUParam>(attrs.parsed);
  TShape dshape = in_shape->at(0);
  NNVM_ASSIGN_INPUT_SHAPE(attrs, *in_shape, 0, dshape);

  // The case of parametric relu
  CHECK_EQ(dshape.ndim(), 4) << "Input data should be 4D, but got " << dshape.ndim();
  CHECK(size_t(param.axis) < dshape.Size())
      << "Wrong axis ("  << param.axis << ")value.";

  NNVM_ASSIGN_INPUT_SHAPE(attrs, *in_shape, 1, TShape({dshape[param.axis]}));

  TShape oshape(dshape);
  NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_shape, 0, oshape);
  return true;
}

inline bool PReluCorrectLayout(const NodeAttrs& attrs,
                               std::vector<Layout> *in_layouts,
                               const std::vector<Layout> *last_in_layouts,
                               std::vector<Layout> *out_layouts) {
  const PReLUParam& param = cvm::get<PReLUParam>(attrs.parsed);
  CHECK_EQ(in_layouts->size(), 2U);
  CHECK_EQ(last_in_layouts->size(), 2U);
  CHECK_EQ(out_layouts->size(), 1U);

  const Layout& data_layout = last_in_layouts->at(0).defined() ?
                              last_in_layouts->at(0) : in_layouts->at(0);
  if (data_layout.defined()) {
    CHECK(data_layout.indexof('C') == param.axis && !data_layout.contains('c'))
      << "Channel in data layout " << data_layout
      << " is not at index " << param.axis;
  }

  NNVM_ASSIGN_LAYOUT(*in_layouts, 0, data_layout);
  NNVM_ASSIGN_LAYOUT(*in_layouts, 1, Layout("C"));
  NNVM_ASSIGN_LAYOUT(*out_layouts, 0, data_layout);

  return true;
}

NNVM_REGISTER_OP(prelu)
.describe(R"code(Parametric version of a Rectified Linear Unit.
It accepts two arguments: an input ``x`` and a channelwise slope ``alpha``
and computes the output as :math:`PReLU(x) y = x > 0 ? x : alpha * x`,
where :math:`*` is an channelwise multiplication for each sample in the

)code" NNVM_ADD_FILELINE)
.add_argument("data", "Tensor", "Input data.")
.add_argument("alpha", "Tensor", "Input channelwise alpha.")
.add_arguments(PReLUParam::__FIELDS__())
.set_attr_parser(ParamParser<PReLUParam>)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<FInferShape>("FInferShape", PReluInferShape)
.set_attr<FCorrectLayout>("FCorrectLayout", PReluCorrectLayout)
.set_attr<FListInputNames>("FListInputNames", [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data", "alpha"};
  });

DMLC_REGISTER_PARAMETER(PadParam);

inline bool PadInferShape(const cvm::NodeAttrs& attrs,
                          std::vector<TShape>* in_shape,
                          std::vector<TShape>* out_shape) {
  const PadParam& param = cvm::get<PadParam>(attrs.parsed);
  CHECK_EQ(in_shape->size(), 1U);
  CHECK_EQ(out_shape->size(), 1U);
  TShape dshape = (*in_shape)[0];
  if (dshape.ndim() == 0) return false;
  CHECK_EQ(param.pad_width.ndim(), dshape.ndim());
  TShape oshape = dshape;
  for (uint32_t i = 0; i < dshape.ndim(); i++) {
    CHECK_EQ(param.pad_width[i].ndim(), 2U);
    int pad_before = param.pad_width[i][0];
    int pad_after = param.pad_width[i][1];
    oshape[i] = dshape[i] + pad_before + pad_after;
  }
  NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_shape, 0, oshape);
  return true;
}

NNVM_REGISTER_OP(pad)
.describe(R"code(Pad for n-D tensor.

)code" NNVM_ADD_FILELINE)
.add_argument("data", "n-D Tensor", "Input data.")
.add_arguments(PadParam::__FIELDS__())
.set_attr_parser(ParamParser<PadParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<PadParam>)
.set_num_outputs(1)
.set_num_inputs(1)
.set_attr<FInferShape>("FInferShape", PadInferShape)
.set_attr<FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<FCorrectLayout>("FCorrectLayout", ElemwiseFixedLayoutCopyToOut<1, 1>)
.set_support_level(1);

// layout transformer
DMLC_REGISTER_PARAMETER(LayoutTransformParam);

inline bool LayoutTransformInferShape(const NodeAttrs& attrs,
                                      std::vector<TShape>* in_attrs,
                                      std::vector<TShape>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U) << "Input: [data]";
  CHECK_EQ(out_attrs->size(), 1U);
  const LayoutTransformParam& param = cvm::get<LayoutTransformParam>(attrs.parsed);
  const TShape &dshape = (*in_attrs)[0];
  if (dshape.ndim() == 0) return false;
  const TShape &oshape = ConvertLayout(dshape,
                                       Layout(param.src_layout),
                                       Layout(param.dst_layout));
  NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_attrs, 0, oshape);
  return true;
}

NNVM_REGISTER_OP(__layout_transform__)
.describe(R"code(Transform the input data layout.

For transforming from NCHW to N16cHWC, the `__layout_transform__` operator reshapes
the input array by output[n, c, h, w, C] = data[n, C*16+c, h, w]

)code" NNVM_ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(1)
.add_argument("data", "Tensor", "Input data.")
.add_arguments(LayoutTransformParam::__FIELDS__())
.set_attr_parser(ParamParser<LayoutTransformParam>)
.set_attr<FInferShape>("FInferShape", LayoutTransformInferShape)
.set_attr<FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<FCorrectLayout>(
  "FCorrectLayout", [](const NodeAttrs& attrs,
                     std::vector<Layout> *ilayouts,
                     const std::vector<Layout> *last_ilayouts,
                     std::vector<Layout> *olayouts) {
    const LayoutTransformParam& param = cvm::get<LayoutTransformParam>(attrs.parsed);
    CHECK_EQ(ilayouts->size(), 1U);
    CHECK_EQ(olayouts->size(), 1U);
    NNVM_ASSIGN_LAYOUT(*ilayouts, 0, Layout(param.src_layout));
    NNVM_ASSIGN_LAYOUT(*olayouts, 0, Layout(param.dst_layout));
    return true;
})
.set_support_level(1);

DMLC_REGISTER_PARAMETER(LRNParam);

inline bool LRNInferShape(const cvm::NodeAttrs& attrs,
                          std::vector<TShape>* in_shape,
                          std::vector<TShape>* out_shape) {
  TShape dshape = (*in_shape)[0];
  TShape oshape = dshape;

  NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_shape, 0, oshape);
  return true;
}

NNVM_REGISTER_OP(lrn)
.describe(R"code(LRN layer)code" NNVM_ADD_FILELINE)
.add_argument("data", "4D Tensor", "Input data.")
.set_attr_parser(ParamParser<LRNParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<LRNParam>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<FInferShape>("FInferShape", LRNInferShape)
.set_attr<FInferType>("FInferType", ElemwiseType<1, 1>)
.set_support_level(1);

DMLC_REGISTER_PARAMETER(L2NormalizeParam);

inline bool L2NormalizeInferShape(const cvm::NodeAttrs& attrs,
                                  std::vector<TShape>* in_shape,
                                  std::vector<TShape>* out_shape) {
  TShape dshape = (*in_shape)[0];
  TShape oshape = dshape;

  NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_shape, 0, oshape);
  return true;
}

NNVM_REGISTER_OP(l2_normalize)
.describe(R"code(L2NORMALIZE layer)code" NNVM_ADD_FILELINE)
.add_argument("data", "4D Tensor", "Input data.")
.set_attr_parser(ParamParser<L2NormalizeParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<L2NormalizeParam>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<FInferShape>("FInferShape", L2NormalizeInferShape)
.set_attr<FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<FCorrectLayout>("FCorrectLayout", ElemwiseArbitraryLayout<1, 1>)
.set_support_level(1);
*/
}  // namespace top
}  // namespace cvm
