/*!
 *  Copyright (c) 2017 by Contributors
 * \file nnvm/top/tensor.h
 * \brief Auxiliary param for tensor primitive.
 */
#ifndef NNVM_TOP_CVM_OP_H_
#define NNVM_TOP_CVM_OP_H_

#include <dmlc/base.h>
#include <dmlc/parameter.h>
#include <nnvm/tuple.h>

namespace nnvm {
namespace top {

struct CVMLUTParam : public dmlc::Parameter<CVMLUTParam> {
	int in_dim;
  DMLC_DECLARE_PARAMETER(CVMLUTParam) {
    DMLC_DECLARE_FIELD(in_dim)
      .describe("In dimension indicates the inputs value range.");
  }
};

struct CVMClipParam : public dmlc::Parameter<CVMClipParam> {
	int precision;
	bool is_sign;
  DMLC_DECLARE_PARAMETER(CVMClipParam) {
    DMLC_DECLARE_FIELD(precision)
      .describe("Precision such that value out of range this will be clipped.");
    DMLC_DECLARE_FIELD(is_sign).set_default(true)
      .describe("Clip range is sign int or unsigned int.");
  }
};

struct CVMLeftShiftParam : public dmlc::Parameter<CVMLeftShiftParam> {
	int precision;
	bool is_sign;
	int shift_bit;
  DMLC_DECLARE_PARAMETER(CVMLeftShiftParam) {
    DMLC_DECLARE_FIELD(precision)
      .describe("Precision such that value out of range this will be clipped.");
    DMLC_DECLARE_FIELD(is_sign).set_default(true)
      .describe("Clip range is sign int or unsigned int.");
		DMLC_DECLARE_FIELD(shift_bit)
			.describe("Left shift bit.");
  }
};

struct CVMRightShiftParam : public dmlc::Parameter<CVMRightShiftParam> {
	int precision;
	bool is_sign;
	int shift_bit;
  DMLC_DECLARE_PARAMETER(CVMRightShiftParam) {
    DMLC_DECLARE_FIELD(precision)
      .describe("Precision such that value out of range this will be clipped.");
    DMLC_DECLARE_FIELD(is_sign).set_default(true)
      .describe("Clip range is sign int or unsigned int.");
		DMLC_DECLARE_FIELD(shift_bit)
			.describe("Left shift bit.");
  }
};


}  // namespace top
}  // namespace nnvm

#endif  // NNVM_TOP_CVM_OP_H_
