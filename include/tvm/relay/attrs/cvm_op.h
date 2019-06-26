
#ifndef TVM_RELAY_ATTRS_TRANSFORM_H_
#define TVM_RELAY_ATTRS_TRANSFORM_H_

#include <tvm/attrs.h>
#include <string>

namespace tvm {
namespace relay {

struct CVMLUTAttrs : public tvm::AttrsNode<CVMLUTAttrs> {
  int in_dim;

  TVM_DECLARE_ATTRS(CVMLUTAttrs, "relay.attrs.CVMLUTAttrs") {
    TVM_ATTR_FIELD(in_dim)
        .describe("");
  }
};

struct CVMClipAttrs : public tvm::AttrsNode<CVMClipAttrs> {
  int precision;

  TVM_DECLARE_ATTRS(CVMClipAttrs, "relay.attrs.CVMClipAttrs") {
    TVM_ATTR_FIELD(precision)
        .describe("");
  }
};
struct CVMLeftShiftAttrs : public tvm::AttrsNode<CVMLeftShiftAttrs> {
  int precision;
  int shift_bit;

  TVM_DECLARE_ATTRS(CVMLeftShiftAttrs, "relay.attrs.CVMLeftShiftAttrs") {
    TVM_ATTR_FIELD(precision)
        .describe("");
    TVM_ATTR_FIELD(shift_bit)
        .describe("");
  }
};
struct CVMRightShiftAttrs : public tvm::AttrsNode<CVMRightShiftAttrs> {
  int precision;
  int shift_bit;

  TVM_DECLARE_ATTRS(CVMRightShiftAttrs, "relay.attrs.CVMRightShiftAttrs") {
    TVM_ATTR_FIELD(precision)
        .describe("");
    TVM_ATTR_FIELD(shift_bit)
        .describe("");
  }
};

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_ATTRS_TRANSFORM_H_
