#include <dmlc/any.h>
#include <cvm/bind.h>
#include <cvm/top/nn.h>
#include <cvm/top/tensor.h>

namespace cvm {

using dmlc::any;
using namespace top;

bool OpParamBinding::has(const std::string& name) {
  auto it = _map.find(name);
  return it != _map.end();
}

any OpParamBinding::get(const std::string& name, const std::string& json_) {
  auto it = _map.find(name);
  return std::move(it->second(json_));
}

void OpParamBinding::reg(const std::string& name, PtrCreateParam method) {
  _map[name] = method;
}

OpParamBinding& OpParamBinding::instance() {
  static OpParamBinding binding;
  return binding;
}

class BindingOpParamAction {
  public:
    BindingOpParamAction(const std::string& name, PtrCreateParam ptr) {
      OpParamBinding::instance().reg(name, ptr);
    }
};

#define BIND_OP_PARAM(OpName, Param)                            \
  any paramCreator##OpName(const std::string& json_) {          \
    std::istringstream is(json_);                               \
    dmlc::JSONReader reader(&is);                               \
    Param t;                                                    \
    t.Load(&reader);                                            \
    return t;                                                   \
  }                                                             \
  BindingOpParamAction g_bindOp##OpName(                        \
    #OpName, (PtrCreateParam)paramCreator##OpName)

BIND_OP_PARAM(conv2d, Conv2DParam);

BIND_OP_PARAM(dense, DenseParam);

BIND_OP_PARAM(conv2dt, Conv2DTransposeParam);

BIND_OP_PARAM(max_pool2d, MaxPool2DParam);

BIND_OP_PARAM(clip, ClipParam);

BIND_OP_PARAM(reshape, ReshapeParam);

BIND_OP_PARAM(flatten, ReshapeParam);

BIND_OP_PARAM(broadcast_add, BroadcastToParam);

BIND_OP_PARAM(broadcast_sub, BroadcastToParam);

BIND_OP_PARAM(broadcast_div, BroadcastToParam);

BIND_OP_PARAM(broadcast_mul, BroadcastToParam);

BIND_OP_PARAM(broadcast_right_shift, BroadcastToParam);

BIND_OP_PARAM(broadcast_left_shift, BroadcastToParam);

BIND_OP_PARAM(cast, CastParam);

BIND_OP_PARAM(sum, ReduceParam);

BIND_OP_PARAM(max, ReduceParam);

BIND_OP_PARAM(cvm_clip, CVMClipParam);

BIND_OP_PARAM(cvm_left_shift, CVMLeftShiftParam);

BIND_OP_PARAM(cvm_right_shift, CVMRightShiftParam);

};
