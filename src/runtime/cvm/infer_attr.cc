/*!
 *  Copyright (c) 2016 by Contributors
 * \file infer_attr.cc
 * \brief Inference the attrs given existin information.
 */
#include "graph_runtime.h"
#include "infer_precision.h"
#include <nnvm/op.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/graph_attr_types.h>
#include <iostream>

using nnvm::Op;
using nnvm::TShape;

namespace tvm {
namespace runtime {

void CvmRuntime::SetupAttr() {
  std::cout << "try to setup shape" << std::endl;
  SetupShape();
  std::cout << "try to setup type" << std::endl;
  SetupType();
  std::cout << "try to setup prec" << std::endl;
  SetupPrecision();
  for (auto p:  attrs_.precision) {
    std::cout << p << ' ';
  }
  std::cout << std::endl;
}

std::string GetOpName(std::string name) {
  bool has_underline = false;
  for (int i = name.size() - 1; i >= 0; --i) {
    if (name[i] >= '0' && name[i] <= '9') continue;
    if (name[i] == '_') has_underline = true;
    else if (has_underline) {
      return name.substr(0, i + 1);
    }
  }
  return name;
}

void CvmRuntime::SetupPrecision() {
  std::vector<Node> &idx = nodes_;
  std::vector<int> &precision = attrs_.precision;
  precision.resize(nodes_.size(), -1);
  // Temp space for shape inference.
  std::vector<int> iprec, oprec;
	auto finfer_prec = FInferPrecisionMap::getInstance();

  // inference step function for nid
  auto infer_prec = [&](uint32_t nid) {
    const auto& inode = idx[nid];
    if (inode.op_type == "null") {
      if (precision[nid] == -1) {
         precision[nid] = 8;
      }
      // Variable node. No operator. Only one output entry.
    } else {
      const uint32_t num_inputs = inode.param.num_inputs;
      const uint32_t num_outputs = inode.param.num_outputs;
      bool forward_known = true;
      // Forward operator inference.
      iprec.resize(num_inputs, -1);
      for (uint32_t i = 0; i < iprec.size(); ++i) {
        iprec[i] = precision[inode.inputs[i].node_id];
        if (iprec[i] == -1) forward_known = false;
      }
      oprec.resize(num_outputs, -1);
      oprec[0] = precision[nid];
      if (oprec[0] == -1) forward_known = false;
      // which raise an error if the op has bit been registered.
      // TODO: pre-check or try-catch is needed.
      auto opname = GetOpName(inode.param.func_name);
      std::cout << opname << std::endl;
      auto op = Op::Get(opname);
      auto finfer = finfer_prec.get(opname);
      if (!forward_known) {
        if (finfer != nullptr) {
          // Call inference function of the operator.
          try {
            nnvm::NodeAttrs attrs;
            attrs.op = op;
            attrs.name = opname;
            forward_known = finfer(opname, &iprec, &oprec, nullptr);
          } catch (const std::exception& e) {
            throw dmlc::Error(e.what() + std::string(" with ") + opname);
          }
        } else {
          // TODO: Error
        }
      }
      // Save to the result map.
      for (uint32_t i = 0; i < num_inputs; ++i) {
        if (precision[inode.inputs[i].node_id] == -1) {
          precision[inode.inputs[i].node_id] = iprec[i];
        } else {
          CHECK_EQ(precision[inode.inputs[i].node_id], iprec[i])
             << "Check precision failed, "
            << "expected to be " << iprec[i]
            << " but " << precision[inode.inputs[i].node_id];
        }
			}
      precision[nid] = oprec[0];
    }
  };

  for (uint32_t nid = 0; nid < idx.size(); ++nid) {
    infer_prec(nid);
  }
}

std::vector<TShape> GetTShapeArray(const std::vector<std::vector<int64_t> > &shapes) {
  std::vector<TShape> ret;
  for (auto shape : shapes) {
    if (shape.size() == 1) {
      ret.push_back(TShape{shape[0]});
    } else if (shape.size() == 2) {
      ret.push_back(TShape{shape[0], shape[1]});
    } else if (shape.size() == 3) {
      ret.push_back(TShape{shape[0], shape[1], shape[2]});
    } else if (shape.size() == 4) {
      ret.push_back(TShape{shape[0], shape[1], shape[2], shape[3]});
    } else {
      ret.push_back(TShape());
    }
  }
  return ret;
}

void CvmRuntime::SetupShape() {
  auto &idx = nodes_;
  const auto rshape = GetTShapeArray(attrs_.shape);
  static auto& finfer_shape =
      Op::GetAttr<nnvm::FInferNodeEntryAttr<TShape> >("FInferShape");
  // reshape shape vector
  // Temp space for shape inference.
  std::vector<TShape> ishape, oshape;

  // inference step function for nid
  auto infer_shape = [&](uint32_t nid) {
    const auto& inode = idx[nid];
    if (inode.op_type == "null") {
      // Variable node. No operator. Only one output entry.
      CHECK(rshape[nid].ndim()) << "Invalid variable shape";
    } else {
      const uint32_t num_inputs = inode.param.num_inputs;
      const uint32_t num_outputs = inode.param.num_outputs;
      bool forward_known = true;
      // Forward operator inference.
      ishape.resize(num_inputs, TShape());
      for (uint32_t i = 0; i < ishape.size(); ++i) {
        ishape[i] = rshape[inode.inputs[i].node_id];
        if (ishape[i].ndim() == 0) forward_known = false;
      }
      oshape.resize(num_outputs, TShape());
      oshape[0] = rshape[nid];
      if (oshape[0].ndim() == 0) forward_known = false;
      // which raise an error if the op has not been registered.
      // TODO: pre-check or try-catch is needed.
      auto opname = GetOpName(inode.param.func_name);
      auto op = nnvm::Op::Get(opname);
      auto finfer = finfer_shape.get(op, nullptr);
      if (!forward_known) {
        if (finfer != nullptr) {
          // Call inference function of the operator.
          try {
            nnvm::NodeAttrs attrs;
            attrs.op = op;
            attrs.name = opname;
            forward_known = finfer(attrs, &ishape, &oshape);
          } catch (const std::exception& e) {
            throw dmlc::Error(e.what() + std::string(" with ") + opname);
          }
        } else {
          // TODO: error
        }
      }
      // Save to the result map.
      for (uint32_t i = 0; i < num_inputs; ++i) {
        CHECK_EQ(ishape[i], rshape[inode.inputs[i].node_id])
            << "Check shape failed, "
            << "expected to be " << ishape[i]
            << " but " << rshape[inode.inputs[i].node_id];
      }
      CHECK_EQ(oshape[0], rshape[nid])
          << "Check shape failed, "
          << "expected to be " << oshape[0]
          << " but " << rshape[nid];
    }
  };

  for (uint32_t nid = 0; nid < idx.size(); ++nid) {
    infer_shape(nid);
  }
}

// inference fucntion for same type
inline bool SameType(const std::vector<int>& attrs,
                     std::vector<int> *iattr,
                     std::vector<int> *oattr) {
  int def_v = -1;
  for (int v : *oattr) {
    if (v != -1) {
      def_v = v; break;
    }
  }
  if (def_v == -1) {
    for (int v : *iattr) {
      if (v != -1) {
        def_v = v; break;
      }
    }
  }
  if (def_v == -1) return false;
  for (int& v : *oattr) {
    v = def_v;
  }
  for (int& v : *iattr) {
    v = def_v;
  }
  return true;
}

void CvmRuntime::SetupType() {
  auto &idx = nodes_;
  std::vector<int> rtype;
  rtype.resize(nodes_.size(), 4);
  static auto& finfer_type =
      Op::GetAttr<nnvm::FInferNodeEntryAttr<int> >("FInferType");
  // reshape shape vector
  // Temp space for shape inference.
  std::vector<int> itype, otype;

  // inference step function for nid
  auto infer_type = [&](uint32_t nid) {
    const auto& inode = idx[nid];
    if (inode.op_type == "null") {
      // Variable node. No operator. Only one output entry.
    } else {
      const uint32_t num_inputs = inode.param.num_inputs;
      const uint32_t num_outputs = inode.param.num_outputs;
      bool forward_known = true;
      // Forward operator inference.
      itype.resize(num_inputs, -1);
      for (uint32_t i = 0; i < itype.size(); ++i) {
        itype[i] = rtype[inode.inputs[i].node_id];
        if (itype[i] == -1) forward_known = false;
      }
      otype.resize(num_outputs, -1);
      otype[0] = rtype[nid];
      if (otype[0] == -1) forward_known = false;
      // which raise an error if the op has bit been registered.
      // TODO: pre-check or try-catch is needed.
      auto opname = GetOpName(inode.param.func_name);
      auto op = nnvm::Op::Get(opname);
      auto finfer = finfer_type.get(op, nullptr);
      if (!forward_known) {
        if (finfer != nullptr) {
          // Call inference function of the operator.
          try {
            nnvm::NodeAttrs attrs;
            attrs.op = op;
            attrs.name = opname;
            forward_known = finfer(attrs, &itype, &otype);
          } catch (const std::exception& e) {
            throw dmlc::Error(e.what() + std::string(" with ") + opname);
          }
        } else {
          // TODO: Error
        }
      }
      // Save to the result map.
      for (uint32_t i = 0; i < num_inputs; ++i) {
        CHECK_EQ(itype[i], rtype[inode.inputs[i].node_id])
          << "Check type failed, "
          << "expected to be " << itype[i]
          << " but " << rtype[inode.inputs[i].node_id];
			}
      CHECK_EQ(otype[0], rtype[nid])
        << "Check type failed, "
        << "expected to be " << otype[0]
        << " but " << rtype[nid];
    }
  };

  for (uint32_t nid = 0; nid < idx.size(); ++nid) {
    infer_type(nid);
  }
}
/*
 * assign
 * avg_pool2d
 * batch_norm
 * cast
 * clip
 * concatenate
 * conv2d
 * conv2d_transpose
 * dense
 * dropout
 * elemwise_add
 * exp
 * expand_dims
 * expand_like
 * flatten
 * flip
 * gather_nd
 * global_avg_pool2d
 * global_max_pool2d
 * l2_normalize
 * leaky_relu
 * log_softmax
 * lrn
 * matmul
 * max_pool2d
 * multibox_prior
 * multibox_transform_loc
 * non_max_suppression
 * pad
 * prelu
 * relu
 * reshape
 * reshape_like
 * resize
 * slice_like
 * softmax
 * split
 * squeeze
 * strided_slice
 * take
 * transpose
 * upsampling
 * where
 * yolo_reorg
 * broadcast_add
 * broadcast_sub
 * broadcast_mul
 * broadcast_div
 * broadcast_mod
 * broadcast_max
 * broadcast_min
 * broadcast_pow
 * broadcast_left_shift
 * broadcast_right_shift
 * broadcast_greater
 * broadcast_less
 * broadcast_equal
 * broadcast_not_equal
 * broadcast_greater_equal
 * broadcast_less_equal
 * floor
 * ceil
* trunc
* round
* abs
* sigmoid
* tanh
* exp
* log2
* log
* sqrt
* negative
* logical_not
* copy
* elemwise_add
* elemwise_sub
* elemwise_mul
* elemwise_div
* elemwise_mod
* elemwise_pow
* logical_and
* logical_or
* full
* zeros
* ones
* elemwise_sum
* greater
* less
 * */


}
}
