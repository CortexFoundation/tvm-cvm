/*!
 *  Copyright (c) 2016 by Contributors
 * \file infer_prec.h
 * \brief Register the shapes given existin information.
 */
#include <string>
#include <unordered_map>
#include <nnvm/op_attr_types.h>

using nnvm::Op;
using nnvm::TShape;
using FInferPrecision = nnvm::FInferNodeEntryAttr<int>;

class FInferPrecisionDict {
  public:
    FInferPrecisionDict(){}
    ~FInferPrecisionDict(){}
    FInferPrecision get(const std::string& key, FInferPrecision fdefault) {
      auto it = map_.find(key);
      if (it != map_.end()) {
        return it->second;
      }
      return fdefault;
    }
  protected:
    std::unordered_map<std::string, FInferPrecision> map_;
};

class FInferPrecisionBase {
};


