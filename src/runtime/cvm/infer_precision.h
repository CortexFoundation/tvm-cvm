/*!
 *  Copyright (c) 2016 by Contributors
 * \file infer_prec.h
 * \brief Register the shapes given existin information.
 */
#include <string>
#include <vector>
#include <utility>
#include <unordered_map>
#include <nnvm/op_attr_types.h>

using std::string;
using std::vector;
using nnvm::Op;
using nnvm::TShape;
using FInferPrecision = std::function<bool (const string& op,
																						vector<int>* iattr,
																						vector<int>* oattr,
																						vector<int>* attr)>;

#define REGISTER_OP_INFER_PREC(OpName, ComputeRule)                   \
  RegisterAction g_creatorRegister##OpName(#OpName, ComputeRule)

inline bool SamePrec(const string& op,
										 vector<int>* iattr,
                     vector<int>* oattr,
                     vector<int>* attr) {
  int def_v = -1;
  for (int v : *iattr) {
    if (v != -1) {
      def_v = v; break;
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

class FInferPrecisionMap {
private:
	std::unordered_map<string, FInferPrecision> map_;
	FInferPrecisionMap() {};
public:
	FInferPrecision get(const string &key, FInferPrecision fdefault);
	void Register(const string &name, FInferPrecision method);
	static FInferPrecisionMap& getInstance();
};

FInferPrecision FInferPrecisionMap::get(const string &key, FInferPrecision fdefault = SamePrec) {
	auto it = map_.find(key);
	if (it == map_.end())
		return fdefault;
	else
		return it->second;
}

void FInferPrecisionMap::Register(const string &name, FInferPrecision method) {
	map_.insert(std::pair<string, FInferPrecision>(name, method));
}

FInferPrecisionMap& FInferPrecisionMap::getInstance() {
	static FInferPrecisionMap finfermap;
	return finfermap;
}

class RegisterAction {
public:
  RegisterAction(const string &OpName, FInferPrecision finfer) {
		FInferPrecisionMap::getInstance().Register(OpName, finfer);
	}
};

inline bool MaxPlus_1_(const string& op,
											 vector<int>* iattr,
									     vector<int>* oattr,
                       vector<int>* attr) {
  int def_v = -1;
  for (int v : *iattr) {
    if (v > def_v) {
      def_v = v;
    }
  }
  if (def_v == -1) return false;
  for (int& v : *oattr) {
    v = def_v + 1;
  }
  for (int& v : *iattr) {
    if (v == -1) v = def_v;
  }
  return true;
}

inline bool Sum_(const string& op,
                     vector<int>* iattr,
                     vector<int>* oattr,
                     vector<int>* attr) {
  int def_v = 0;
  for (int v : *iattr) {
    if (v == -1) {
      return false;
    }
		def_v += v;
  }
  for (int& v : *oattr) {
    v = def_v;
  }
  return true;
}

inline bool First_(const string& op,
											 vector<int>* iattr,
									     vector<int>* oattr,
                       vector<int>* attr) {
  int def_v = iattr->at(0);
  if (def_v == -1) return false;
  for (int& v : *oattr) {
    v = def_v;
  }
  for (int& v : *iattr) {
    if (v == -1) v = def_v;
  }
  return true;
}

REGISTER_OP_INFER_PREC(add, MaxPlus_1_);

REGISTER_OP_INFER_PREC(broadcast_add, MaxPlus_1_);

REGISTER_OP_INFER_PREC(sub, MaxPlus_1_);

REGISTER_OP_INFER_PREC(broadcast_sub, MaxPlus_1_);

REGISTER_OP_INFER_PREC(mul, Sum_);

REGISTER_OP_INFER_PREC(broadcast_mul, Sum_);

REGISTER_OP_INFER_PREC(div, First_);

REGISTER_OP_INFER_PREC(broadcast_div, First_);


