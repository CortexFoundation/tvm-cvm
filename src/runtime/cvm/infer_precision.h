/*!
 *  Copyright (c) 2016 by Contributors
 * \file infer_prec.h
 * \brief Register the shapes given existin information.
 */
#include <string>
#include <utility>
#include <unordered_map>
#include <nnvm/op_attr_types.h>

using nnvm::Op;
using nnvm::TShape;
using FInferPrecision = std::function<bool (const std::string& op,
																						std::vector<int> iattr,
																						std::vector<int> oattr)>;

class FInferPrecisionMap {
private:
	std::unordered_map<std::string, FInferPrecision> map_;
	FInferPrecisionMap() {};
public:
	FInferPrecision Get(std::string key);
	void Register(std::string name, FInferPrecision method);
	static FInferPrecisionMap& getInstance();
};

FInferPrecision FInferPrecisionMap::Get(std::string key) {
	auto it = map_.find(key);
	if (it == map_.end())
		return NULL;
	else
		return it->second;
}

void FInferPrecisionMap::Register(std::string name, FInferPrecision method) {
	map_.insert(std::pair<std::string, FInferPrecision>(name, method));
}

FInferPrecisionMap& FInferPrecisionMap::getInstance() {
	static FInferPrecisionMap finfermap;
	return finfermap;
}

class RegisterAction {
public:
  RegisterAction(const std::string &OpName, FInferPrecision finfer) {
		FInferPrecisionMap::getInstance().Register(OpName, finfer);
	}
  RegisterAction(const char* OpName, FInferPrecision finfer) {
		FInferPrecisionMap::getInstance().Register(std::string(OpName), finfer);
	}
};
/*
#define REGISTER_OP_INFER_PREC(OpName, ComputeRule)                   \
  RegisterAction g_creatorRegister##OpName(#OpName,                   \
                 static_cast<FInferPrecision>(ComputeRule))

inline bool PrecPlus_1(const std::string& op,
											 std::vector<int> *iattr,
                       std::vector<int> *oattr) {
  return true;
}

REGISTER_OP_INFER_PREC(add, PrecPlus_1);
*/

