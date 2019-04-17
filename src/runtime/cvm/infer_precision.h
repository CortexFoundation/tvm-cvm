/*!
 *  Copyright (c) 2016 by Contributors
 * \file infer_prec.h
 * \brief Register the shapes given existin information.
 */
#include <string>
#include <vector>
#include <utility>
#include <unordered_map>
#include <functional>
#include <stdio.h>

using std::string;
using std::vector;
using std::pair;
using AttrMap = std::unordered_map<string, string>;
using FInferPrecision = 
	std::function<bool (vector<int>* iattr, vector<int>* oattr, AttrMap* attr)>;

#define REGISTER_OP_INFERPREC(OpName, ComputeRule)                   \
  RegisterAction g_creatorRegister##OpName(#OpName, ComputeRule)

inline bool Same_(vector<int>* iattr, vector<int>* oattr, AttrMap* attr) {
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

inline pair<int, int> GetPair(const string& str) {
	int ret0, ret1;
  sscanf(str.c_str(), "%d,%d", &ret0, &ret1);
	return std::make_pair(ret0, ret1);
}

inline int GetInt(const string& str) {
	int ret;
  sscanf(str.c_str(), "%d", &ret);
	return ret;
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

FInferPrecision FInferPrecisionMap::get(const string &key, FInferPrecision fdefault = Same_) {
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

inline bool MaxPlus_1_(vector<int>* iattr, vector<int>* oattr, AttrMap* attr) {
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

inline bool Sum_(vector<int>* iattr, vector<int>* oattr, AttrMap* attr) {
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

inline bool First_(vector<int>* iattr, vector<int>* oattr, AttrMap* attr) {
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

REGISTER_OP_INFERPREC(add, MaxPlus_1_);

REGISTER_OP_INFERPREC(broadcast_add, MaxPlus_1_);

REGISTER_OP_INFERPREC(sub, MaxPlus_1_);

REGISTER_OP_INFERPREC(broadcast_sub, MaxPlus_1_);

REGISTER_OP_INFERPREC(mul, Sum_);

REGISTER_OP_INFERPREC(broadcast_mul, Sum_);

REGISTER_OP_INFERPREC(div, First_);

REGISTER_OP_INFERPREC(broadcast_div, First_);

inline bool Dense_(vector<int>* iattr, vector<int> *oattr, AttrMap* attr){
  if (iattr->size() == 3) {
    if (iattr->at(2) == 8) {
      (*iattr)[2] = 31;
    }
  }
  (*oattr)[0] = 32;
  return true;
}

REGISTER_OP_INFERPREC(dense, Dense_);

inline bool Clip_(vector<int>* iattr, vector<int> *oattr, AttrMap* attr){
  (*oattr)[0] = 8;
  return true;
}

REGISTER_OP_INFERPREC(clip, Clip_);

