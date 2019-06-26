#ifndef CVM_RUNTIME_OPPARAM_BIND_H_
#define CVM_RUNTIME_OPPARAM_BIND_H_

#include <dmlc/json.h>
#include <dmlc/any.h>
#include <cvm/node.h>
#include <unordered_map>
#include <string>

namespace cvm {

using dmlc::any;

typedef any (*PtrCreateParam)(const std::string&);

class OpParamBinding {
  private:
    std::unordered_map<std::string, PtrCreateParam> _map;
    OpParamBinding(){}
  public:
    bool has(const std::string&);
    any get(const std::string&, const std::string&);
    void reg(const std::string &name, PtrCreateParam method);
    static OpParamBinding& instance();
};
};

#endif
