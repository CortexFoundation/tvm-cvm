/*!
 *  Copyright (c) 2017 by Contributors
 * \file cvm/compiler/packed_func_ext.h
 * \brief Extension to enable packed functionn for cvm types
 */
#ifndef NNVM_COMPILER_PACKED_FUNC_EXT_H_
#define NNVM_COMPILER_PACKED_FUNC_EXT_H_

#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <cvm/graph.h>
#include <cvm/symbolic.h>
#include <string>
#include <vector>
#include <unordered_map>

namespace cvm {
namespace compiler {

using tvm::runtime::PackedFunc;

using AttrDict = std::unordered_map<std::string, std::string>;

/*!
 * \brief Get PackedFunction from global registry and
 *  report error if it does not exist
 * \param name The name of the function.
 * \return The created PackedFunc.
 */
inline const PackedFunc& GetPackedFunc(const std::string& name) {
  const PackedFunc* pf = tvm::runtime::Registry::Get(name);
  CHECK(pf != nullptr) << "Cannot find function " << name << " in registry";
  return *pf;
}
}  // namespace compiler
}  // namespace cvm

// Enable the graph and symbol object exchange.
namespace tvm {
namespace runtime {

template<>
struct extension_type_info<cvm::Symbol> {
  static const int code = 16;
};

template<>
struct extension_type_info<cvm::Graph> {
  static const int code = 17;
};

template<>
struct extension_type_info<cvm::compiler::AttrDict> {
  static const int code = 18;
};

}  // namespace runtime
}  // namespace tvm
#endif  // NNVM_COMPILER_PACKED_FUNC_EXT_H_
