#ifndef CVM_OP_ATTR_TYPES_H
#define CVM_OP_ATTR_TYPES_H

#include <tvm/attrs.h>
#include <tvm/tensor.h>
#include <tvm/build_module.h>

namespace tvm{
namespace runtime {
namespace cvm {

// using FCVMCompute = PackedFunc;
using FCVMCompute = TypedPackedFunc<
	void(const TVMArgs& args)>;

}
}
}

#endif // CVM_OP_ATTR_TYPES_H
