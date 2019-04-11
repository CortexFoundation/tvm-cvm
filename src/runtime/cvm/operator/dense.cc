#include "../graph_runtime.h"
#include "../op.h"

#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/serializer.h>

#include <algorithm>
#include <functional>
#include <numeric>
#include <vector>
#include <string>
#include <memory>
#include <utility>
namespace tvm {
namespace runtime {
namespace cvm{

void CVMAdd(TVMArgs args, TVMRetValue* rv) {
  DLTensor *x = args[0];
  DLTensor *w = args[1];
  DLTensor *y = args[2];
  // std::cout << x->dtype << "  " << w->dtype << " " << y->dtype << "\n";
  // std::cout << "dim  = " << x->ndim << "  " << w->ndim << " " << y->ndim << "\n";
  // std::cout << x->shape[0] << "  " << w->shape[0] << " " << y->shape[0] << "\n";
  for (uint32_t i = 0; i < x->shape[0]; i++) {
		static_cast<int32_t*>(y->data)[i] = static_cast<int32_t*>(x->data)[i] + static_cast<int32_t*>(w->data)[i];
  }
}

void CVMClip(TVMArgs args) {
  DLTensor *x = args[0];
  DLTensor *y = args[1];
  for (uint32_t i = 0; i < x->shape[0]; i++) {
		static_cast<int32_t*>(y->data)[i] = std::max(std::min(127, static_cast<int32_t*>(x->data)[i]), -127);
  }
  // auto cudnn = tvm::runtime::Registry::Get("tvm.contrib.cudnn.conv2d.forward");
}

void Dense(TVMArgs args) {
	std::cout << "fuse_dens = " <<  args.size() << "\n";
	DLTensor *x = args[0];
	DLTensor *w = args[1];
	DLTensor *y = args[2];
	CVMPrint(std::vector<uint32_t>(x->shape, x->shape + x->ndim), "xshape");
	CVMPrint(std::vector<uint32_t>(y->shape, y->shape + y->ndim), "yshape");
	CVMPrint(std::vector<uint32_t>(w->shape, w->shape + w->ndim), "wshape");
	auto dx = static_cast<int32_t*>(x->data);
	auto dy = static_cast<int32_t*>(y->data);
	auto dw = static_cast<int32_t*>(w->data);
	CVMPrint(std::vector<int32_t>(dx, dx + x->shape[0] * x->shape[1]), "x");
	assert(y->shape[0] == 1); // not tested yet
	for (uint32_t di = 0; di < y->shape[0]; di++) {
			for (uint32_t oi = 0; oi < y->shape[1]; oi++) {
					int32_t sum = 0;
					for (uint32_t xi = 0; xi < x->shape[1]; xi++) {
							sum += dx[di * y->shape[1] + xi] * dw[oi * w->shape[1] + xi];
					}
					dy[di * y->shape[1] + oi] = sum;
			}
	}
}

CVM_REGISTER_OP(dense)
.set_attr<FCVMCompute>("FCVMCompute", Dense)

}
}
}
