#include "graph_runtime.h"

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

void CVMClip(TVMArgs args, TVMRetValue* rv) {
  DLTensor *x = args[0];
  DLTensor *y = args[1];
  for (uint32_t i = 0; i < x->shape[0]; i++) {
		static_cast<int32_t*>(y->data)[i] = std::max(std::min(127, static_cast<int32_t*>(x->data)[i]), -127);
  }
  // auto cudnn = tvm::runtime::Registry::Get("tvm.contrib.cudnn.conv2d.forward");
}

}
}
