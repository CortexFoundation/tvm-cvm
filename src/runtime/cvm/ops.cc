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

TVM_REGISTER_GLOBAL("tvm.runtime.cvm.add").set_body([](TVMArgs args, TVMRetValue* rv) {
   DLTensor *x = args[0];
   DLTensor *w = args[1];
   DLTensor *y = args[2];
   // std::cout << x->dtype << "  " << w->dtype << " " << y->dtype << "\n";
   // std::cout << "dim  = " << x->ndim << "  " << w->ndim << " " << y->ndim << "\n";
   // std::cout << x->shape[0] << "  " << w->shape[0] << " " << y->shape[0] << "\n";
   for (uint32_t i = 0; i < x->shape[0]; i++) {
 		static_cast<int32_t*>(y->data)[i] = static_cast<int32_t*>(x->data)[i] + static_cast<int32_t*>(w->data)[i];
   }
 });

TVM_REGISTER_GLOBAL("tvm.runtime.cvm.clip").set_body([](TVMArgs args, TVMRetValue* rv) {
   DLTensor *x = args[0];
   DLTensor *y = args[1];
   for (uint32_t i = 0; i < x->shape[0]; i++) {
 		static_cast<int32_t*>(y->data)[i] = std::max(std::min(127, static_cast<int32_t*>(x->data)[i]), -127);
   }
   // auto cudnn = tvm::runtime::Registry::Get("tvm.runtime.cudnn.conv2d");
 });

 TVM_REGISTER_GLOBAL("tvm.runtime.cvm.relu").set_body([](TVMArgs args, TVMRetValue* rv) {
   DLTensor *x = args[0];
   DLTensor *y = args[1];
   for (uint32_t i = 0; i < x->shape[0]; i++) {
 		static_cast<int32_t*>(y->data)[i] = std::max(static_cast<int32_t*>(x->data)[i], 0);
   }
   // auto cudnn = tvm::runtime::Registry::Get("tvm.runtime.cudnn.conv2d");
 });

TVM_REGISTER_GLOBAL("tvm.runtime.cvm.dense").set_body([](TVMArgs args, TVMRetValue* rv) {
  DLTensor *x = args[0];
  DLTensor *w = args[1];
  DLTensor *y = args[2];
  auto dx = static_cast<int32_t*>(x->data);
  auto dy = static_cast<int32_t*>(y->data);
  auto dw = static_cast<int32_t*>(w->data);
  // assert(y->shape[0] == 1); // not tested yet
  for (uint32_t di = 0; di < y->shape[0]; di++) {
      for (uint32_t oi = 0; oi < y->shape[1]; oi++) {
          int32_t sum = 0;
          for (uint32_t xi = 0; xi < x->shape[1]; xi++) {
              sum += dx[di * y->shape[1] + xi] * dw[oi * w->shape[1] + xi];
          }
          dy[di * y->shape[1] + oi] = sum;
      }
  }
});

TVM_REGISTER_GLOBAL("tvm.runtime.cvm.flatten").set_body([]
(TVMArgs args, TVMRetValue* rv){
     DLTensor *x = args[0];
     DLTensor *y = args[1];
     for (uint32_t i = 0; i < x->shape[0]; i++) {
         static_cast<int32_t*>(y->data)[i] = static_cast<int32_t*>(x->data)[i];
     }

});

TVM_REGISTER_GLOBAL("tvm.runtime.cvm.conv2d").set_body([]
 (TVMArgs args, TVMRetValue* rv){
    return;
     int stride_h = args[0];
     int stride_w = args[1];
     DLTensor *x = args[2];
     DLTensor *w = args[3];
     DLTensor *y = args[4];

     int32_t* x_data = (int32_t*)x->data;
     int32_t* w_data = (int32_t*)w->data;
     int32_t* y_data = (int32_t*)y->data;

     int out_channels = static_cast<int>(w->shape[0]);
     int filter_h = static_cast<int>(w->shape[2]);
     int filter_w = static_cast<int>(w->shape[3]);

     int n_batch = static_cast<int>(x->shape[0]);
     int in_channels = static_cast<int>(x->shape[1]);
     int x_h = static_cast<int>(x->shape[2]);
     int x_w = static_cast<int>(x->shape[3]);

     for (int n = 0; n < n_batch; ++n) {
         for (int k = 0; k < out_channels; ++k) {
             for (int p = 0; p < x_h; ++p) {
                 for (int q = 0; q < x_w; ++q) {
                     int32_t y_sum = 0;
                     for (int c = 0; c < in_channels; ++c) {
                         for (int r = 0; r < filter_h; ++r) {
                             for (int s = 0; s < filter_w; ++s) {
                                 y_sum +=
                                     x_data[(n * in_channels + c) * x_h * x_w + (p + r * stride_h) * x_w + (q + s * stride_w)] *
                                     w_data[(k * in_channels + c) * filter_h * filter_w + r * filter_w + s];
                             }
                         }
                     }
                     y_data[(n * out_channels + k) * x_h * x_w + p * x_w + q] = y_sum;
                 }
             }
         }
     }
 });

 inline int32_t getSize(DLTensor *dlTensor){
     int32_t size = 1;
     for(int i = 0; i < dlTensor->ndim; i++){
         size *= dlTensor->shape[i];
     }
     return size;
 }

TVM_REGISTER_GLOBAL("tvm.runtime.cvm.broadcast_add")
    .set_body([](TVMArgs args, TVMRetValue *ret){
            return;
        DLTensor *args0 = args[0];
        DLTensor *args1 = args[1];
        DLTensor *args2 = args[2];
        int32_t *a = static_cast<int32_t*>(args0->data);
        int32_t *b = static_cast<int32_t*>(args1->data);
        int32_t *c = static_cast<int32_t*>(args2->data);

        for(int i = 0; i < getSize(args0); i++){
            c[i] = a[i] + b[i];
        }
    });

TVM_REGISTER_GLOBAL("tvm.runtime.cvm.broadcast_sub")
    .set_body([](TVMArgs args, TVMRetValue *ret){
            return;
        DLTensor *args0 = args[0];
        DLTensor *args1 = args[1];
        DLTensor *args2 = args[2];
        int32_t *a = static_cast<int32_t*>(args0->data);
        int32_t *b = static_cast<int32_t*>(args1->data);
        int32_t *c = static_cast<int32_t*>(args2->data);

        for(int i = 0; i < getSize(args0); i++){
            c[i] = a[i] - b[i];
        }
    });
TVM_REGISTER_GLOBAL("tvm.runtime.cvm.broadcast_mul")
    .set_body([](TVMArgs args, TVMRetValue *ret){
            return;
        DLTensor *args0 = args[0];
        DLTensor *args1 = args[1];
        DLTensor *args2 = args[2];
        int32_t *a = static_cast<int32_t*>(args0->data);
        int32_t *b = static_cast<int32_t*>(args1->data);
        int32_t *c = static_cast<int32_t*>(args2->data);

        for(int i = 0; i < getSize(args0); i++){
            c[i] = a[i] * b[i];
        }
    });
TVM_REGISTER_GLOBAL("tvm.runtime.cvm.broadcast_div")
    .set_body([](TVMArgs args, TVMRetValue *ret){
            return;
        DLTensor *args0 = args[0];
        DLTensor *args1 = args[1];
        DLTensor *args2 = args[2];
        int32_t *a = static_cast<int32_t*>(args0->data);
        int32_t *b = static_cast<int32_t*>(args1->data);
        int32_t *c = static_cast<int32_t*>(args2->data);

        for(int i = 0; i < getSize(args0); i++){
            c[i] = a[i] / b[i];
        }
    });
TVM_REGISTER_GLOBAL("tvm.runtime.cvm.broadcast_right_shift")
    .set_body([](TVMArgs args, TVMRetValue *ret){
    return;
        DLTensor *args0 = args[0];
        DLTensor *args1 = args[1];
        DLTensor *args2 = args[2];
        int32_t *a = static_cast<int32_t*>(args0->data);
        int32_t *b = static_cast<int32_t*>(args1->data);
        int32_t *c = static_cast<int32_t*>(args2->data);

        for(int i = 0; i < getSize(args0); i++){
            int32_t rightA = ((a[i] >> (b[i] - 1)) + 1) >> 1;
            rightA = (rightA < 127 ? rightA : 127);
            rightA = (rightA > -127 ? rightA : -127);
            c[i] = rightA;
        }
    });
TVM_REGISTER_GLOBAL("tvm.runtime.cvm.broadcast_left_shift")
    .set_body([](TVMArgs args, TVMRetValue *ret){
            return; //TODO
        DLTensor *args0 = args[0];
        DLTensor *args1 = args[1];
        DLTensor *args2 = args[2];
        int32_t *a = static_cast<int32_t*>(args0->data);
        int32_t *b = static_cast<int32_t*>(args1->data);
        int32_t *c = static_cast<int32_t*>(args2->data);

        for(int i = 0; i < getSize(args0); i++){
            int32_t clipA = a[i] < 127 ? a[i] : 127;
            clipA = clipA > -127 ? clipA : -127;
            int32_t leftA = clipA << b[i];
            leftA = leftA < 127 ? leftA : 127;
            leftA = leftA > -127 ? leftA : -127;
            c[i] = leftA;
        }
    });

TVM_REGISTER_GLOBAL("tvm.runtime.cvm.max_pool2d")
    .set_body([](TVMArgs args, TVMRetValue *ret){
            return; //TODO
    });
TVM_REGISTER_GLOBAL("tvm.runtime.cvm.sum")
    .set_body([](TVMArgs args, TVMRetValue *ret){
            return; //TODO
    });
TVM_REGISTER_GLOBAL("tvm.runtime.cvm.elemwise_add")
    .set_body([](TVMArgs args, TVMRetValue *ret){
            return; //TODO
    });
TVM_REGISTER_GLOBAL("tvm.runtime.cvm.reshap")
    .set_body([](TVMArgs args, TVMRetValue *ret){
            return; //TODO
    });
}
}
