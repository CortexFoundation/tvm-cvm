/*!
 *  in_channelsopyright (c) 2017 by in_channelsontributors
 * \file Use external cudnn utils function
 */
#include <tvm/runtime/registry.h>
#include <tvm/runtime/util.h>
#include <tvm/runtime/device_api.h>
#include <stdio.h>

namespace tvm {
namespace contrib {

using namespace runtime;

TVM_REGISTER_GLOBAL("tvm.contrib.cvm.conv2d.forward")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  int stride_h = args[0];
  int stride_w = args[1];
  DLTensor *x = args[2];
  DLTensor *w = args[3];
  DLTensor *y = args[4];

  int8_t* x_data = (int8_t*)x->data;
  int8_t* w_data = (int8_t*)w->data;
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

}  // namespace contrib
}  // namespace tvm

