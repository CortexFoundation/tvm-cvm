/*!
 *  Copyright (c) 2017 by Contributors
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
  int mode = args[0];
  int format = args[1];
  int algo = args[2];
  int pad_h = args[3];
  int pad_w = args[4];
  int stride_h = args[5];
  int stride_w = args[6];
  int dilation_h = args[7];
  int dilation_w = args[8];
  DLTensor *x = args[9];
  DLTensor *w = args[10];
  DLTensor *y = args[11];
  printf("%d\n", y->dtype);
  TVMContext ctx = x->ctx;

  int K = static_cast<int>(w->shape[0]);
  int R = static_cast<int>(w->shape[2]);
  int S = static_cast<int>(w->shape[3]);

  int N = static_cast<int>(x->shape[0]);
  int C = static_cast<int>(x->shape[1]);
  int H = static_cast<int>(x->shape[2]);
  int W = static_cast<int>(x->shape[3]);

//  CHECK(TypeMatch(x->dtype, kDLInt, 8) || TypeMatch(x->dtype, kDLInt, 16));
  size_t workspace_size = N * K * H * W;
  DeviceAPI* cpu_api = DeviceAPI::Get(ctx);
  int64_t in_shape[4] = {N, K, H, W};
  for (int n = 0; n < N; ++n) {
    for (int k = 0; k < K; ++k) {
      for (int p = 0; p < H; ++p) {
        for (int q = 0; q < W; ++q) {
          int y_acc = 0;
          for (int c = 0; c < C; ++c) {
            for (int r = 0; r < R; ++r) {
              for (int s = 0; s < S; ++s) {
                y_acc +=
                    ((char*)x->data)[(n * C + c) * H * W + (p + r * stride_h) * W + (q + s * stride_w)] *
                    ((char*)w->data)[(k * C + c) * R * S + r * S + s];
              }
            }
          }
          ((char*)y->data)[(n * K + k) * H * W + p * W + q] = static_cast<char>(y_acc & 0x7f);
        }
      }
    }
  }
});

}  // namespace contrib
}  // namespace tvm

