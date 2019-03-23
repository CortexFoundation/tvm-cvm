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

/*
 * args = {
 * data, 2D int8
 * weight, 2D int8
 * bias, 1D int32, optional
 * out, 2D int32
 * }
 *
 */
TVM_REGISTER_GLOBAL("tvm.contrib.cvm.dense.forward")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    int argsSize = args.num_args;
    DLTensor *a = args[0];
    DLTensor *b = args[1];
    DLTensor *c = args[2];
    DLTensor *d;
    if(argsSize == 4) d = args[3];
    int8_t *data = (int8_t*)(a->data);
    int data_h = static_cast<int>(a->shape[0]);
    int data_w = static_cast<int>(a->shape[1]);

    int8_t *weight = (int8_t*)(b->data);
    int weight_h = static_cast<int>(b->shape[0]);
    int weight_w = static_cast<int>(b->shape[1]);

    int32_t *bias = argsSize == 4 ? (int32_t*)(c->data) : NULL;
    int bias_l = bias != NULL ? static_cast<int>(c->shape[0]) : 0;

    int32_t *out = argsSize == 4 ? (int32_t*)(d->data) : (int32_t*)(c->data);

    CHECK_EQ(data_w, weight_h) << "data width should equal weight height";

    for(int oh = 0; oh < data_h; oh++){
        for(int ow = 0; ow < weight_w; ow++){
            int32_t sum = 0;
            for(int k = 0; k < data_w; k++){
                sum += data[oh * data_w + k] * weight[k * weight_w + ow];
            }
            out[oh * weight_w + ow] = sum;
        }
    }

    if(bias_l > 0){
        for(int i = 0; i < data_h; i++){
            for(int j = 0; j < weight_w; j++){
                out[i *weight_w + j] += bias[j];
            }
        }
    }
});

}
}
