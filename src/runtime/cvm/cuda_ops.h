#ifndef CUDA_OP_H
#define CUDA_OP_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

void cuda_elemwise_add(int32_t *a, int32_t *b, int32_t *c, int32_t n);
void cuda_conv2d(
        int32_t *input, int32_t i_n, int32_t i_c, int32_t i_h, int32_t i_w,
        int32_t *filter, int32_t f_n, int32_t f_c, int32_t f_h, int32_t f_w,
        int32_t *bias,
        int32_t padding,
        int32_t stride,
        int32_t dilation,
        int32_t groups,
        int32_t *output, int32_t o_n, int32_t o_c, int32_t o_h, int32_t o_w);

#endif
