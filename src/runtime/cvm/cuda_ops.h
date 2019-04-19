#ifndef CUDA_OP_H
#define CUDA_OP_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

void cuda_elemwise_add(int32_t *a, int32_t *b, int32_t *c, int32_t n, bool debug);
void cuda_conv2d(
        int32_t *input, int32_t i_n, int32_t i_c, int32_t i_h, int32_t i_w,
        int32_t *filter, int32_t f_n, int32_t f_c, int32_t f_h, int32_t f_w,
        int32_t *bias,
        int32_t padding,
        int32_t stride,
        int32_t dilation,
        int32_t groups,
        int32_t *output, int32_t o_n, int32_t o_c, int32_t o_h, int32_t o_w, bool debug);
void cuda_depthwise_conv2d(
        int32_t *input, int32_t i_n, int32_t i_c, int32_t i_h, int32_t i_w,
        int32_t *filter, int32_t f_n, int32_t f_c, int32_t f_h, int32_t f_w,
        int32_t *bias,
        int32_t padding,
        int32_t stride,
        int32_t dilation,
        int32_t groups,
        int32_t *output, int32_t o_n, int32_t o_c, int32_t o_h, int32_t o_w, bool debug);
void cuda_max_pool(
        int32_t *input, int32_t i_n, int32_t i_c, int32_t i_h, int32_t i_w,
        int32_t f_h, int32_t f_w,
        int32_t padding,
        int32_t stride,
        int32_t *output, int32_t o_n, int32_t o_c, int32_t o_h, int32_t o_w, bool debug);
void cuda_dense(
        int32_t *a,
        int32_t *b,
        int32_t *c,
        const int m, const int k, const int n, int32_t *bias, bool debug);
void cuda_clip(const int32_t *x, int32_t *y, const int32_t n, const int32_t max, const int32_t min, bool debug);
void cuda_relu(const int32_t *x, int32_t *y, const int32_t n, bool debug);
void cuda_flatten(const int32_t *x, int32_t *y, const int32_t n, bool debug);
void cuda_broadcast_add(const int32_t *a, const int32_t *b, int32_t* c, const int32_t n, bool debug);
void cuda_broadcast_sub(const int32_t *a, const int32_t *b, int32_t* c, const int32_t n, bool debug);
void cuda_broadcast_mul(const int32_t *a, const int32_t *b, int32_t* c, const int32_t n, bool debug);
void cuda_broadcast_div(const int32_t *a, const int32_t *b, int32_t* c, const int32_t n, bool debug);
void cuda_broadcast_right_shift(const int32_t *a, const int32_t *b, int32_t* c, const int32_t n, bool debug);
void cuda_broadcast_left_shift(const int32_t *a, const int32_t *b, int32_t* c, const int32_t n, bool debug);
void cuda_sum(
        const int32_t *x,
        const int32_t n_batch, const int32_t channels, const int32_t h, const int32_t w,
        int32_t *y, bool debug);
void cuda_reshape(const int32_t *x, int32_t *y, int32_t size, bool debug);

#endif
