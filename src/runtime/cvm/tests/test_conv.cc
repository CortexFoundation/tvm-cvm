#include <iostream>
#include <string.h>
#include <time.h>
#include "../cuda_ops.h"

void conv_cpu(int* x_data, int n_batch, int x_h, int x_w, int in_channels,
        int *w_data, int filter_h, int filter_w,
        int *b_data,
        int *y_data, int o_h, int o_w, int out_channels,
        int stride_h, int stride_w,
        int padding
        ){
#define GETX(n, c, h, w) x_data[(n) * in_channels * x_h * x_w + (c) * x_h * x_w + (h) * x_w + (w)]
#define GETW(o, i, h, w) w_data[(o) * in_channels * filter_h * filter_w + (i) * filter_h * filter_w + (h) * filter_w + (w)]
#define GETY(n, c, h, w) y_data[(n) * out_channels * o_h * o_w + (c) * o_h * o_w + (h) * o_w + (w)]
    auto calc_func = [&](int n, int k, int p, int q) {
        int y_sum = 0;
        for (int c = 0; c < in_channels; ++c) {
            for (int r = 0; r < filter_h; ++r) {
                for (int s = 0; s < filter_w; ++s) {
                    auto tp = p * stride_h + r - padding;
                    auto tq = q * stride_w + s - padding;
                    if (tp < 0 || tq < 0 || tp >= x_h || tq >= x_w)
                        continue;
                    y_sum += GETX(n, c, tp, tq) * GETW(k, c, r, s);
                }
            }
        }
        return y_sum;

    };
    for (int n = 0; n < n_batch; ++n) {
        for (int k = 0; k < out_channels; ++k) {
            for (int p = 0; p < o_h; ++p) {
                for (int q = 0; q < o_w; ++q) {
                    GETY(n, k, p, q) = b_data[k] + calc_func(n, k, p, q);
                }
            }
        }
    }
}


void print(int* data, int n, int c, int h, int w){
    for(int in = 0; in < n; in++){
    for(int i = 0; i < c; i++){
        for(int j = 0; j < h; j++){
            for(int k = 0; k < w; k++){
                std::cout << data[in*c*h*w + i*h*w+j*w+k] << " ";
            }
            std::cout << std::endl;
        }
    }
    }
}
int main(){
    int i_n = 30;
    int i_c = 10;
    int i_h = 34;
    int i_w = 32;
    int f_h = 3;
    int f_w = 3;
    int o_c = 102;
    int padding = 1;
    int stride = 1;
    int o_h = (i_h + 2 * padding - f_h) / stride + 1;
    int o_w = (i_w + 2 * padding - f_w) / stride + 1;
    size_t s_i = i_n * i_c * i_h * i_w;
    size_t s_f = o_c * i_c * f_h * f_w;
    size_t s_o = i_n * o_c * o_h * o_w;
    int *input = new int[s_i];
    int *filter = new int[s_f];
    int *b_data = new int[o_c];
    int *output = new int[s_o];
    for(int i = 0; i < s_i; i++)
        input[i] = 1;
    for(int i = 0; i < s_f; i++)
        filter[i] = 1;
    for(int i = 0; i < o_c; i++)
        b_data[i] = 0;
//    print(input, i_c, i_h, i_w);
    clock_t start = clock();
    conv_cpu(input, i_n, i_h, i_w, i_c,
        filter, f_h, f_w,
        b_data,
        output, o_h, o_w, o_c,
        stride, stride,
        padding);
    clock_t end = clock();
    std::cout << "cpu time: " << end-start << std::endl;
//    print(output, i_n, o_c, o_h, o_w);

    int* output2 = new int[s_o];
    cuda_conv2d(
        input, i_n, i_c, i_h, i_w,
        filter, o_c, i_c, f_h, f_w,
        b_data,
        padding,
        stride,
        1,
        1,
        output2, i_n, o_c, o_h, o_w, true);

    clock_t gpu_end = clock();
    std::cout << "gpu all time: " << gpu_end - end << std::endl;
//    print(output2, i_n, o_c, o_h, o_w);
    int ret = memcmp(output, output2, sizeof(int) * s_o);
    std::cout << (ret == 0 ? "pass" : "failed") << std::endl;
    return 0;
}
