#include <iostream>
#include <string.h>
#include <time.h>
#include "../cuda_ops.h"

void conv_cpu(int* x_data, int n_batch, int x_h, int x_w, int in_channels,
        int *w_data, int filter_h, int filter_w,
        int *b_data,
        int *y_data, int o_h, int o_w, int out_channels,
        int stride_h, int stride_w,
        int padding_h, int32_t padding_w,
        int dilation_h, int dilation_w
        ){
#define GETX(n, c, h, w) x_data[(n) * in_channels * x_h * x_w + (c) * x_h * x_w + (h) * x_w + (w)]
#define GETW(o, i, h, w) w_data[(o) * in_channels * filter_h * filter_w + (i) * filter_h * filter_w + (h) * filter_w + (w)]
#define GETY(n, c, h, w) y_data[(n) * out_channels * o_h * o_w + (c) * o_h * o_w + (h) * o_w + (w)]
    auto calc_func = [&](int n, int k, int p, int q) {
        int y_sum = 0;
        for (int c = 0; c < in_channels; ++c) {
            for (int r = 0; r < filter_h; ++r) {
                for (int s = 0; s < filter_w; ++s) {
                    auto tp = p * stride_h + r*dilation_h - padding_h;
                    auto tq = q * stride_w + s*dilation_w - padding_w;
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
void conv_v2(int *x_data, int n_batch, int in_channels, int x_h, int x_w,
	int *w_data, int filter_h, int filter_w,
	int *b_data,
	int *y_data, int out_channels, int o_h, int o_w,
	int stride_h, int stride_w, int padding_h, int padding_w, int dilation_h, int dilation_w){
        const int y_n_offset = out_channels * o_h * o_w;
        const int y_c_offset = o_h * o_w;
        const int y_h_offset = o_w;
        //const int x_n_offset = in_channels * x_h * x_w;
        const int x_c_offset = x_h * x_w;
        const int x_h_offset = x_w;
        const int w_o_offset = in_channels * filter_h * filter_w;
        const int w_i_offset = filter_h * filter_w;
        const int w_h_offset = filter_w;
#define CONV2d_X(n, c, h, w) x_data[(n) * y_n_offset + (c) * x_c_offset + (h) * x_h_offset + (w)]
#define CONV2d_W(o, i, h, w) w_data[(o) * w_o_offset + (i) * w_i_offset + (h) * w_h_offset + (w)]
#define CONV2d_Y(n, c, h, w) y_data[(n) * y_n_offset + (c) * y_c_offset + (h) * y_h_offset + (w)]
        auto calc_func = [&](int n, int k, int p, int q) {
            int y_sum = 0;
            for (int c = 0; c < in_channels; ++c) {
                for (int r = 0; r < filter_h; ++r) {
                    auto tp = p * stride_h + r*dilation_h - padding_h;
                    if (tp < 0 || tp >= x_h)
                        continue;
                    auto tq_start = q * stride_w - padding_w;
                    auto tq_end = q * stride_w - padding_w + filter_w;
                    for (auto tq = std::max(tq_start, 0); tq < std::min(tq_end, x_h); ++tq) {
                        auto s = tq - tq_start;
                        y_sum += CONV2d_X(n, c, tp, tq-s+s*dilation_w) * CONV2d_W(k, c, r, s);
                    }
                }
            }
            return y_sum;

        };
    for (int n = 0; n < n_batch; ++n) {
	for (int k = 0; k < out_channels; ++k) {
	    for (int p = 0; p < o_h; ++p) {
		for (int q = 0; q < o_w; ++q) {
		    CONV2d_Y(n, k, p, q) = (b_data != nullptr ? b_data[k] : 0) + calc_func(n, k, p, q);
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
    int i_n = 1;
    int i_c = 160;
    int i_h = 12;
    int i_w = 12;
    int f_h = 1;
    int f_w = 7;
    int o_c = 160;
    int padding_h = 0;
    int padding_w = 3;
    int stride_h = 1;
    int stride_w = 1;
    int dilation_h = 1;
    int dilation_w= 1;

    for(i_n = 1; i_n < 4; i_n++){
    for(i_c = 1; i_c < 2; i_c++){
    for(f_w = 1; i_h < 64; i_h++){
        i_w = i_h;
    for(f_h = 1; f_h <= 11; f_h+=2){
    for(f_w = 1; f_w <= 11; f_w += 2){
//        f_w = f_h;
    for(o_c = 1; o_c <= 16; o_c++){
        int tmp_f_h = (f_h - 1) * dilation_h + 1;
        int tmp_f_w = (f_w - 1) * dilation_w + 1;
        int o_h = (i_h + 2 * padding_h - tmp_f_h) / stride_h + 1;
        int o_w = (i_w + 2 * padding_w - tmp_f_w) / stride_w + 1;
        if(o_h <= 0 || o_w <= 0) continue;
        std::cout << i_n << " " << i_c << " " << i_h << " " << i_w << " " << f_h << " " << f_w << " " << o_c << std::endl;
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
            filter[i] = 2;
        for(int i = 0; i < o_c; i++)
            b_data[i] = 1;
    //    print(input, i_c, i_h, i_w);
        clock_t start = clock();
        for(int i = 0; i < 1; i++){
            conv_cpu(input, i_n, i_h, i_w, i_c,
                    filter, f_h, f_w,
                    b_data,
                    output, o_h, o_w, o_c,
                    stride_h, stride_w,
                    padding_h, padding_w,
                    dilation_h, dilation_w);
        }
        clock_t end = clock();
    //    print(output, i_n, o_c, o_h, o_w);
        std::cout << "cpu time: " << end-start << std::endl;
		int *output3 = new int[s_o];
            conv_v2(input, i_n, i_c, i_h, i_w,
                    filter, f_h, f_w,
                    b_data,
                    output3, o_c, o_h, o_w,
                    stride_h, stride_w,
                    padding_h, padding_w,
                    dilation_h, dilation_w);
	
        int ret3 = memcmp(output, output3, sizeof(int) * s_o);
        std::cout << (ret3 == 0 ? "pass" : "failed") << std::endl;
        if(ret3 != 0){
            std::cout << "cpu output:\n";
            print(output, i_n, i_c, i_h, i_w);
            std::cout << "cpu output2:\n";
            print(output3, i_n, o_c, o_h, o_w);
            return 0;
        }
	delete output3;
    //    int *output3 = new int[s_o];
    //    clock_t start2 = clock();
    //    for(int i = 0; i < 10; i++){
    //        conv_cpu_v2(input, i_n, i_h, i_w, i_c,
    //                filter, f_h, f_w,
    //                b_data,
    //                output3, o_h, o_w, o_c,
    //                stride, stride,
    //                padding);
    //    }
    //    clock_t end2 = clock();
    ////  print(output3, i_n, o_c, o_h, o_w);
    //    std::cout << "cpu time: " << end2-start2 << std::endl;
    //    int ret = memcmp(output, output3, s_o*sizeof(int32_t));
    //    std::cout << (ret == 0 ? "pass" : "failed") << std::endl;

        int* output2 = new int[s_o];
        const char* errorStr = cuda_conv2d(
            input, i_n, i_c, i_h, i_w,
            filter, o_c, i_c, f_h, f_w,
            b_data,
            padding_h, padding_w,
            stride_h, stride_w,
            dilation_h, dilation_w,
            1,
            output2, i_n, o_c, o_h, o_w, 0, true);

        clock_t gpu_end = clock();
        std::cout << "gpu all time: " << gpu_end - end << std::endl;
        if(errorStr != NULL){
            std::cout << errorStr << std::endl;
            return 0;
        }
    //    print(output2, i_n, o_c, o_h, o_w);
        int ret2 = memcmp(output, output2, sizeof(int) * s_o);
        std::cout << (ret2 == 0 ? "pass" : "failed") << std::endl;
        if(ret2 != 0){
            std::cout << "cpu output:\n";
            print(output, i_n, i_c, i_h, i_w);
            std::cout << "cuda output:\n";
            print(output2, i_n, o_c, o_h, o_w);
            return 0;
        }
        delete input;
        delete filter;
        delete b_data;
        delete output;
        delete output2;
    }
    }
    }
    }
    }
    }
    return 0;
}
