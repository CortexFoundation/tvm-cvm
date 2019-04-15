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

inline void parseToIntPair(std::string str, int* ret){
	char a,b;
    sscanf(str.c_str(), "%c%d,%d%c", &a,ret, ret + 1, &b);
}
/**
* x
* y
* a_min -127
* a_max 127
*/
TVM_REGISTER_GLOBAL("tvm.runtime.cvm.clip").set_body([](TVMArgs args, TVMRetValue* rv) {
   DLTensor *x = args[0];
   DLTensor *y = args[1];
   std::string min_str = args[2];
   std::string max_str = args[3];
   int min = std::atoi(min_str.c_str());
   int max = std::atoi(max_str.c_str());
   for (uint32_t i = 0; i < x->shape[0]; i++) {
 		static_cast<int32_t*>(y->data)[i] = std::max(std::min(max, static_cast<int32_t*>(x->data)[i]), min);
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

/*
* x
* w
* b
* y
* units 1000
* use_bias True
*/
TVM_REGISTER_GLOBAL("tvm.runtime.cvm.dense").set_body([](TVMArgs args, TVMRetValue* rv) {
  int ndim = args.num_args;
  DLTensor *x = args[0];
  DLTensor *w = args[1];
  DLTensor *b = nullptr;
  DLTensor *y = nullptr;
  int32_t* db = nullptr;
  if(ndim == 6){
	b = args[2];
	y = args[3];
    db = static_cast<int32_t*>(b->data);
  }else{
	y = args[2];
  }
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
		  if(db != nullptr){
			  sum += db[oi];
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

/*
input
weight
bias
output
groups 1
dilation (1, 1)
channels 512
layout NCHW
kernel_layout OIHW
kernel_size [1, 1]
padding (0, 0)
use_bias True
strides (1, 1)
*/
TVM_REGISTER_GLOBAL("tvm.runtime.cvm.conv2d").set_body([]
 (TVMArgs args, TVMRetValue* rv){
    DLTensor *x = args[0];
    DLTensor *w = args[1];
	DLTensor *b = args[2];
    DLTensor *y = args[3];
    auto time_start = clock();
	std::string groups_str = args[4];
	std::string dilation_str = args[5];
	std::string channels_str = args[6];
	std::string layout_str = args[7];
	std::string kernel_layout_str = args[8];
	std::string kernel_size_str = args[9];
	std::string padding_str = args[10];
	std::string use_bias_str = args[11];
	std::string strides_str = args[12];
	int groups = std::atoi(groups_str.c_str());
	int dilation[2] = {0};
	parseToIntPair(dilation_str, dilation);
	int channels = std::atoi(channels_str.c_str());
	int kernel_size[2] = {0};
	parseToIntPair(kernel_size_str, kernel_size);
	int padding[2] = {0};
	parseToIntPair(padding_str, padding);
	int strides[2] = {0};
	parseToIntPair(strides_str, strides);

    int stride_h = strides[0];
    int stride_w = strides[1];

    int32_t* x_data = (int32_t*)x->data;
    int32_t* w_data = (int32_t*)w->data;
    int32_t* y_data = (int32_t*)y->data;
	int32_t* b_data = (int32_t*)b->data;

    int out_channels = static_cast<int>(w->shape[0]);
    int filter_h = static_cast<int>(w->shape[2]);
    int filter_w = static_cast<int>(w->shape[3]);
	filter_h = (filter_h - 1) * dilation[0] + 1;
	filter_w = (filter_w - 1) * dilation[1] + 1;

    int n_batch = static_cast<int>(x->shape[0]);
    int in_channels = static_cast<int>(x->shape[1]);
    int x_h = static_cast<int>(x->shape[2]);
    int x_w = static_cast<int>(x->shape[3]);
//	int o_h = (x_h + 2 * padding[0] - filter_h) / strides[0] + 1;
//	int o_w = (x_w + 2 * padding[1] - filter_w) / strides[1] + 1;
	int o_h = static_cast<int>(y->shape[2]);
	int o_w = static_cast<int>(y->shape[3]);
//	std::cout << o_h << " " << o_w << " "
//              << (x_h + 2 * padding[0] - filter_h) / strides[0] + 1 << " "
//              << (x_w + 2 * padding[1] - filter_w) / strides[1] + 1 << "\n";
//    std::cout << "dim = " << b->ndim << " shape = " << b->shape[0] << "\n";
//    std::cout << "padding = " << padding[0] << " " << padding[1] << "\n";

    const int y_n_offset = out_channels * o_h * o_w;
    const int y_c_offset = o_h * o_w;
    const int y_h_offset = o_w;
    const int x_n_offset = in_channels * x_h * x_w;
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
				auto tp = p * stride_h + r - padding[0];
				if (tp < 0 || tp >= x_h)
                    continue;
				auto tq_start = q * stride_w - padding[1];
				auto tq_end = q * stride_w - padding[1] + filter_w;
				for (auto tq = std::max(tq_start, 0); tq < std::min(tq_end, x_h); ++tq) {
                    auto s = tq - tq_start;
                    y_sum += CONV2d_X(n, c, tp, tq) * CONV2d_W(k, c, r, s);
				}
			}
		}
		return y_sum;

	};
	auto calc_func1x1 = [&](int n, int k, int p, int q, int r = 0, int s = 0) {
		int y_sum = 0;
		for (int c = 0; c < in_channels; ++c) {
            y_sum += CONV2d_X(n, c, p, q) * CONV2d_W(k, c, r, s);
		}
		return y_sum;
	};
    if (filter_w == 1) {
        for (int n = 0; n < n_batch; ++n) {
            for (int k = 0; k < out_channels; ++k) {
                for (int p = 0; p < o_h; ++p) {
                    auto tp = p * stride_h - padding[0];
                    if (tp < 0 || tp >= x_h)
                        continue;
                    for (int q = 0; q < o_w; ++q) {
                        auto tq = q * stride_w - padding[1];
                        if (tq < 0 || tq >= x_w)
                            continue;
                        CONV2d_Y(n, k, p, q) = b_data[k] + calc_func1x1(n, k, tp, tq);
                    }
                }
            }
        }
    } else if (filter_w == 3) {
        std::vector<int32_t> y_sum(in_channels * o_h * o_w, 0);
        std::cout << "buff " << y_sum.size() << "\n";
        for (int n = 0; n < n_batch; ++n) {
            for (int k = 0; k < out_channels; ++k) {
                std::fill(y_sum.begin(), y_sum.end(), 0);
                for (int c = 0; c < in_channels; ++c) {
                    auto conv2d_w_kc = w_data + (k) * w_o_offset + (c) * w_i_offset;
                    for (int p = 0; p < o_h; ++p) {
                        for (int q = 0; q < o_w; ++q) {
                            const int y_idx = c * o_w * o_h + p * o_w + q;
                            auto tq_start = q * stride_w - padding[1];
                            auto tq_begin = std::max(tq_start, 0);
                            auto tq_end = std::min(q * stride_w - padding[1] + filter_w, x_w);
                            {
                                int r = 0;
                                auto tp = p * stride_h + r - padding[0];
                                if (tp >= 0) {
                                    if (tp >= x_h)
                                        continue;
                                    for (auto tq = tq_begin; tq < tq_end; ++tq) {
                                        auto s = tq - tq_start;
                                        y_sum[y_idx] += CONV2d_X(n, c, tp, tq) * conv2d_w_kc[r * filter_h + s];
                                    }
                                }
                            }
                            {
                                int r = 1;
                                auto tp = p * stride_h + r - padding[0];
                                if (tp >= 0) {
                                    if (tp >= x_h)
                                        continue;
                                    for (auto tq = tq_begin; tq < tq_end; ++tq) {
                                        auto s = tq - tq_start;
                                        y_sum[y_idx] += CONV2d_X(n, c, tp, tq) * conv2d_w_kc[r * filter_h + s];
                                    }
                                }
                            }
                            {
                                int r = 2;
                                auto tp = p * stride_h + r - padding[0];
                                if (tp >= 0) {
                                    if (tp >= x_h)
                                        continue;
                                    for (auto tq = tq_begin; tq < tq_end; ++tq) {
                                        auto s = tq - tq_start;
                                        y_sum[y_idx] += CONV2d_X(n, c, tp, tq) * conv2d_w_kc[r * filter_h + s];
                                    }
                                }
                            }
                        }
                    }
                }
                for (int p = 0; p < o_h; ++p) {
                    for (int q = 0; q < o_w; ++q) {
                        uint32_t tmp = 0;
                        for (int c = 0; c < in_channels; ++c) {
                            tmp += y_sum[c * o_h * o_w + p * o_h + q];
                        }
                        CONV2d_Y(n, k, p, q) = b_data[k] + tmp;
                    }
                }
            }
        }
    } else {
        for (int n = 0; n < n_batch; ++n) {
            for (int k = 0; k < out_channels; ++k) {
                for (int p = 0; p < o_h; ++p) {
                    for (int q = 0; q < o_w; ++q) {
                        CONV2d_Y(n, k, p, q) = b_data[k] + calc_func(n, k, p, q);
                    }
                }
            }
        }
    }
    std::cout << o_h << " " << o_w << " (" << filter_h << "," << " " << filter_w << ")"
              << in_channels << " " << out_channels << " "
              << (clock() - time_start + .0) / CLOCKS_PER_SEC << "\n";
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
        DLTensor *args0 = args[0];
        DLTensor *args1 = args[1];
        DLTensor *args2 = args[2];
        int32_t *a = static_cast<int32_t*>(args0->data);
        int32_t *b = static_cast<int32_t*>(args1->data);
        int32_t *c = static_cast<int32_t*>(args2->data);

        for(int i = 0; i < getSize(args0); i++){
			if(args1->ndim > 1)
	            c[i] = a[i] + b[i];
			else c[i] = a[i] + b[0];
        }
    });

TVM_REGISTER_GLOBAL("tvm.runtime.cvm.broadcast_sub")
    .set_body([](TVMArgs args, TVMRetValue *ret){
        DLTensor *args0 = args[0];
        DLTensor *args1 = args[1];
        DLTensor *args2 = args[2];
        int32_t *a = static_cast<int32_t*>(args0->data);
        int32_t *b = static_cast<int32_t*>(args1->data);
        int32_t *c = static_cast<int32_t*>(args2->data);

        for(int i = 0; i < getSize(args0); i++){
			if(args1->ndim > 1)
            	c[i] = a[i] - b[i];
			else c[i] = a[i] - b[0];
        }
    });
TVM_REGISTER_GLOBAL("tvm.runtime.cvm.broadcast_mul")
    .set_body([](TVMArgs args, TVMRetValue *ret){
        DLTensor *args0 = args[0];
        DLTensor *args1 = args[1];
        DLTensor *args2 = args[2];
        int32_t *a = static_cast<int32_t*>(args0->data);
        int32_t *b = static_cast<int32_t*>(args1->data);
        int32_t *c = static_cast<int32_t*>(args2->data);

        for(int i = 0; i < getSize(args0); i++){
			if(args1->ndim > 1)
            c[i] = a[i] * b[i];
			else c[i] = a[i] * b[0];
        }
    });
TVM_REGISTER_GLOBAL("tvm.runtime.cvm.broadcast_div")
    .set_body([](TVMArgs args, TVMRetValue *ret){
        DLTensor *args0 = args[0];
        DLTensor *args1 = args[1];
        DLTensor *args2 = args[2];
        int32_t *a = static_cast<int32_t*>(args0->data);
        int32_t *b = static_cast<int32_t*>(args1->data);
        int32_t *c = static_cast<int32_t*>(args2->data);

        for(int i = 0; i < getSize(args0); i++){
			if(args1->ndim > 1)
            c[i] = a[i] / b[i];
			else c[i] = a[i]/b[0];
        }
    });
TVM_REGISTER_GLOBAL("tvm.runtime.cvm.broadcast_right_shift")
    .set_body([](TVMArgs args, TVMRetValue *ret){
        DLTensor *args0 = args[0];
        DLTensor *args1 = args[1];
        DLTensor *args2 = args[2];
        int32_t *a = static_cast<int32_t*>(args0->data);
        int32_t *b = static_cast<int32_t*>(args1->data);
        int32_t *c = static_cast<int32_t*>(args2->data);

        for(int i = 0; i < getSize(args0); i++){
			if( args1->ndim > 1){
				int32_t rightA = ((a[i] >> (b[i] - 1)) + 1) >> 1;
				rightA = (rightA < 127 ? rightA : 127);
				rightA = (rightA > -127 ? rightA : -127);
				c[i] = rightA;
			}else{
			//	int32_t rightA = ((a[i] >> (b[0] - 1)) + 1) >> 1;
			//	rightA = (rightA < 127 ? rightA : 127);
			//	rightA = (rightA > -127 ? rightA : -127);
			//	c[i] = rightA;
				c[i] = a[i] >> b[0];
			}
        }
    });
TVM_REGISTER_GLOBAL("tvm.runtime.cvm.broadcast_left_shift")
    .set_body([](TVMArgs args, TVMRetValue *ret){
        DLTensor *args0 = args[0];
        DLTensor *args1 = args[1];
        DLTensor *args2 = args[2];
        int32_t *a = static_cast<int32_t*>(args0->data);
        int32_t *b = static_cast<int32_t*>(args1->data);
        int32_t *c = static_cast<int32_t*>(args2->data);

        for(int i = 0; i < getSize(args0); i++){
			if(args1->ndim > 1){
				int32_t clipA = a[i] < 127 ? a[i] : 127;
				clipA = clipA > -127 ? clipA : -127;
				int32_t leftA = clipA << b[i];
				leftA = leftA < 127 ? leftA : 127;
				leftA = leftA > -127 ? leftA : -127;
				c[i] = leftA;
			}else{
			//	int32_t clipA = a[i] < 127 ? a[i] : 127;
			//	clipA = clipA > -127 ? clipA : -127;
			//	int32_t leftA = clipA << b[0];
			//	leftA = leftA < 127 ? leftA : 127;
			//	leftA = leftA > -127 ? leftA : -127;
			//	c[i] = leftA;
				c[i] = a[i] << b[0];
			}
        }
    });

/*
* strides (2, 2)
* pool_size [3, 3]
* ceil_mode False
* padding (1, 1)
*/
TVM_REGISTER_GLOBAL("tvm.runtime.cvm.max_pool2d")
    .set_body([](TVMArgs args, TVMRetValue *ret){
	DLTensor *x = args[0];
	DLTensor *y = args[1];
	std::string strides_str = args[2];
	std::string pool_size_str = args[3];
	std::string ceil_mode = args[4];
	std::string padding_str = args[5];
	int strides[2] = {0};
	parseToIntPair(strides_str, strides);
	int pool_size[2] = {0};
	parseToIntPair(pool_size_str, pool_size);
	int padding[2] = {0};
	parseToIntPair(padding_str, padding);

    int stride_h = strides[0];
    int stride_w = strides[1];

    int32_t* x_data = (int32_t*)x->data;
    int32_t* y_data = (int32_t*)y->data;

    int filter_h = pool_size[0];
    int filter_w = pool_size[1];

    int n_batch = static_cast<int>(x->shape[0]);
    int in_channels = static_cast<int>(x->shape[1]);
	int out_channels = in_channels;
    int x_h = static_cast<int>(x->shape[2]);
    int x_w = static_cast<int>(x->shape[3]);
//	int o_h = (x_h + 2 * padding[0] - filter_h) / strides[0] + 1;
//	int o_w = (x_w + 2 * padding[1] - filter_w) / strides[1] + 1;
	int o_h = static_cast<int>(y->shape[2]);
	int o_w = static_cast<int>(y->shape[3]);

	#define GETX(n, c, h, w) x_data[(n) * in_channels * x_h * x_w + (c) * x_h * x_w + (h) * x_w + (w)]
	#define GETW(o, i, h, w) w_data[(o) * in_channels * filter_h * filter_w + (i) * filter_h * filter_w + (h) * filter_w + (w)]
	#define GETY(n, c, h, w) y_data[(n) * out_channels * o_h * o_w + (c) * o_h * o_w + (h) * o_w + (w)]
	auto calc_func = [&](int n, int k, int p, int q) {
		int y_sum = -(2<<31)-1;
		for (int r = 0; r < filter_h; ++r) {
			for (int s = 0; s < filter_w; ++s) {
				auto tp = p * stride_h + r - padding[0];
				auto tq = q * stride_w + s - padding[1];
				int32_t x_tmp = 0;
				if (!(tp < 0 || tq < 0 || tp >= x_h || tq >= x_w))
					x_tmp = GETX(n, k, tp, tq);
				y_sum = std::max(x_tmp, y_sum);
			}
		}
		return y_sum;

	};
    for (int n = 0; n < n_batch; ++n) {
        for (int k = 0; k < out_channels; ++k) {
            for (int p = 0; p < o_h; ++p) {
                for (int q = 0; q < o_w; ++q) {
                    GETY(n, k, p, q) = calc_func(n, k, p, q);
                }
            }
        }
    }
});

/*
* axis (2, 3)
*/
TVM_REGISTER_GLOBAL("tvm.runtime.cvm.sum")
    .set_body([](TVMArgs args, TVMRetValue *ret){
		DLTensor *x = args[0];
		DLTensor *y = args[1];
		std::string axis_str = args[2];
		int axis[2] = {0};
		parseToIntPair(axis_str, axis);

		int32_t *x_data = static_cast<int32_t*>(x->data);
		int32_t *y_data = static_cast<int32_t*>(y->data);
		int n_batch = static_cast<int>(x->shape[0]);
		int channels = static_cast<int>(x->shape[1]);
		int x_h = static_cast<int>(x->shape[2]);
		int x_w = static_cast<int>(x->shape[3]);
		for(int i = 0; i < n_batch; i++){
			for(int j = 0; j < channels; j++){
				int32_t sum = 0;
				for(int h = 0; h < x_h; h++){
					for(int w = 0; w < x_w; w++){
						sum += x_data[i * channels * x_h * x_w + j * x_h * x_w + h * x_w + w];
					}
				}
				y_data[i*channels + j] = sum;
			}
		}
    });

TVM_REGISTER_GLOBAL("tvm.runtime.cvm.elemwise_add")
    .set_body([](TVMArgs args, TVMRetValue *ret){
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
TVM_REGISTER_GLOBAL("tvm.runtime.cvm.reshap")
    .set_body([](TVMArgs args, TVMRetValue *ret){
         DLTensor *x = args[0];
		 DLTensor *y = args[1];
		 if(x->data == y->data) return;
		 std::memcpy(y->data, x->data, getSize(x) * sizeof(int32_t));
    });
}
}
