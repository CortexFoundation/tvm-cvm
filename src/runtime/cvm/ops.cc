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

#include "cuda_ops.h"

namespace tvm {
namespace runtime {

#define DEBUG_OP false
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
 });

 TVM_REGISTER_GLOBAL("tvm.runtime.cvm.relu").set_body([](TVMArgs args, TVMRetValue* rv) {
   DLTensor *x = args[0];
   DLTensor *y = args[1];
   for (uint32_t i = 0; i < x->shape[0]; i++) {
 		static_cast<int32_t*>(y->data)[i] = std::max(static_cast<int32_t*>(x->data)[i], 0);
   }
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

inline int32_t getSize(DLTensor *dlTensor){
    int32_t size = 1;
    for(int i = 0; i < dlTensor->ndim; i++){
        size *= dlTensor->shape[i];
    }
    return size;
}
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
	//int channels = std::atoi(channels_str.c_str());
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
	int o_h = (x_h + 2 * padding[0] - filter_h) / strides[0] + 1;
	int o_w = (x_w + 2 * padding[1] - filter_w) / strides[1] + 1;
//	int o_h = static_cast<int>(y->shape[2]);
//	int o_w = static_cast<int>(y->shape[3]);
//	std::cout << o_h << " " << o_w << " "
//              << (x_h + 2 * padding[0] - filter_h) / strides[0] + 1 << " "
//              << (x_w + 2 * padding[1] - filter_w) / strides[1] + 1 << "\n";
//    std::cout << "dim = " << b->ndim << " shape = " << b->shape[0] << "\n";
//    std::cout << "padding = " << padding[0] << " " << padding[1] << "\n";
	#define GETX(n, c, h, w) x_data[(n) * in_channels * x_h * x_w + (c) * x_h * x_w + (h) * x_w + (w)]
	#define GETW(o, i, h, w) w_data[(o) * in_channels * filter_h * filter_w + (i) * filter_h * filter_w + (h) * filter_w + (w)]
	#define GETY(n, c, h, w) y_data[(n) * out_channels * o_h * o_w + (c) * o_h * o_w + (h) * o_w + (w)]
	auto calc_func = [&](int n, int k, int p, int q) {
		int y_sum = 0;
		for (int c = 0; c < in_channels; ++c) {
			for (int r = 0; r < filter_h; ++r) {
				for (int s = 0; s < filter_w; ++s) {
					auto tp = p * stride_h + r - padding[0];
					auto tq = q * stride_w + s - padding[1];
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
 });


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

TVM_REGISTER_GLOBAL("tvm.runtime.cvm.reshape")
    .set_body([](TVMArgs args, TVMRetValue *ret){
         DLTensor *x = args[0];
		 DLTensor *y = args[1];
		 if(x->data == y->data) return;
		 std::memcpy(y->data, x->data, getSize(x) * sizeof(int32_t));
    });

TVM_REGISTER_GLOBAL("tvm.runtime.cvm_cuda.elemwise_add")
.set_body([](tvm::runtime::TVMArgs args, tvm::runtime::TVMRetValue *rv){
    DLTensor *a = args[0];
    DLTensor *b = args[1];
    DLTensor *c = args[2];
    int32_t *a_data = static_cast<int32_t*>(a->data);
    int32_t *b_data = static_cast<int32_t*>(b->data);
    int32_t *c_data = static_cast<int32_t*>(c->data);
    int32_t n = getSize(a);
    cuda_elemwise_add(a_data, b_data, c_data, n, DEBUG_OP);
});

TVM_REGISTER_GLOBAL("tvm.runtime.cvm_cuda.conv2d")
.set_body([](TVMArgs args, TVMRetValue* rv){
    DLTensor *x = args[0];
    DLTensor *w = args[1];
	DLTensor *b = args[2];
    DLTensor *y = args[3];
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
	//int channels = std::atoi(channels_str.c_str());
	int kernel_size[2] = {0};
	parseToIntPair(kernel_size_str, kernel_size);
	int padding[2] = {0};
	parseToIntPair(padding_str, padding);
	int strides[2] = {0};
	parseToIntPair(strides_str, strides);

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
	int o_h = (x_h + 2 * padding[0] - filter_h) / strides[0] + 1;
	int o_w = (x_w + 2 * padding[1] - filter_w) / strides[1] + 1;
//	int o_h = static_cast<int>(y->shape[2]);
//	int o_w = static_cast<int>(y->shape[3]);

    cuda_conv2d(
            x_data, n_batch, in_channels, x_h, x_w,
            w_data, out_channels, in_channels, filter_h, filter_w,
            b_data,
            padding[0],
            strides[0],
            dilation[0],
            groups,
            y_data, n_batch, out_channels, o_h, o_w, DEBUG_OP);
 });

TVM_REGISTER_GLOBAL("tvm.runtime.cvm.cuda_max_pool2d")
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

    cuda_max_pool(
            x_data, n_batch, in_channels, x_h, x_w,
            filter_h, filter_w,
            padding[0],
            strides[0],
            y_data, n_batch, out_channels, o_h, o_w, DEBUG_OP);
    });
TVM_REGISTER_GLOBAL("tvm.runtime.cvm_cuda.dense")
.set_body([](TVMArgs args, TVMRetValue* rv) {
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

  cuda_dense(
          dx, dw, dy,
          static_cast<int32_t>(x->shape[0]),
          static_cast<int32_t>(x->shape[1]),
          static_cast<int32_t>(y->shape[1]),
          db,
          DEBUG_OP);
});

TVM_REGISTER_GLOBAL("tvm.runtime.cvm_cuda.clip").set_body([](TVMArgs args, TVMRetValue* rv) {
   DLTensor *x = args[0];
   DLTensor *y = args[1];
   std::string min_str = args[2];
   std::string max_str = args[3];
   int min = std::atoi(min_str.c_str());
   int max = std::atoi(max_str.c_str());

    cuda_clip(
            static_cast<int32_t*>(x->data),
            static_cast<int32_t*>(y->data),
            static_cast<int32_t>(x->shape[0]),
            max, min, DEBUG_OP);
 });

 TVM_REGISTER_GLOBAL("tvm.runtime.cvm_cuda.relu").set_body([](TVMArgs args, TVMRetValue* rv) {
   DLTensor *x = args[0];
   DLTensor *y = args[1];
    cuda_relu(
            static_cast<int32_t*>(x->data),
            static_cast<int32_t*>(y->data),
            static_cast<int32_t>(x->shape[0]),
            DEBUG_OP);
 });
TVM_REGISTER_GLOBAL("tvm.runtime.cvm_cuda.flatten").set_body([]
(TVMArgs args, TVMRetValue* rv){
     DLTensor *x = args[0];
     DLTensor *y = args[1];

    cuda_flatten(
            static_cast<int32_t*>(x->data),
            static_cast<int32_t*>(y->data),
            static_cast<int32_t>(x->shape[0]),
            DEBUG_OP);
});
TVM_REGISTER_GLOBAL("tvm.runtime.cvm_cuda.broadcast_add")
    .set_body([](TVMArgs args, TVMRetValue *ret){
        DLTensor *args0 = args[0];
        DLTensor *args1 = args[1];
        DLTensor *args2 = args[2];
        int32_t *a = static_cast<int32_t*>(args0->data);
        int32_t *b = static_cast<int32_t*>(args1->data);
        int32_t *c = static_cast<int32_t*>(args2->data);

        cuda_broadcast_add(a, b, c, getSize(args0), DEBUG_OP);
    });

TVM_REGISTER_GLOBAL("tvm.runtime.cvm_cuda.broadcast_sub")
    .set_body([](TVMArgs args, TVMRetValue *ret){
        DLTensor *args0 = args[0];
        DLTensor *args1 = args[1];
        DLTensor *args2 = args[2];
        int32_t *a = static_cast<int32_t*>(args0->data);
        int32_t *b = static_cast<int32_t*>(args1->data);
        int32_t *c = static_cast<int32_t*>(args2->data);

        cuda_broadcast_sub(a, b, c, getSize(args0), DEBUG_OP);
    });
TVM_REGISTER_GLOBAL("tvm.runtime.cvm_cuda.broadcast_mul")
    .set_body([](TVMArgs args, TVMRetValue *ret){
        DLTensor *args0 = args[0];
        DLTensor *args1 = args[1];
        DLTensor *args2 = args[2];
        int32_t *a = static_cast<int32_t*>(args0->data);
        int32_t *b = static_cast<int32_t*>(args1->data);
        int32_t *c = static_cast<int32_t*>(args2->data);

        cuda_broadcast_mul(a, b, c, getSize(args0), DEBUG_OP);
    });
TVM_REGISTER_GLOBAL("tvm.runtime.cvm_cuda.broadcast_div")
    .set_body([](TVMArgs args, TVMRetValue *ret){
        DLTensor *args0 = args[0];
        DLTensor *args1 = args[1];
        DLTensor *args2 = args[2];
        int32_t *a = static_cast<int32_t*>(args0->data);
        int32_t *b = static_cast<int32_t*>(args1->data);
        int32_t *c = static_cast<int32_t*>(args2->data);

        cuda_broadcast_div(a, b, c, getSize(args0), DEBUG_OP);
    });
TVM_REGISTER_GLOBAL("tvm.runtime.cvm_cuda.broadcast_right_shift")
    .set_body([](TVMArgs args, TVMRetValue *ret){
        DLTensor *args0 = args[0];
        DLTensor *args1 = args[1];
        DLTensor *args2 = args[2];
        int32_t *a = static_cast<int32_t*>(args0->data);
        int32_t *b = static_cast<int32_t*>(args1->data);
        int32_t *c = static_cast<int32_t*>(args2->data);

        cuda_broadcast_right_shift(a, b, c, getSize(args0), DEBUG_OP);
    });
TVM_REGISTER_GLOBAL("tvm.runtime.cvm_cuda.broadcast_left_shift")
    .set_body([](TVMArgs args, TVMRetValue *ret){
        DLTensor *args0 = args[0];
        DLTensor *args1 = args[1];
        DLTensor *args2 = args[2];
        int32_t *a = static_cast<int32_t*>(args0->data);
        int32_t *b = static_cast<int32_t*>(args1->data);
        int32_t *c = static_cast<int32_t*>(args2->data);

        cuda_broadcast_left_shift(a, b, c, getSize(args0), DEBUG_OP);
    });

/*
* strides (2, 2)
* pool_size [3, 3]
* ceil_mode False
* padding (1, 1)
*/
TVM_REGISTER_GLOBAL("tvm.runtime.cvm_cuda.max_pool2d")
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
    cuda_max_pool(
            x_data, n_batch, in_channels, x_h, x_w,
            filter_h, filter_w,
            padding[0],
            strides[0],
            y_data, n_batch, out_channels, o_h, o_w, DEBUG_OP);
});

/*
* axis (2, 3)
*/
TVM_REGISTER_GLOBAL("tvm.runtime.cvm_cuda.sum")
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
        cuda_sum(x_data, n_batch, channels, x_h, x_w, y_data, DEBUG_OP);
    });

TVM_REGISTER_GLOBAL("tvm.runtime.cvm_cuda.reshape")
    .set_body([](TVMArgs args, TVMRetValue *ret){
         DLTensor *x = args[0];
		 DLTensor *y = args[1];
         cuda_reshape(
                 static_cast<int32_t*>(x->data),
                 static_cast<int32_t*>(y->data),
                 getSize(x),
                 DEBUG_OP);
    });

}
}

