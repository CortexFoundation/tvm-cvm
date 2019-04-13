#include "cuda_ops.h"
#include <stdio.h>
#include <time.h>

__global__ void kernel_elemwise_add(int32_t *a, int32_t *b, int32_t *c, int32_t n){
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if(i < n)
        c[i] = a[i] + b[i];
}

void cuda_elemwise_add(int32_t *a, int32_t *b, int32_t *c, int32_t n){
    int32_t *dev_a, *dev_b, *dev_c;
    size_t size = sizeof(int32_t) * n;
    cudaMalloc((void**)&dev_a, size);
    cudaMalloc((void**)&dev_b, size);
    cudaMalloc((void**)&dev_c, size);
    cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    kernel_elemwise_add<<<gridSize, blockSize>>>(dev_a, dev_b, dev_c, n);
    cudaDeviceSynchronize();
    cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
}

// n = 1
#define BS 32
template<int F_H, int F_W>
__global__ void kernel_conv2d(
        int32_t *input, int32_t i_n, int32_t i_c, int32_t i_h, int32_t i_w,
        int32_t *filter, int32_t f_n, int32_t f_c, int32_t f_h, int32_t f_w,
        int32_t *bias,
        int32_t padding,
        int32_t stride,
        int32_t dilation,
        int32_t groups,
        int32_t *output, int32_t o_n, int32_t o_c, int32_t o_h, int32_t o_w){
    int g_y = blockDim.y * blockIdx.y + threadIdx.y;
    int g_x = blockDim.x * blockIdx.x + threadIdx.x;
    int l_y = threadIdx.y; 
    int l_x = threadIdx.x;
    int perBlockYOneImage = (o_h+BS-1) / BS;
    int perBlockXOneImage = (o_w+BS-1) / BS;
    int l_o_c = blockIdx.y / perBlockYOneImage;
    int l_o_hi = blockIdx.y % perBlockYOneImage;
    int l_o_wi = blockIdx.x % perBlockXOneImage;
    int l_o_h = l_o_hi * BS + l_y;
    int l_o_w = l_o_wi * BS + l_x;
    if(l_o_h >= o_h || g_x >= o_w) return;

    __shared__ int32_t shared_i[BS + F_H - 1][BS + F_W - 1];
    __shared__ int32_t shared_f[F_H][F_W];

    int32_t sum = 0; 
    int min_s_y = (l_o_hi+1) * BS <= o_h ? BS : o_h%BS;
    int min_s_x = (l_o_wi+1) * BS <= o_w ? BS : o_w%BS;

    for(int c = 0; c < i_c; c++){
        //load input to shared
        int l_i_h = l_o_h * stride - padding;
        int i_y = c * i_h + l_i_h;
        int i_x = g_x * stride - padding;
//        if(l_i_h < 0 || i_x < 0 || l_i_h >= i_h || i_x >= i_w)
//            shared_i[l_y][l_x] = 0;
//        else{
        shared_i[l_y][l_x] = input[i_y * i_w + i_x];

        if(l_y < F_H-1){
            printf("%d,%d\n", l_y, min_s_y);
            shared_i[l_y + min_s_y][l_x] = input[(i_y + min_s_y) * i_w + i_x];     
        }
        if(l_x < F_W-1){
            shared_i[l_y][l_x + min_s_x] = input[i_y * i_w + i_x + min_s_x];
        }
        if(l_y < F_H-1 && l_x < F_W-1){
            shared_i[l_y+min_s_y][l_x+min_s_x] = input[(i_y+min_s_y)*i_w + i_x + min_s_x];
        }
//        }
        //load filter to shared;
        if(l_y < F_H && l_x < F_W){
            shared_f[l_y][l_x] = filter[l_o_c * F_H * F_W * f_c + c * F_H * F_W + l_y * F_W + l_x];
        }
        __syncthreads();

        for(int fy = 0; fy < F_H; fy++){
            for(int fx = 0; fx < F_W; fx++){
                sum += shared_i[l_y+fy][l_x+fx] * shared_f[fy][fx];
            }
        } 
        __syncthreads();
    }

    int oi = l_o_c * o_h * o_w + l_o_h * o_w + g_x;
    output[oi] = sum;
}
void cuda_conv2d(
        int32_t *input, int32_t i_n, int32_t i_c, int32_t i_h, int32_t i_w,
        int32_t *filter, int32_t f_n, int32_t f_c, const int32_t f_h, const int32_t f_w,
        int32_t *bias,
        int32_t padding,
        int32_t stride,
        int32_t dilation,
        int32_t groups,
        int32_t *output, int32_t o_n, int32_t o_c, int32_t o_h, int32_t o_w){
    int32_t *dev_i, *dev_f, *dev_o, *dev_b;
    size_t s_i = i_n * i_c * i_h * i_w * sizeof(int32_t);
    size_t s_f = f_n * f_c * f_h * f_w * sizeof(int32_t);
    size_t s_b = o_c * sizeof(int32_t); 
    size_t s_o = o_n * o_c * o_h * o_w * sizeof(int32_t);
    cudaMalloc((void**)&dev_i, s_i);
    cudaMalloc((void**)&dev_f, s_f);
    cudaMalloc((void**)&dev_b, s_b);
    cudaMalloc((void**)&dev_o, s_o);
    cudaMemcpy(dev_i, input, s_i, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_f, filter, s_f, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, bias, s_b, cudaMemcpyHostToDevice);
    
    clock_t start = clock();
    int b_h = 32;
    int b_w = 32;
    int32_t g_h = o_n * o_c * (o_h + b_h - 1) / b_h;
    int32_t g_w = (o_w + b_w - 1) / b_w;
    dim3 bDim(b_w, b_h, 1);
    dim3 gDim(g_w, g_h, 1);
    kernel_conv2d<3, 3><<<gDim, bDim>>>(
            dev_i, i_n, i_c, i_h, i_w,
            dev_f, f_n, f_c, f_h, f_w,
            dev_b, 
            padding, 
            stride,
            dilation,
            groups,
            dev_o, o_n, o_c, o_h, o_w);
    cudaDeviceSynchronize();
    clock_t end = clock();
    printf("gpu cal time: %d\n", end-start);
    cudaMemcpy(output, dev_o, s_o, cudaMemcpyDeviceToHost);
    cudaFree(dev_i);
    cudaFree(dev_f);
    cudaFree(dev_o);
    cudaFree(dev_b);
}
