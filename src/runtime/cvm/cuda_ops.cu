#include "cuda_ops.h"
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <memory>
#include <string.h>

__global__ void kernel_elemwise_add(int32_t *a, int32_t *b, int32_t *c, int32_t n){
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if(i < n)
        c[i] = a[i] + b[i];
}

void cuda_elemwise_add(int32_t *a, int32_t *b, int32_t *c, int32_t n, bool debug){
    int32_t *dev_a = a, *dev_b = b, *dev_c = c;
    size_t size = sizeof(int32_t) * n;
    if(debug){
        cudaMalloc((void**)&dev_a, size);
        cudaMalloc((void**)&dev_b, size);
        cudaMalloc((void**)&dev_c, size);
        cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);
    }
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    kernel_elemwise_add<<<gridSize, blockSize>>>(dev_a, dev_b, dev_c, n);
//    cudaDeviceSynchronize();
    if(debug){
        cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost);
        cudaFree(dev_a);
        cudaFree(dev_b);
        cudaFree(dev_c);
    }
}

#define BS 32
//template<int F_H, int F_W, int STRIDE>
__global__ void kernel_conv2d(
        int32_t *input, int32_t i_n/*TODO i_n > 1*/, int32_t i_c, int32_t i_h, int32_t i_w,
        int32_t *filter, int32_t f_n, int32_t f_c, int32_t f_h, int32_t f_w,
        int32_t *bias,
        int32_t padding,
        int32_t stride,
        int32_t dilation, // TODO dilation > 1
        int32_t groups, // TODO groups > 1
        int32_t *output, int32_t o_n, int32_t o_c, int32_t o_h, int32_t o_w){
//    int g_y = blockDim.y * blockIdx.y + threadIdx.y;
    int g_x = blockDim.x * blockIdx.x + threadIdx.x;
    int l_y = threadIdx.y; 
    int l_x = threadIdx.x;
    int tmp_o_h = i_h + 2 * padding - f_h + 1; // for stride
    int tmp_o_w = i_w + 2 * padding - f_w + 1;
    int perBlockYOneImage = (tmp_o_h+BS-1) / BS;
    int perBlockXOneImage = (tmp_o_w+BS-1) / BS;
    int l_o_c = blockIdx.y / perBlockYOneImage;
    int n = l_o_c / o_c;
    int nsize = n * i_c * i_h * i_w; 
    int l_f_c = l_o_c % o_c;
    int l_o_hi = blockIdx.y % perBlockYOneImage;
    int l_o_wi = blockIdx.x % perBlockXOneImage;
    int l_o_h = l_o_hi * BS + l_y;
//    int l_o_w = l_o_wi * BS + l_x;
    if(l_o_h >= tmp_o_h || g_x >= tmp_o_w) return;

    const int32_t F_H = f_h;
    const int32_t F_W = f_w;
//    __shared__ int32_t shared_i[BS + F_H - 1][BS + F_W - 1];
    int32_t sih = BS + F_H - 1;
    int32_t siw = BS + F_W - 1;
    extern __shared__ int32_t  share[];
    int32_t *shared_i = (int32_t*)share; 
    int32_t *shared_f = &share[sih * siw];

    int32_t sum = 0; 
    int min_s_y = (l_o_hi+1) * BS <= tmp_o_h ? BS : tmp_o_h%BS;
    int min_s_x = (l_o_wi+1) * BS <= tmp_o_w ? BS : tmp_o_w%BS;

    for(int c = 0; c < i_c; c++){
        //load input to shared
        int l_i_h = l_o_h - padding;
        int i_y = c * i_h + l_i_h;
        int i_x = g_x - padding;
        // 0~2-> -1~1
        if(l_i_h < 0 || i_x < 0 || l_i_h >= i_h || i_x >= i_w)
            shared_i[l_y*siw + l_x] = 0;
        else
            shared_i[l_y*siw + l_x] = input[nsize + i_y * i_w + i_x];

        if(l_y < F_H-1){
            for(int i = l_y; i < F_H-1; i+=min_s_y){
                if(l_i_h+min_s_y+i-l_y < 0 || i_x < 0 || l_i_h+min_s_y+i-l_y >= i_h || i_x >= i_w)
                    shared_i[(i+min_s_y)*siw + l_x] = 0;
                else
                    shared_i[(i + min_s_y)*siw + l_x] = input[nsize + (i_y + min_s_y + i - l_y) * i_w + i_x];     
            }
        }
        if(l_x < F_W-1){
            for(int i = l_x; i < F_W-1; i+= min_s_x){
                if(l_i_h < 0 || i_x+min_s_x+i-l_x < 0 || l_i_h >= i_h || i_x+min_s_x+i-l_x >= i_w)
                    shared_i[l_y * siw + i+min_s_x] = 0;
                else
                    shared_i[l_y * siw + i + min_s_x] = input[nsize + i_y * i_w + i_x + min_s_x + i - l_x];
            }
        }
        if(l_y < F_H-1 && l_x < F_W-1){
            for(int i = l_y; i < F_H-1; i+=min_s_y){
                for(int j = l_x; j < F_W-1; j+=min_s_x){
                    if(l_i_h+min_s_y+i-l_y < 0 || i_x+min_s_x+j-l_x < 0 || l_i_h+min_s_y+i-l_y >= i_h || i_x+min_s_x+j-l_x >= i_w)
                        shared_i[(i+min_s_y) * siw + j+min_s_x] = 0;
                    else
                        shared_i[(i+min_s_y) * siw + j+min_s_x] = input[nsize + (i_y+min_s_y + i-l_y)*i_w + i_x + min_s_x + j - l_x];
                }
            }
        }
        
        //load filter to shared;
        if(l_y < F_H && l_x < F_W){
            for(int i = l_y; i < F_H; i+= min_s_y)
                for(int j = l_x; j < F_W; j+=min_s_x)
                    shared_f[i*F_W + j] = filter[l_f_c * F_H * F_W * f_c + c * F_H * F_W + i * F_W + j];
        }
        __syncthreads();

        for(int fy = 0; fy < F_H; fy++){
            for(int fx = 0; fx < F_W; fx++){
                sum += shared_i[(l_y+fy)*siw + l_x+fx] * shared_f[fy*F_W + fx];
            }
        } 
        __syncthreads();
    }

    if(l_o_h % stride == 0 && g_x % stride == 0){
    //int oi = l_o_c * o_h * o_w + l_o_h * o_w + g_x;
    int oi = l_o_c * o_h * o_w + l_o_h/stride * o_w + g_x/stride;
    output[oi] = sum + bias[l_o_c];
    }
}
void cuda_conv2d(
        int32_t *input, int32_t i_n/*TODO i_n > 1*/, int32_t i_c, int32_t i_h, int32_t i_w,
        int32_t *filter, int32_t f_n, int32_t f_c, const int32_t f_h, const int32_t f_w,
        int32_t *bias,
        int32_t padding,
        int32_t stride,
        int32_t dilation, //TODO dilation > 1
        int32_t groups, //TODO groups > 1
        int32_t *output, int32_t o_n, int32_t o_c, int32_t o_h, int32_t o_w, bool debug){
    int32_t *dev_i = input, *dev_f = filter, *dev_o = output, *dev_b = bias;
    size_t s_i = i_n * i_c * i_h * i_w * sizeof(int32_t);
    size_t s_f = f_n * f_c * f_h * f_w * sizeof(int32_t);
    size_t s_b = o_c * sizeof(int32_t); 
    size_t s_o = o_n * o_c * o_h * o_w * sizeof(int32_t);
    if(debug){
        cudaMalloc((void**)&dev_i, s_i);
        cudaMalloc((void**)&dev_f, s_f);
        cudaMalloc((void**)&dev_b, s_b);
        cudaMalloc((void**)&dev_o, s_o);
        cudaMemcpy(dev_i, input, s_i, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_f, filter, s_f, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_b, bias, s_b, cudaMemcpyHostToDevice);
    }
//    clock_t start = clock();
    int b_h = BS;
    int b_w = BS;
    int tmp_o_h = i_h + 2 * padding - f_h + 1; //for stride > 1
    int tmp_o_w = i_w + 2 * padding - f_w + 1;
    int32_t g_h = o_n * o_c * ((tmp_o_h + b_h - 1) / b_h);
    int32_t g_w = (tmp_o_w + b_w - 1) / b_w;
    dim3 bDim(b_w, b_h, 1);
    dim3 gDim(g_w, g_h, 1);
    size_t share_size = (BS + f_h - 1) * (BS + f_w - 1) * sizeof(int32_t) + f_h * f_w * sizeof(int32_t);
    kernel_conv2d<<<gDim, bDim, share_size>>>(
            dev_i, i_n, i_c, i_h, i_w,
            dev_f, f_n, f_c, f_h, f_w,
            dev_b, 
            padding, 
            stride,
            dilation,
            groups,
            dev_o, o_n, o_c, o_h, o_w);
//    cudaDeviceSynchronize();
//    clock_t end = clock();
//    printf("gpu cal time: %d\n", end-start);
    if(debug){
        cudaMemcpy(output, dev_o, s_o, cudaMemcpyDeviceToHost);
        cudaFree(dev_i);
        cudaFree(dev_f);
        cudaFree(dev_o);
        cudaFree(dev_b);
    }
}
__global__ void kernel_depthwise_conv2d(
        int32_t *input, int32_t i_n/*TODO i_n > 1*/, int32_t i_c, int32_t i_h, int32_t i_w,
        int32_t *filter, int32_t f_n, int32_t f_c, int32_t f_h, int32_t f_w,
        int32_t *bias,
        int32_t padding,
        int32_t stride,
        int32_t dilation, // TODO dilation > 1
        int32_t groups, // TODO groups > 1
        int32_t *output, int32_t o_n, int32_t o_c, int32_t o_h, int32_t o_w){
    //    int g_y = blockDim.y * blockIdx.y + threadIdx.y;
    int g_x = blockDim.x * blockIdx.x + threadIdx.x;
    int l_y = threadIdx.y; 
    int l_x = threadIdx.x;
    int tmp_o_h = i_h + 2 * padding - f_h + 1; // for stride
    int tmp_o_w = i_w + 2 * padding - f_w + 1;
    int perBlockYOneImage = (tmp_o_h+BS-1) / BS;
    int perBlockXOneImage = (tmp_o_w+BS-1) / BS;
    int l_o_c = blockIdx.y / perBlockYOneImage;
    int l_o_hi = blockIdx.y % perBlockYOneImage;
    int l_o_wi = blockIdx.x % perBlockXOneImage;
    int l_o_h = l_o_hi * BS + l_y;
    //    int l_o_w = l_o_wi * BS + l_x;
    if(l_o_h >= tmp_o_h || g_x >= tmp_o_w) return;

    const int32_t F_H = f_h;
    const int32_t F_W = f_w;
    //    __shared__ int32_t shared_i[BS + F_H - 1][BS + F_W - 1];
    int32_t sih = BS + F_H - 1;
    int32_t siw = BS + F_W - 1;
    extern __shared__ int32_t  share[];
    int32_t *shared_i = (int32_t*)share; 
    int32_t *shared_f = &share[sih * siw];

    int32_t sum = 0; 
    int min_s_y = (l_o_hi+1) * BS <= tmp_o_h ? BS : tmp_o_h%BS;
    int min_s_x = (l_o_wi+1) * BS <= tmp_o_w ? BS : tmp_o_w%BS;

    //load input to shared
    int l_i_h = l_o_h - padding;
    int i_y = l_o_c * i_h + l_i_h;
    int i_x = g_x - padding;
    // 0~2-> -1~1
    if(l_i_h < 0 || i_x < 0 || l_i_h >= i_h || i_x >= i_w)
        shared_i[l_y*siw + l_x] = 0;
    else
        shared_i[l_y*siw + l_x] = input[i_y * i_w + i_x];

    if(l_y < F_H-1){
        for(int i = l_y; i < F_H-1; i+=min_s_y){
            if(l_i_h+min_s_y+i-l_y < 0 || i_x < 0 || l_i_h+min_s_y+i-l_y >= i_h || i_x >= i_w)
                shared_i[(i+min_s_y)*siw + l_x] = 0;
            else
                shared_i[(i + min_s_y)*siw + l_x] = input[(i_y + min_s_y + i - l_y) * i_w + i_x];     
        }
    }
    if(l_x < F_W-1){
        for(int i = l_x; i < F_W-1; i+= min_s_x){
            if(l_i_h < 0 || i_x+min_s_x+i-l_x < 0 || l_i_h >= i_h || i_x+min_s_x+i-l_x >= i_w)
                shared_i[l_y * siw + i+min_s_x] = 0;
            else
                shared_i[l_y * siw + i + min_s_x] = input[i_y * i_w + i_x + min_s_x + i - l_x];
        }
    }
    if(l_y < F_H-1 && l_x < F_W-1){
        for(int i = l_y; i < F_H-1; i+=min_s_y){
            for(int j = l_x; j < F_W-1; j+=min_s_x){
                if(l_i_h+min_s_y+i-l_y < 0 || i_x+min_s_x+j-l_x < 0 || l_i_h+min_s_y+i-l_y >= i_h || i_x+min_s_x+j-l_x >= i_w)
                    shared_i[(i+min_s_y) * siw + j+min_s_x] = 0;
                else
                    shared_i[(i+min_s_y) * siw + j+min_s_x] = input[(i_y+min_s_y + i-l_y)*i_w + i_x + min_s_x + j - l_x];
            }
        }
    }

    //load filter to shared;
    if(l_y < F_H && l_x < F_W){
        for(int i = l_y; i < F_H; i+= min_s_y)
            for(int j = l_x; j < F_W; j+=min_s_x)
                shared_f[i*F_W + j] = filter[l_o_c * F_H * F_W + i * F_W + j];
    }
    __syncthreads();

    for(int fy = 0; fy < F_H; fy++){
        for(int fx = 0; fx < F_W; fx++){
            sum += shared_i[(l_y+fy)*siw + l_x+fx] * shared_f[fy*F_W + fx];
        }
    } 
    __syncthreads();

    if(l_o_h % stride == 0 && g_x % stride == 0){
        //int oi = l_o_c * o_h * o_w + l_o_h * o_w + g_x;
        int oi = l_o_c * o_h * o_w + l_o_h/stride * o_w + g_x/stride;
        output[oi] = sum + bias[l_o_c];
    }
}
void cuda_depthwise_conv2d(
        int32_t *input, int32_t i_n, int32_t i_c, int32_t i_h, int32_t i_w,
        int32_t *filter, int32_t f_n, int32_t f_c, int32_t f_h, int32_t f_w,
        int32_t *bias,
        int32_t padding,
        int32_t stride,
        int32_t dilation,
        int32_t groups,
        int32_t *output, int32_t o_n, int32_t o_c, int32_t o_h, int32_t o_w, bool debug){
    int32_t *dev_i = input, *dev_f = filter, *dev_o = output, *dev_b = bias;
    size_t s_i = i_n * i_c * i_h * i_w * sizeof(int32_t);
    size_t s_f = f_n * f_c * f_h * f_w * sizeof(int32_t);
    size_t s_b = o_c * sizeof(int32_t); 
    size_t s_o = o_n * o_c * o_h * o_w * sizeof(int32_t);
    if(debug){
        cudaMalloc((void**)&dev_i, s_i);
        cudaMalloc((void**)&dev_f, s_f);
        cudaMalloc((void**)&dev_b, s_b);
        cudaMalloc((void**)&dev_o, s_o);
        cudaMemcpy(dev_i, input, s_i, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_f, filter, s_f, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_b, bias, s_b, cudaMemcpyHostToDevice);
    }
//    clock_t start = clock();
    int b_h = BS;
    int b_w = BS;
    int tmp_o_h = i_h + 2 * padding - f_h + 1; //for stride > 1
    int tmp_o_w = i_w + 2 * padding - f_w + 1;
    int32_t g_h = o_n * o_c * ((tmp_o_h + b_h - 1) / b_h);
    int32_t g_w = (tmp_o_w + b_w - 1) / b_w;
    dim3 bDim(b_w, b_h, 1);
    dim3 gDim(g_w, g_h, 1);
    size_t share_size = (BS + f_h - 1) * (BS + f_w - 1) * sizeof(int32_t) + f_h * f_w * sizeof(int32_t);
    kernel_depthwise_conv2d<<<gDim, bDim, share_size>>>(
            dev_i, i_n, i_c, i_h, i_w,
            dev_f, f_n, f_c, f_h, f_w,
            dev_b, 
            padding, 
            stride,
            dilation,
            groups,
            dev_o, o_n, o_c, o_h, o_w);
    //cudaDeviceSynchronize();
//    clock_t end = clock();
//    printf("gpu cal time: %d\n", end-start);
    if(debug){
        cudaMemcpy(output, dev_o, s_o, cudaMemcpyDeviceToHost);
        cudaFree(dev_i);
        cudaFree(dev_f);
        cudaFree(dev_o);
        cudaFree(dev_b);
    }
}

__global__ void kernel_max_pool(
        int32_t *input, int32_t i_n/*TODO i_n > 1*/, int32_t i_c, int32_t i_h, int32_t i_w,
        int32_t f_h, int32_t f_w,
        int32_t padding,
        int32_t stride,
        int32_t *output, int32_t o_n, int32_t o_c, int32_t o_h, int32_t o_w){
//    int g_y = blockDim.y * blockIdx.y + threadIdx.y;
    int g_x = blockDim.x * blockIdx.x + threadIdx.x;
    int l_y = threadIdx.y; 
    int l_x = threadIdx.x;
    int tmp_o_h = i_h + 2 * padding - f_h + 1; // for stride
    int tmp_o_w = i_w + 2 * padding - f_w + 1;
    int perBlockYOneImage = (tmp_o_h+BS-1) / BS;
    int perBlockXOneImage = (tmp_o_w+BS-1) / BS;
    int l_o_c = blockIdx.y / perBlockYOneImage;
    int l_o_hi = blockIdx.y % perBlockYOneImage;
    int l_o_wi = blockIdx.x % perBlockXOneImage;
    int l_o_h = l_o_hi * BS + l_y;
//    int l_o_w = l_o_wi * BS + l_x;
    if(l_o_h >= tmp_o_h || g_x >= tmp_o_w) return;

    const int32_t F_H = f_h;
    const int32_t F_W = f_w;
//    __shared__ int32_t shared_i[BS + F_H - 1][BS + F_W - 1];
//    int32_t sih = BS + F_H - 1;
    int32_t siw = BS + F_W - 1;
    extern __shared__ int32_t  share[];
    int32_t *shared_i = (int32_t*)share; 

    int32_t max_elem = -(2<<31)-1; 
    int min_s_y = (l_o_hi+1) * BS <= tmp_o_h ? BS : tmp_o_h%BS;
    int min_s_x = (l_o_wi+1) * BS <= tmp_o_w ? BS : tmp_o_w%BS;

    //load input to shared
    int l_i_h = l_o_h - padding;
    int i_y = l_o_c * i_h + l_i_h;
    int i_x = g_x - padding;
    // 0~2-> -1~1
    if(l_i_h < 0 || i_x < 0 || l_i_h >= i_h || i_x >= i_w)
        shared_i[l_y*siw + l_x] = 0;
    else
        shared_i[l_y*siw + l_x] = input[i_y * i_w + i_x];

    if(l_y < F_H-1){
        for(int i = l_y; i < F_H-1; i+=min_s_y){
            if(l_i_h+min_s_y+i-l_y < 0 || i_x < 0 || l_i_h+min_s_y+i-l_y >= i_h || i_x >= i_w)
                shared_i[(i+min_s_y)*siw + l_x] = 0;
            else
                shared_i[(i + min_s_y)*siw + l_x] = input[(i_y + min_s_y + i - l_y) * i_w + i_x];     
        }
    }
    if(l_x < F_W-1){
        for(int i = l_x; i < F_W-1; i+= min_s_x){
            if(l_i_h < 0 || i_x+min_s_x+i-l_x < 0 || l_i_h >= i_h || i_x+min_s_x+i-l_x >= i_w)
                shared_i[l_y * siw + i+min_s_x] = 0;
            else
                shared_i[l_y * siw + i + min_s_x] = input[i_y * i_w + i_x + min_s_x + i - l_x];
        }
    }
    if(l_y < F_H-1 && l_x < F_W-1){
        for(int i = l_y; i < F_H-1; i+=min_s_y){
            for(int j = l_x; j < F_W-1; j+=min_s_x){
                if(l_i_h+min_s_y+i-l_y < 0 || i_x+min_s_x+j-l_x < 0 || l_i_h+min_s_y+i-l_y >= i_h || i_x+min_s_x+j-l_x >= i_w)
                    shared_i[(i+min_s_y) * siw + j+min_s_x] = 0;
                else
                    shared_i[(i+min_s_y) * siw + j+min_s_x] = input[(i_y+min_s_y + i-l_y)*i_w + i_x + min_s_x + j - l_x];
            }
        }
    }
    __syncthreads();

    for(int fy = 0; fy < F_H; fy++){
        for(int fx = 0; fx < F_W; fx++){
            int32_t tmp =  shared_i[(l_y+fy)*siw + l_x+fx];
            max_elem = max_elem < tmp ? tmp : max_elem;
        }
    } 
    __syncthreads();

    if(l_o_h % stride == 0 && g_x % stride == 0){
        //int oi = l_o_c * o_h * o_w + l_o_h * o_w + g_x;
        int oi = l_o_c * o_h * o_w + l_o_h/stride * o_w + g_x/stride;
        output[oi] = max_elem;
    }
}

void cuda_max_pool(
        int32_t *input, int32_t i_n/*TODO i_n > 1*/, int32_t i_c, int32_t i_h, int32_t i_w,
        const int32_t f_h, const int32_t f_w,
        int32_t padding,
        int32_t stride,
        int32_t *output, int32_t o_n, int32_t o_c, int32_t o_h, int32_t o_w, bool debug){
    int32_t *dev_i = input, *dev_o = output;
    size_t s_i = i_n * i_c * i_h * i_w * sizeof(int32_t);
    size_t s_o = o_n * o_c * o_h * o_w * sizeof(int32_t);
    if(debug){
        cudaMalloc((void**)&dev_i, s_i);
        cudaMalloc((void**)&dev_o, s_o);
        cudaMemcpy(dev_i, input, s_i, cudaMemcpyHostToDevice);
    }
    
//    clock_t start = clock();
    int b_h = BS;
    int b_w = BS;
    int tmp_o_h = i_h + 2 * padding - f_h + 1; //for stride > 1
    int tmp_o_w = i_w + 2 * padding - f_w + 1;
    int32_t g_h = o_n * o_c * ((tmp_o_h + b_h - 1) / b_h);
    int32_t g_w = (tmp_o_w + b_w - 1) / b_w;
    dim3 bDim(b_w, b_h, 1);
    dim3 gDim(g_w, g_h, 1);
    size_t share_size = (BS + f_h - 1) * (BS + f_w - 1) * sizeof(int32_t);
    kernel_max_pool<<<gDim, bDim, share_size>>>(
            dev_i, i_n, i_c, i_h, i_w,
            f_h, f_w,
            padding, 
            stride,
            dev_o, o_n, o_c, o_h, o_w);
    //cudaDeviceSynchronize();
//    clock_t end = clock();
//    printf("gpu cal time: %ld\n", end-start);
    if(debug){
        cudaMemcpy(output, dev_o, s_o, cudaMemcpyDeviceToHost);
        cudaFree(dev_i);
        cudaFree(dev_o);
    }
}

#define TILE_WIDTH 32
__global__ void kernel_dense(
        int32_t *A, // m*k 
        int32_t *B, // was transposed, n*k
        int32_t *C, // m*n
        int32_t m, int32_t k, int32_t n, int32_t *bias, int32_t useBias){
    __shared__ int32_t sharedM[TILE_WIDTH][TILE_WIDTH];
    __shared__ int32_t sharedN[TILE_WIDTH][TILE_WIDTH];
    int bx = blockIdx.x;//0
    int by = blockIdx.y;//0
    int tx = threadIdx.x;//0~31
    int ty = threadIdx.y;//0~31
    int row = by*TILE_WIDTH + ty;//0
    int col = bx*TILE_WIDTH + tx;//31
    int v = 0;

    for (int i = 0; i < (int)(ceil((float)k/TILE_WIDTH)); i++)
    {
        if (i*TILE_WIDTH + tx < k && row < m)//m*k
            sharedM[ty][tx] = A[row*k + i*TILE_WIDTH + tx];
        else
            sharedM[ty][tx] = 0;

        if(i*TILE_WIDTH + ty < k && col < n)//n*k
            sharedN[tx][ty] = B[col * k + i * TILE_WIDTH + ty];
        else
            sharedN[tx][ty] = 0;
        __syncthreads();

        for(int j = 0; j < TILE_WIDTH; j++)
            v += sharedM[ty][j] * sharedN[tx][j];
        __syncthreads();
    }
    if (row < m && col < n){
        if(useBias == 1) v += bias[col];
        C[row*n + col] = v;
    }
}

void cuda_dense(
        int32_t *a,
        int32_t *b,
        int32_t *c,
        const int m, const int k, const int n, int32_t* bias, bool debug){
    int32_t *dev_a = a, *dev_b = b, *dev_c = c, *dev_bias = bias, useBias = 0;
    size_t s_a = sizeof(int32_t) * m * k;
    size_t s_b = sizeof(int32_t) * k * n;
    size_t s_c = sizeof(int32_t) * m * n;
    size_t s_bias = sizeof(int32_t) * n;
    if(debug){
        cudaMalloc((void**)&dev_a, s_a);
        cudaMalloc((void**)&dev_b, s_b);
        cudaMalloc((void**)&dev_c, s_c);
        if(bias != NULL){
            cudaMalloc((void**)&dev_bias, s_bias);
            cudaMemcpy(dev_bias, bias, s_bias, cudaMemcpyHostToDevice);
            useBias = 1;
        }
        cudaMemcpy(dev_a, a, s_a, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_b, b, s_b, cudaMemcpyHostToDevice);
    }
    if(bias != NULL) useBias = 1;

    dim3 bDim(TILE_WIDTH, TILE_WIDTH, 1);
    int gh = (m + TILE_WIDTH - 1) / TILE_WIDTH;
    int gw = (n + TILE_WIDTH - 1) / TILE_WIDTH;
    dim3 gDim(gw, gh, 1);
    kernel_dense<<<gDim, bDim>>>(dev_a, dev_b, dev_c, m, k, n, dev_bias, useBias);
    //cudaDeviceSynchronize();
    if(debug){
        cudaMemcpy(c, dev_c, s_c, cudaMemcpyDeviceToHost);
        cudaFree(dev_a);
        cudaFree(dev_b);
        cudaFree(dev_c);
        cudaFree(dev_bias);
    }
}

__global__ void kernel_clip(const int32_t *x, int32_t *y,
        const int32_t n, const int32_t maxV, const int32_t minV){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
        y[i] = max(min(x[i], maxV), minV);
    }
}
void cuda_clip(const int32_t *x, int32_t *y, const int32_t n, const int32_t max, const int32_t min, bool debug){
    const int32_t *dev_x = x;
    int32_t *tmp_x;
    int32_t *dev_y = y;
    if(debug) {
        cudaMalloc((void**)&tmp_x, n*sizeof(int32_t));
        dev_x = tmp_x;
        cudaMalloc((void**)&dev_y, n*sizeof(int32_t));
        cudaMemcpy(tmp_x, x, sizeof(int32_t)*n, cudaMemcpyHostToDevice);
    }

    int threadSize = 256;
    int blockSize = (n + threadSize - 1) / threadSize;
    kernel_clip<<<blockSize, threadSize>>>(dev_x, dev_y, n, max, min);
   // cudaDeviceSynchronize();

    if(debug){
        cudaMemcpy(y, dev_y, sizeof(int32_t)*n, cudaMemcpyDeviceToHost);
        cudaFree(tmp_x);
        cudaFree(dev_y);
    }
}

__global__ void kernel_relu(const int32_t *x, int32_t*y, const int32_t n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
        y[i] = max(x[i], 0);
    }
}
void cuda_relu(const int32_t *x, int32_t *y, const int32_t n, bool debug){
    const int32_t *dev_x = x;
    int32_t *tmp_x;
    int32_t *dev_y = y;
    if(debug) {
        cudaMalloc((void**)&tmp_x, n*sizeof(int32_t));
        dev_x = tmp_x;
        cudaMalloc((void**)&dev_y, n*sizeof(int32_t));
        cudaMemcpy(tmp_x, x, sizeof(int32_t)*n, cudaMemcpyHostToDevice);
    }

    int threadSize = 256;
    int blockSize = (n + threadSize - 1) / threadSize;
    kernel_relu<<<blockSize, threadSize>>>(dev_x, dev_y, n);
    //cudaDeviceSynchronize();

    if(debug){
        cudaMemcpy(y, dev_y, sizeof(int32_t)*n, cudaMemcpyDeviceToHost);
        cudaFree(tmp_x);
        cudaFree(dev_y);
    }
}

__global__ void kernel_flatten(const int32_t *x, int32_t*y, const int32_t n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
        y[i] = x[i];
    }
}
void cuda_flatten(const int32_t *x, int32_t *y, const int32_t n, bool debug){
    const int32_t *dev_x = x;
    int32_t *tmp_x;
    int32_t *dev_y = y;
    if(debug) {
        cudaMalloc((void**)&tmp_x, n*sizeof(int32_t));
        dev_x = tmp_x;
        cudaMalloc((void**)&dev_y, n*sizeof(int32_t));
        cudaMemcpy(tmp_x, x, sizeof(int32_t)*n, cudaMemcpyHostToDevice);
    }

    int threadSize = 256;
    int blockSize = (n + threadSize - 1) / threadSize;
    kernel_flatten<<<blockSize, threadSize>>>(dev_x, dev_y, n);
    //cudaDeviceSynchronize();

    if(debug){
        cudaMemcpy(y, dev_y, sizeof(int32_t)*n, cudaMemcpyDeviceToHost);
        cudaFree(tmp_x);
        cudaFree(dev_y);
    }
}

__global__ void kernel_broadcast_add(const int32_t *a, const int32_t *b, int32_t*c, const int32_t n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
        c[i] = a[i] + b[0];
    }
}
void cuda_broadcast_add(const int32_t *a, const int32_t *b, int32_t* c, const int32_t n, bool debug){
    const int32_t *dev_a = a, *dev_b = b;
    int32_t *tmp_a, *tmp_b;
    int32_t *dev_c = c;
    if(debug) {
        cudaMalloc((void**)&tmp_a, n*sizeof(int32_t));
        dev_a = tmp_a;
        cudaMalloc((void**)&tmp_b, sizeof(int32_t));
        dev_b = tmp_b;
        cudaMalloc((void**)&dev_c, n*sizeof(int32_t));
        cudaMemcpy(tmp_a, a, sizeof(int32_t)*n, cudaMemcpyHostToDevice);
        cudaMemcpy(tmp_b, b, sizeof(int32_t), cudaMemcpyHostToDevice);
    }

    int threadSize = 256;
    int blockSize = (n + threadSize - 1) / threadSize;
    kernel_broadcast_add<<<blockSize, threadSize>>>(dev_a, dev_b, dev_c, n);
    //cudaDeviceSynchronize();

    if(debug){
        cudaMemcpy(c, dev_c, sizeof(int32_t)*n, cudaMemcpyDeviceToHost);
        cudaFree(tmp_a);
        cudaFree(dev_c);
        cudaFree(tmp_b);
    }
}
__global__ void kernel_broadcast_sub(const int32_t *a, const int32_t *b, int32_t*c, const int32_t n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
        c[i] = a[i] - b[0];
    }
}
void cuda_broadcast_sub(const int32_t *a, const int32_t *b, int32_t* c, const int32_t n, bool debug){
    const int32_t *dev_a = a, *dev_b = b;
    int32_t *tmp_a, *tmp_b;
    int32_t *dev_c = c;
    if(debug) {
        cudaMalloc((void**)&tmp_a, n*sizeof(int32_t));
        dev_a = tmp_a;
        cudaMalloc((void**)&tmp_b, sizeof(int32_t));
        dev_b = tmp_b;
        cudaMalloc((void**)&dev_c, n*sizeof(int32_t));
        cudaMemcpy(tmp_a, a, sizeof(int32_t)*n, cudaMemcpyHostToDevice);
        cudaMemcpy(tmp_b, b, sizeof(int32_t), cudaMemcpyHostToDevice);
    }

    int threadSize = 256;
    int blockSize = (n + threadSize - 1) / threadSize;
    kernel_broadcast_sub<<<blockSize, threadSize>>>(dev_a, dev_b, dev_c, n);
    //cudaDeviceSynchronize();

    if(debug){
        cudaMemcpy(c, dev_c, sizeof(int32_t)*n, cudaMemcpyDeviceToHost);
        cudaFree(tmp_a);
        cudaFree(dev_c);
        cudaFree(tmp_b);
    }
}
__global__ void kernel_broadcast_mul(const int32_t *a, const int32_t *b, int32_t*c, const int32_t n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
        c[i] = a[i] * b[0];
    }
}
void cuda_broadcast_mul(const int32_t *a, const int32_t *b, int32_t* c, const int32_t n, bool debug){
    const int32_t *dev_a = a, *dev_b = b;
    int32_t *tmp_a, *tmp_b;
    int32_t *dev_c = c;
    if(debug) {
        cudaMalloc((void**)&tmp_a, n*sizeof(int32_t));
        dev_a = tmp_a;
        cudaMalloc((void**)&tmp_b, sizeof(int32_t));
        dev_b = tmp_b;
        cudaMalloc((void**)&dev_c, n*sizeof(int32_t));
        cudaMemcpy(tmp_a, a, sizeof(int32_t)*n, cudaMemcpyHostToDevice);
        cudaMemcpy(tmp_b, b, sizeof(int32_t), cudaMemcpyHostToDevice);
    }

    int threadSize = 256;
    int blockSize = (n + threadSize - 1) / threadSize;
    kernel_broadcast_mul<<<blockSize, threadSize>>>(dev_a, dev_b, dev_c, n);
    //cudaDeviceSynchronize();

    if(debug){
        cudaMemcpy(c, dev_c, sizeof(int32_t)*n, cudaMemcpyDeviceToHost);
        cudaFree(tmp_a);
        cudaFree(dev_c);
        cudaFree(tmp_b);
    }
}
__global__ void kernel_broadcast_div(const int32_t *a, const int32_t *b, int32_t*c, const int32_t n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
        c[i] = a[i] / b[0];
    }
}
void cuda_broadcast_div(const int32_t *a, const int32_t *b, int32_t* c, const int32_t n, bool debug){
    const int32_t *dev_a = a, *dev_b = b;
    int32_t *tmp_a, *tmp_b;
    int32_t *dev_c = c;
    if(debug) {
        cudaMalloc((void**)&tmp_a, n*sizeof(int32_t));
        dev_a = tmp_a;
        cudaMalloc((void**)&tmp_b, sizeof(int32_t));
        dev_b = tmp_b;
        cudaMalloc((void**)&dev_c, n*sizeof(int32_t));
        cudaMemcpy(tmp_a, a, sizeof(int32_t)*n, cudaMemcpyHostToDevice);
        cudaMemcpy(tmp_b, b, sizeof(int32_t), cudaMemcpyHostToDevice);
    }

    int threadSize = 256;
    int blockSize = (n + threadSize - 1) / threadSize;
    kernel_broadcast_div<<<blockSize, threadSize>>>(dev_a, dev_b, dev_c, n);
    //cudaDeviceSynchronize();

    if(debug){
        cudaMemcpy(c, dev_c, sizeof(int32_t)*n, cudaMemcpyDeviceToHost);
        cudaFree(tmp_a);
        cudaFree(dev_c);
        cudaFree(tmp_b);
    }
}
__global__ void kernel_broadcast_right_shift(const int32_t *a, const int32_t *b, int32_t*c, const int32_t n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
        c[i] = a[i] >> b[0];
    }
}
void cuda_broadcast_right_shift(const int32_t *a, const int32_t* b, int32_t* c, const int32_t n, bool debug){
    const int32_t *dev_a = a;
    const int32_t *dev_b = b;
    int32_t *tmp_a, *tmp_b;
    int32_t *dev_c = c;
    if(debug) {
        cudaMalloc((void**)&tmp_a, n*sizeof(int32_t));
        dev_a = tmp_a;
        cudaMalloc((void**)&tmp_b, sizeof(int32_t));
        dev_b = tmp_b;
        cudaMalloc((void**)&dev_c, n*sizeof(int32_t));
        cudaMemcpy(tmp_a, a, sizeof(int32_t)*n, cudaMemcpyHostToDevice);
        cudaMemcpy(tmp_b, b, sizeof(int32_t), cudaMemcpyHostToDevice);
    }

    int threadSize = 256;
    int blockSize = (n + threadSize - 1) / threadSize;
    kernel_broadcast_right_shift<<<blockSize, threadSize>>>(dev_a, dev_b, dev_c, n);
    //cudaDeviceSynchronize();

    if(debug){
        cudaMemcpy(c, dev_c, sizeof(int32_t)*n, cudaMemcpyDeviceToHost);
        cudaFree(tmp_a);
        cudaFree(dev_c);
        cudaFree(tmp_b);
    }
}
__global__ void kernel_broadcast_left_shift(const int32_t *a, const int32_t *b, int32_t*c, const int32_t n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
        c[i] = a[i] << b[0];
    }
}
void cuda_broadcast_left_shift(const int32_t *a, const int32_t *b, int32_t* c, const int32_t n, bool debug){
    const int32_t *dev_a = a, *dev_b = b;
    int32_t *tmp_a, *tmp_b;
    int32_t *dev_c = c;
    if(debug) {
        cudaMalloc((void**)&tmp_a, n*sizeof(int32_t));
        dev_a = tmp_a;
        cudaMalloc((void**)&tmp_b, sizeof(int32_t));
        dev_b = tmp_b;
        cudaMalloc((void**)&dev_c, n*sizeof(int32_t));
        cudaMemcpy(tmp_a, a, sizeof(int32_t)*n, cudaMemcpyHostToDevice);
        cudaMemcpy(tmp_b, b, sizeof(int32_t), cudaMemcpyHostToDevice);
    }

    int threadSize = 256;
    int blockSize = (n + threadSize - 1) / threadSize;
    kernel_broadcast_left_shift<<<blockSize, threadSize>>>(dev_a, dev_b, dev_c, n);
    //cudaDeviceSynchronize();

    if(debug){
        cudaMemcpy(c, dev_c, sizeof(int32_t)*n, cudaMemcpyDeviceToHost);
        cudaFree(tmp_a);
        cudaFree(dev_c);
        cudaFree(tmp_b);
    }
}

//TODO use reduce
__global__ void kernel_sum(const int32_t *x,
        const int n_batch,
        const int channels,
        const int h,
        const int w,
        int32_t *y){
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int n = n_batch * channels;
    if(tid < n){
        int32_t sum = 0;
        for(int i = 0; i < h; i++){
            for(int j = 0; j < w; j++){
                sum += x[tid * h * w + i * w + j];
            }
        }
        y[tid] = sum;
    }
    
}
void cuda_sum(
        const int32_t *x,
        const int32_t n_batch, const int32_t channels, const int32_t h, const int32_t w,
        int32_t *y,
        bool debug){
    const int32_t *dev_x = x;
    int32_t *dev_y = y;
    int32_t *tmp_x;
    if(debug){
        size_t size = n_batch * channels * h * w * sizeof(int32_t);
        cudaMalloc((void**)&tmp_x, size);
        dev_x = tmp_x;
        cudaMalloc((void**)&dev_y, n_batch * channels * sizeof(int32_t));
        cudaMemcpy(tmp_x, x, size, cudaMemcpyHostToDevice);
    }

    int n = n_batch * channels;
    int threadSize = 256;
    int blockSize = (n + threadSize - 1) / threadSize;
    kernel_sum<<<blockSize, threadSize>>>(dev_x, n_batch, channels, h, w, dev_y);
    //cudaDeviceSynchronize();
    if(debug){
        cudaMemcpy(y, dev_y, n * sizeof(int32_t), cudaMemcpyDeviceToHost);
        cudaFree(tmp_x);
        cudaFree(dev_y);
    }
}

void cuda_reshape(const int32_t *x, int32_t *y, int32_t n, bool debug){
    if(x == y) return;
    if(debug)
		 memcpy(y, x, n * sizeof(int32_t));
    else
        cudaMemcpy(y, x, n*sizeof(int32_t), cudaMemcpyDeviceToDevice);
}
