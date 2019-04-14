#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>

#include <fstream>
#include <iterator>
#include <algorithm>

int dtype_code = kDLInt;
int dtype_bits = 32;
int dtype_lanes = 1;
int device_type = kDLCPU;
int device_id = 0;

void RunCVM(DLTensor* x, TVMByteArray& params_arr, std::string json_data,
        tvm::runtime::Module &mod_syslib  ,  std::string runtime_name, DLTensor *y) {
    // get global function module for graph runtime
    tvm::runtime::Module mod = (*tvm::runtime::Registry::Get("tvm." + runtime_name + ".create"))(json_data, mod_syslib, device_type, device_id);

    // load image data saved in binary
    // std::ifstream data_fin("cat.bin", std::ios::binary);
    // data_fin.read(static_cast<char*>(x->data), 3 * 224 * 224 * 4);

    // get the function from the module(set input data)
    tvm::runtime::PackedFunc set_input = mod.GetFunction("set_input");
    set_input("data", x);


    // get the function from the module(load patameters)
    tvm::runtime::PackedFunc load_params = mod.GetFunction("load_params");
    load_params(params_arr);

    // get the function from the module(run it)
    tvm::runtime::PackedFunc run = mod.GetFunction("run");
    run();


    // get the function from the module(get output data)
    tvm::runtime::PackedFunc get_output = mod.GetFunction("get_output");
    get_output(0, y);

//    auto y_iter = static_cast<int*>(y->data);
//    // get the maximum position in output vector
//    auto max_iter = std::max_element(y_iter, y_iter + out_shape[1]);
//    auto max_index = std::distance(y_iter, max_iter);
//    std::cout << "The maximum position in output vector is: " << max_index << std::endl;

//    for (auto i = 0; i < out_shape[1]; i++) {
//        if (i < 1000)
//            std::cout << y_iter[i] << " ";
//    }
//    std::cout << "\n";
//
//
//    TVMArrayFree(y);
}

int main()
{
    tvm::runtime::Module mod_org = tvm::runtime::Module::LoadFromFile("/tmp/imagenet_llvm.org.so");
    tvm::runtime::Module mod_syslib = (*tvm::runtime::Registry::Get("module._GetSystemLib"))();

    DLTensor* x;
    int in_ndim = 4;
    int64_t in_shape[4] = {1, 3, 224, 224};
    TVMArrayAlloc(in_shape, in_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &x);
    auto x_iter = static_cast<int*>(x->data);
    for (auto i = 0; i < 1 * 3 * 224 * 224; i++) {
        x_iter[i] = (i) % 255;
    }
    std::cout << "\n";

    // parameters in binary
    std::ifstream params_in("/tmp/imagenet_cuda.params", std::ios::binary);
    std::string params_data((std::istreambuf_iterator<char>(params_in)), std::istreambuf_iterator<char>());
    params_in.close();

    // parameters need to be TVMByteArray type to indicate the binary data
    TVMByteArray params_arr;
    params_arr.data = params_data.c_str();
    params_arr.size = params_data.length();

    std::ifstream json_in2("/tmp/imagenet_llvm.org.json", std::ios::in);
    std::string json_data_org((std::istreambuf_iterator<char>(json_in2)), std::istreambuf_iterator<char>());
    json_in2.close();
    // json graph

    DLTensor* y1;
    int out_ndim = 2;
    int64_t out_shape[2] = {1, 1000, };
    TVMArrayAlloc(out_shape, out_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &y1);
    RunCVM(x, params_arr, json_data_org, mod_org, "graph_runtime", y1);

    std::ifstream json_in("/tmp/imagenet_cuda.json", std::ios::in);
    std::string json_data((std::istreambuf_iterator<char>(json_in)), std::istreambuf_iterator<char>());
    json_in.close();

    DLTensor* y2;
    TVMArrayAlloc(out_shape, out_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &y2);
    RunCVM(x, params_arr, json_data, mod_syslib, "cvm_runtime", y2);
    //TVMArrayFree(y_cpu);
    TVMArrayFree(x);

    std::cout << (std::memcmp(y1->data, y2->data, out_shape[1]*sizeof(int32_t)) == 0 ? "pass" : "failed") << std::endl;
    return 0;
}
